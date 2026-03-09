# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
"""cuda.jit_autotune – automatic uint32-transform variant selection.

On the *first* call to a decorated kernel, four variants are compiled and
timed with CUDA events:

  1. ``original``     – no transform
  2. ``range+u32``    – for-range loop kept, loop variable cast to uint32
  3. ``while+u32``    – for → while rewrite with uint32 counter
  4. ``while+u32 x4`` – same as above, manual 4× loop unroll

``hoist=False`` is tried first for every transform; ``hoist=True`` is used
only as a fallback when the transform itself raises an exception.

The winning variant is used for **all subsequent calls** at zero overhead.

Example::

    from numba import cuda

    @cuda.jit_autotune
    def saxpy(a, x, y, n):
        i = cuda.grid(1)
        if i < n:
            y[i] += a * x[i]

    saxpy[blocks, threads](a_d, x_d, y_d, n)   # autotuned on first call
    saxpy[blocks, threads](a_d, x_d, y_d, n)   # winner used directly

Keyword arguments (``debug``, ``opt``, ``fastmath``, …) are forwarded to
``cuda.jit()`` and apply to *all* compiled variants.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import warnings
from typing import Any

from numba.cuda.transform_u64 import transform_source_u64

# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------

#: Default set of variants benchmarked by :func:`jit_autotune`.
DEFAULT_VARIANTS: list[dict] = [
    dict(name="original",     transform=False),
    dict(name="range+u32",    transform=True, mode="range", multistep=True,
         dtype="uint32", unroll=1),
    dict(name="while+u32",    transform=True, mode="while", multistep=True,
         dtype="uint32", unroll=1),
    dict(name="while+u32 x4", transform=True, mode="while", multistep=True,
         dtype="uint32", unroll=4),
]

#: Default number of warm-up launches before timing.
AUTOTUNE_WARMUP: int = 3
#: Default number of timed launches used to compute the mean.
AUTOTUNE_REPS:   int = 10

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_decorators(src: str) -> str:
    """Return *src* with every decorator stripped from all function defs."""
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _apply_transform(clean_src: str, cfg: dict) -> str | None:
    """Transform *clean_src* according to *cfg*.

    Tries ``hoist=False`` first; falls back to ``hoist=True`` on failure.
    Returns ``None`` when both attempts raise.
    """
    if not cfg["transform"]:
        return clean_src
    for hoist in (False, True):
        try:
            return transform_source_u64(
                clean_src,
                mode=cfg["mode"],
                multistep=cfg["multistep"],
                dtype=cfg["dtype"],
                unroll=cfg.get("unroll", 1),
                hoist=hoist,
            )
        except Exception:
            continue
    return None


def _build_dispatcher(src: str, func_name: str, user_globals: dict,
                      jit_kwargs: dict):
    """``exec()`` *src* and wrap the result in a ``cuda.jit`` dispatcher.

    A shallow copy of *user_globals* is used as the exec namespace so that all
    symbols the original function references are available.  ``nb`` and
    ``cuda`` are injected in case the transformed source references them.
    """
    import numba as _nb
    from numba import cuda as _cuda

    g: dict[str, Any] = dict(user_globals)
    g.setdefault("nb", _nb)
    g.setdefault("cuda", _cuda)
    exec(compile(src, "<jit_autotune>", "exec"), g)  # noqa: S102
    fn = g[func_name]
    # Strip keys that are not valid for kernel (non-device) jit
    safe = {k: v for k, v in jit_kwargs.items()
            if k != "device" and v is not None}
    return _cuda.jit(**safe)(fn)


def _bench_variant(disp, argtypes: tuple, griddim, blockdim, stream,
                   sharedmem, args: tuple,
                   warmup: int, reps: int) -> tuple[float | None, str | None]:
    """Compile, warm-up, and time *disp* for *argtypes*.

    Returns ``(ms_per_launch, None)`` on success or ``(None, reason)`` on
    any error.
    """
    from numba import cuda as _cuda

    try:
        disp.compile(argtypes)
    except Exception as exc:
        return None, f"compile error: {exc}"
    disp.disable_compile()

    launcher = disp.configure(griddim, blockdim, stream, sharedmem)
    try:
        for _ in range(warmup):
            launcher(*args)
        _cuda.synchronize()
        t0 = _cuda.event(timing=True)
        t1 = _cuda.event(timing=True)
        t0.record()
        for _ in range(reps):
            launcher(*args)
        t1.record()
        t1.synchronize()
        return _cuda.event_elapsed_time(t0, t1) / reps, None
    except Exception as exc:
        return None, f"launch error: {exc}"


# ---------------------------------------------------------------------------
# Dispatcher classes
# ---------------------------------------------------------------------------


class _AutotuneLaunchConfig:
    """Returned by ``AutotuneDispatcher.__getitem__``.

    Triggers autotuning on the very first ``__call__``, then delegates to
    the winning variant's launch configuration.
    """
    __slots__ = ("_owner", "_griddim", "_blockdim", "_stream", "_sharedmem")

    def __init__(self, owner: "AutotuneDispatcher",
                 griddim, blockdim, stream=0, sharedmem=0) -> None:
        self._owner     = owner
        self._griddim   = griddim
        self._blockdim  = blockdim
        self._stream    = stream
        self._sharedmem = sharedmem

    def __call__(self, *args):
        o = self._owner
        if not o._tuned:
            o._autotune(args, self._griddim, self._blockdim,
                        self._stream, self._sharedmem)
        return o._winner.configure(
            self._griddim, self._blockdim, self._stream, self._sharedmem
        )(*args)


class AutotuneDispatcher:
    """Wraps a CUDA kernel and benchmarks uint32 variants on the first call.

    After the first launch the winning ``CUDADispatcher`` is stored and all
    subsequent calls are routed to it with no extra overhead.

    Attributes
    ----------
    winner : str | None
        Name of the winning variant (available after the first call).
    tuned : bool
        ``True`` after autotuning has completed.
    """

    def __init__(
        self,
        py_func,
        jit_kwargs: dict,
        *,
        variants: list[dict] | None = None,
        warmup:   int  = AUTOTUNE_WARMUP,
        reps:     int  = AUTOTUNE_REPS,
        verbose:  bool = True,
    ) -> None:
        self._py_func    = py_func
        self._jit_kwargs = jit_kwargs
        self._variants   = variants if variants is not None else DEFAULT_VARIANTS
        self._warmup     = warmup
        self._reps       = reps
        self._verbose    = verbose
        self._tuned          = False
        self._winner: Any        = None
        self._winner_name: str | None = None
        self._timing_results: list[tuple[str, float]] = []

        # Extract decorator-free source eagerly while the module context is
        # still available (inspect.getsource fails for exec'd functions).
        try:
            self._clean_src: str | None = _strip_decorators(
                inspect.getsource(py_func)
            )
        except (OSError, TypeError) as exc:
            warnings.warn(
                f"jit_autotune: cannot read source for "
                f"{py_func.__name__!r}: {exc}. "
                "Only the 'original' variant will be benchmarked.",
                stacklevel=3,
            )
            self._clean_src = None

    # ── public interface (mirrors CUDADispatcher) ──────────────────────────

    def __getitem__(self, config):
        """``kernel[griddim, blockdim]`` → launch configuration."""
        if self._tuned:
            return self._winner[config]
        if isinstance(config, tuple):
            griddim   = config[0]
            blockdim  = config[1] if len(config) > 1 else 1
            stream    = config[2] if len(config) > 2 else 0
            sharedmem = config[3] if len(config) > 3 else 0
        else:
            griddim, blockdim, stream, sharedmem = config, 1, 0, 0
        return _AutotuneLaunchConfig(self, griddim, blockdim, stream, sharedmem)

    def configure(self, griddim, blockdim, stream=0, sharedmem=0):
        """``kernel.configure(griddim, blockdim, stream, sharedmem)(*args)``."""
        if self._tuned:
            return self._winner.configure(griddim, blockdim, stream, sharedmem)
        return _AutotuneLaunchConfig(self, griddim, blockdim, stream, sharedmem)

    def __call__(self, *_args, **_kw):
        raise TypeError(
            f"{self._py_func.__name__!r} is a CUDA kernel and must be called "
            "as  kernel[grid, block](*args),  not  kernel(*args)."
        )

    # ── properties ────────────────────────────────────────────────────────

    @property
    def winner(self) -> str | None:
        """Name of the winning variant, or ``None`` if not yet tuned."""
        return self._winner_name

    @property
    def tuned(self) -> bool:
        """``True`` after the first call has triggered autotuning."""
        return self._tuned

    @property
    def timing_results(self) -> list[tuple[str, float]]:
        """List of ``(variant_name, ms)`` for every variant that compiled and
        ran successfully.  Available after the first call; empty before."""
        return self._timing_results

    # ── autotuning core ───────────────────────────────────────────────────

    def _autotune(self, args: tuple, griddim, blockdim, stream,
                  sharedmem) -> None:
        """Benchmark all variants and store the winner."""
        from numba import cuda as _cuda, typeof

        argtypes = tuple(typeof(a) for a in args)
        fname    = self._py_func.__name__
        glo      = self._py_func.__globals__

        if self._verbose:
            print(f"\n[jit_autotune] Tuning {fname!r} …")

        results: list[tuple[str, Any, float]] = []
        base_ms: float | None = None

        for cfg in self._variants:
            vname = cfg["name"]

            # ── build transformed source ──
            if self._clean_src is not None:
                tsrc = _apply_transform(self._clean_src, cfg)
            elif not cfg["transform"]:
                tsrc = None          # use py_func directly (original)
            else:
                if self._verbose:
                    print(f"  {vname:<24} SKIP  (no source for transform)")
                continue

            if tsrc is None and cfg["transform"]:
                if self._verbose:
                    print(f"  {vname:<24} SKIP  (transform raised)")
                continue

            # ── build dispatcher ──
            try:
                if not cfg["transform"]:
                    safe = {k: v for k, v in self._jit_kwargs.items()
                            if k != "device" and v is not None}
                    disp = _cuda.jit(**safe)(self._py_func)
                else:
                    disp = _build_dispatcher(tsrc, fname, glo,
                                             self._jit_kwargs)
            except Exception as exc:
                if self._verbose:
                    print(f"  {vname:<24} SKIP  (build: {exc})")
                continue

            # ── compile & time ──
            ms, err = _bench_variant(
                disp, argtypes, griddim, blockdim, stream, sharedmem,
                args, self._warmup, self._reps,
            )
            if ms is None:
                if self._verbose:
                    print(f"  {vname:<24} SKIP  ({err})")
                continue

            if self._verbose:
                suffix = (f"  ({base_ms / ms:.2f}x)" if base_ms is not None
                          else "")
                print(f"  {vname:<24} {ms:8.4f} ms{suffix}")

            if base_ms is None:
                base_ms = ms
            results.append((vname, disp, ms))

        if not results:
            raise RuntimeError(
                f"jit_autotune: every variant failed for {fname!r}. "
                "Ensure the kernel compiles with cuda.jit()."
            )

        bname, bdisp, bms = min(results, key=lambda t: t[2])
        if self._verbose:
            print(f"  => winner: {bname!r}  ({bms:.4f} ms)\n")

        self._winner        = bdisp
        self._winner_name   = bname
        self._timing_results = [(n, ms) for n, _, ms in results]
        self._tuned         = True


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def jit_autotune(
    func=None,
    *,
    # autotuner knobs
    warmup:   int        = AUTOTUNE_WARMUP,
    reps:     int        = AUTOTUNE_REPS,
    variants: list | None = None,
    verbose:  bool       = True,
    # cuda.jit pass-through options
    debug:          bool | None = None,
    opt:            bool  = True,
    fastmath:       bool  = False,
    lineinfo:       bool  = False,
    link:           tuple = (),
    launch_bounds         = None,
    lto:            bool  = False,
    max_registers:  int | None = None,
    **extra_jit_kwargs,
) -> "AutotuneDispatcher":
    """Decorator – compile & benchmark uint32 variants; use the fastest.

    Can be used with or without parentheses::

        @cuda.jit_autotune
        def kernel(...): ...

        @cuda.jit_autotune(fastmath=True, reps=20)
        def kernel(...): ...

    Parameters
    ----------
    warmup : int
        Warm-up launches before timing (default 3).
    reps : int
        Timed launches per variant (default 10).
    variants : list[dict] | None
        Override :data:`DEFAULT_VARIANTS`.  Each dict must contain at least
        ``name`` and ``transform`` keys, and – when ``transform=True`` –
        ``mode``, ``multistep``, ``dtype``, and optionally ``unroll``.
    verbose : bool
        Print per-variant timing and the chosen winner.
    debug, opt, fastmath, lineinfo, link, launch_bounds, lto, max_registers :
        Forwarded verbatim to ``cuda.jit()``.
    """
    jit_kwargs: dict = dict(
        debug=debug, opt=opt, fastmath=fastmath, lineinfo=lineinfo,
        link=link, launch_bounds=launch_bounds, lto=lto,
    )
    if max_registers is not None:
        jit_kwargs["max_registers"] = max_registers
    jit_kwargs.update(extra_jit_kwargs)

    def _wrap(py_func) -> AutotuneDispatcher:
        return AutotuneDispatcher(
            py_func,
            jit_kwargs=jit_kwargs,
            variants=variants,
            warmup=warmup,
            reps=reps,
            verbose=verbose,
        )

    if func is not None:          # @jit_autotune  (no parentheses)
        return _wrap(func)
    return _wrap                  # @jit_autotune(...)  (with parentheses)

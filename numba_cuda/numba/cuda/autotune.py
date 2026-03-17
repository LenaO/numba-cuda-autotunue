# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
"""cuda.jit_autotune – automatic uint32-transform variant selection with caching.

On the *first* call to a decorated kernel, four variants are compiled and
timed with CUDA events:

  1. ``original``     – no transform
  2. ``range+u32``    – for-range loop kept, loop variable cast to uint32
  3. ``while+u32``    – for → while rewrite with uint32 counter
  4. ``while+u32 x4`` – same as above, manual 4× loop unroll

``hoist=False`` is tried first for every transform; ``hoist=True`` is used
only as a fallback when the transform itself raises an exception.

The winning variant is used for **all subsequent calls** at zero overhead.

Caching
-------
Pass ``cache=True`` (or a directory path) to persist the tuning decision to
disk.  The cache key combines:

* An MD5 fingerprint of the argument types (ensures type-correct reuse).
* An **occupancy bucket**: ``floor(log2(total_threads / GPU_capacity))``,
  clamped to ``[-3, 6]``.  This coarsely captures the utilization regime
  (under-occupied vs. fully occupied) that drives which variant wins.
* A fingerprint of the stripped kernel source (cache auto-invalidates when
  the kernel body changes).

On a cache *hit* only the winning variant is compiled (saving 1-3 compilation
steps), but benchmarking is skipped entirely.

Example::

    from numba import cuda

    @cuda.jit_autotune(cache=True)
    def spmv(data, indices, indptr, x, y, n):
        ...

    spmv[blocks, tpb](...)   # tunes on first call, writes cache
    # restart program
    spmv[blocks, tpb](...)   # cache hit: compiles winner only, no benchmark
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import textwrap
import warnings
from datetime import datetime
from pathlib import Path
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
# Cache helpers
# ---------------------------------------------------------------------------


def _argtypes_fingerprint(argtypes: tuple) -> str:
    """12-hex-char MD5 of the string representation of *argtypes*."""
    return hashlib.md5(str(argtypes).encode()).hexdigest()[:12]


def _occupancy_bucket(griddim, blockdim) -> int:
    """Return an integer bucket in ``[-3, 6]`` representing GPU utilisation.

    Bucket 0  ≈ exactly one SM-wave of threads.
    Negative  = under-occupied; positive = over-occupied (many waves).
    """
    from numba import cuda as _cuda

    def _prod(x) -> int:
        if isinstance(x, int):
            return x
        r = 1
        for v in x:
            r *= v
        return r

    total = _prod(griddim) * _prod(blockdim)
    device = _cuda.get_current_device()
    # 2048 threads/SM is the Ada/Ampere/Hopper architectural maximum
    capacity = device.MULTIPROCESSOR_COUNT * 2048
    ratio = total / max(capacity, 1)
    return max(-3, min(6, int(math.floor(math.log2(max(ratio, 1e-9))))))


class CacheManager:
    """Persist and retrieve autotune decisions for one kernel on one device.

    The cache file is a human-readable JSON document stored at::

        <cache_dir>/<func_name>_<device_id8>.json

    The root-level ``source_hash`` field holds an MD5 of the stripped kernel
    source; the entire file is discarded and rebuilt whenever the source
    changes.

    Each entry is keyed by ``"<argtypes_fp>|<bucket>"`` and stores the
    winner name, full timing table, an example launch config, and a
    timestamp.
    """

    def __init__(self, func_name: str, clean_src: str | None,
                 cache_dir: str | Path = ".autotune_cache") -> None:
        from numba import cuda as _cuda

        self._func_name   = func_name
        self._source_hash = (
            hashlib.md5(clean_src.encode()).hexdigest()[:16]
            if clean_src else ""
        )

        device = _cuda.get_current_device()
        device_name = device.name.decode() if isinstance(device.name, bytes) \
                      else str(device.name)
        self._device_name = device_name
        # Stable 8-char device identifier derived from the device name
        self._device_id   = hashlib.md5(device_name.encode()).hexdigest()[:8]
        cc                = device.compute_capability
        self._cc          = f"{cc[0]}.{cc[1]}"

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = cache_dir / f"{func_name}_{self._device_id}.json"
        self._data = self._load()

    # ── I/O ───────────────────────────────────────────────────────────────

    def _fresh(self) -> dict:
        return {
            "func":        self._func_name,
            "source_hash": self._source_hash,
            "device":      self._device_name,
            "cc":          self._cc,
            "entries":     {},
        }

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                # Invalidate cache when kernel source has changed
                if data.get("source_hash") != self._source_hash:
                    warnings.warn(
                        f"jit_autotune: source of {self._func_name!r} changed "
                        f"– discarding cache {self._path}",
                        stacklevel=5,
                    )
                    return self._fresh()
                return data
            except Exception as exc:
                warnings.warn(
                    f"jit_autotune: could not read cache {self._path}: {exc}",
                    stacklevel=5,
                )
        return self._fresh()

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            warnings.warn(
                f"jit_autotune: could not write cache {self._path}: {exc}",
                stacklevel=5,
            )

    # ── public API ────────────────────────────────────────────────────────

    def lookup(self, argtypes_fp: str, bucket: int) -> dict | None:
        """Return the cached entry for *(argtypes_fp, bucket)*, or ``None``."""
        return self._data["entries"].get(f"{argtypes_fp}|{bucket}")

    def store(self, argtypes_fp: str, bucket: int, winner: str,
              timing_ms: list[tuple[str, float]], griddim, blockdim) -> None:
        """Persist a new tuning result and flush to disk."""
        def _to_list(x):
            return list(x) if hasattr(x, "__iter__") else [x]

        key = f"{argtypes_fp}|{bucket}"
        self._data["entries"][key] = {
            "winner":          winner,
            "timing_ms":       [[n, ms] for n, ms in timing_ms],
            "griddim_example": _to_list(griddim),
            "blockdim":        _to_list(blockdim),
            "tuned_at":        datetime.now().isoformat(),
        }
        self._save()

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Kernel-building helpers
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
    """``exec()`` *src* and wrap the result in a ``cuda.jit`` dispatcher."""
    import numba as _nb
    from numba import cuda as _cuda

    g: dict[str, Any] = dict(user_globals)
    g.setdefault("nb", _nb)
    g.setdefault("cuda", _cuda)
    exec(compile(src, "<jit_autotune>", "exec"), g)  # noqa: S102
    fn = g[func_name]
    safe = {k: v for k, v in jit_kwargs.items()
            if k != "device" and v is not None}
    return _cuda.jit(**safe)(fn)


def _has_range_loop(src: str) -> bool:
    """Return True if *src* contains at least one ``range(`` call.

    Used to skip while-loop variants for kernels that have no for-range loops
    (the while transform only makes sense when there is a range to rewrite).
    A simple text scan is sufficient – we are looking for the token ``range(``.
    """
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True  # be conservative: try all variants if parse fails
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == "range") or (
                isinstance(func, ast.Attribute) and func.attr == "range"
            ):
                return True
    return False


def _copy_args(args: tuple) -> tuple:
    """Return a copy of *args* where every device/host array is duplicated.

    Scalars are passed through unchanged.  This ensures that benchmarking
    transformed variants never corrupts live application data.

    Numba ``DeviceNDArray`` objects do NOT have a ``.copy()`` method; they
    must be copied via ``copy_to_host()`` + ``cuda.to_device()``.
    """
    from numba import cuda as _cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray as _DeviceNDArray
    import numpy as _np
    result = []
    for a in args:
        if isinstance(a, _DeviceNDArray):
            # D2H then H2D: reliable deep copy independent of CUDA stream state
            result.append(_cuda.to_device(a.copy_to_host()))
        elif isinstance(a, _np.ndarray):
            result.append(a.copy())
        else:
            result.append(a)
    return tuple(result)


class _FatalCUDAError(RuntimeError):
    """Raised when a CUDA context-fatal error occurs during benchmarking.

    Unlike ordinary launch failures (wrong args, compile errors, etc.),
    context-fatal errors (ILLEGAL_ADDRESS, CONTEXT_IS_DESTROYED, …) make
    every subsequent CUDA call fail.  Callers must stop benchmarking and
    not attempt any further CUDA work in the same process.
    """


_FATAL_CUDA_SUBSTRINGS = (
    "ILLEGAL_ADDRESS",
    "CONTEXT_IS_DESTROYED",
    "ILLEGAL_INSTRUCTION",
    "MISALIGNED_ADDRESS",
)


def _is_fatal_cuda_error(exc: BaseException) -> bool:
    msg = str(exc)
    return any(s in msg for s in _FATAL_CUDA_SUBSTRINGS)


def _bench_variant(disp, argtypes: tuple, griddim, blockdim, stream,
                   sharedmem, args: tuple,
                   warmup: int, reps: int) -> tuple[float | None, str | None]:
    """Compile, warm-up, and time *disp*.  Returns ``(ms, None)`` or
    ``(None, reason)``.

    Raises :class:`_FatalCUDAError` if the CUDA context is poisoned by the
    variant under test so the caller can abort benchmarking immediately.
    """
    from numba import cuda as _cuda

    try:
        disp.compile(argtypes)
    except Exception as exc:
        if _is_fatal_cuda_error(exc):
            raise _FatalCUDAError(str(exc)) from exc
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
        if _is_fatal_cuda_error(exc):
            raise _FatalCUDAError(str(exc)) from exc
        return None, f"launch error: {exc}"


# ---------------------------------------------------------------------------
# Dispatcher classes
# ---------------------------------------------------------------------------


class _AutotuneLaunchConfig:
    """Returned by ``AutotuneDispatcher.__getitem__``.

    Triggers autotuning (or a cache-assisted fast path) on the very first
    ``__call__``, then delegates to the winning variant's launch configuration.
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
    """Wraps a CUDA kernel; benchmarks uint32 variants on first call and
    uses the fastest for every subsequent call.

    When *cache* is enabled, the winning variant for each
    ``(argtypes, occupancy_bucket)`` combination is saved to a JSON file and
    reloaded on the next run, skipping benchmarking entirely.

    Attributes
    ----------
    winner : str | None
        Name of the winning variant (available after the first call).
    tuned : bool
        ``True`` after autotuning (or a cache hit) has completed.
    timing_results : list[tuple[str, float]]
        Per-variant ``(name, ms)`` from the last full benchmark run.
        Empty on a cache hit where benchmarking was skipped.
    cache_path : Path | None
        Path to the JSON cache file, or ``None`` if caching is disabled.
    """

    def __init__(
        self,
        py_func,
        jit_kwargs: dict,
        *,
        variants:        list[dict] | None = None,
        warmup:          int  = AUTOTUNE_WARMUP,
        reps:            int  = AUTOTUNE_REPS,
        verbose:         bool = True,
        cache:           bool | str | Path = False,
        bench_with_copy: bool = False,
        source:          str | None = None,
    ) -> None:
        self._py_func        = py_func
        self._jit_kwargs     = jit_kwargs
        self._variants       = variants if variants is not None else DEFAULT_VARIANTS
        self._warmup         = warmup
        self._reps           = reps
        self._verbose        = verbose
        self._cache_arg      = cache      # stored; CacheManager created lazily
        self._bench_with_copy = bench_with_copy

        self._tuned                               = False
        self._winner: Any                         = None
        self._winner_name: str | None             = None
        self._timing_results: list[tuple[str, float]] = []
        self._cache_mgr: CacheManager | None      = None

        # Extract decorator-free source eagerly while the module context is
        # still available (inspect.getsource fails for exec'd functions).
        # A caller may supply the source directly via the *source* parameter
        # (e.g. when the function was created via exec()).
        if source is not None:
            self._clean_src: str | None = _strip_decorators(source)
        else:
            try:
                self._clean_src = _strip_decorators(inspect.getsource(py_func))
            except (OSError, TypeError) as exc:
                warnings.warn(
                    f"jit_autotune: cannot read source for "
                    f"{py_func.__name__!r}: {exc}. "
                    "Only the 'original' variant will be benchmarked.",
                    stacklevel=3,
                )
                self._clean_src = None

        # Drop while-loop variants when the kernel has no range() call:
        # the while transform only rewrites range-based loops, so it would
        # produce an identical (or broken) kernel for range-free code.
        if self._clean_src is not None and not _has_range_loop(self._clean_src):
            self._variants = [
                v for v in self._variants
                if not (v.get("transform") and v.get("mode") == "while")
            ]

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
        return self._winner_name

    @property
    def tuned(self) -> bool:
        return self._tuned

    @property
    def timing_results(self) -> list[tuple[str, float]]:
        """Per-variant ``(name, ms)`` from the last benchmark. Empty on cache
        hit (benchmarking was skipped)."""
        return self._timing_results

    @property
    def cache_path(self) -> Path | None:
        return self._cache_mgr.path if self._cache_mgr else None

    # ── internal ──────────────────────────────────────────────────────────

    def _get_cache_mgr(self) -> CacheManager | None:
        """Create CacheManager on first use (CUDA must already be initialised)."""
        if self._cache_mgr is not None or not self._cache_arg:
            return self._cache_mgr
        cache_dir = (
            ".autotune_cache"
            if self._cache_arg is True
            else self._cache_arg
        )
        try:
            self._cache_mgr = CacheManager(
                self._py_func.__name__, self._clean_src, cache_dir
            )
        except Exception as exc:
            warnings.warn(f"jit_autotune: cache disabled ({exc})", stacklevel=4)
        return self._cache_mgr

    def _compile_variant(self, vname: str, fname: str,
                         glo: dict, argtypes: tuple):
        """Build + compile a single named variant.  Returns the dispatcher or
        ``None`` on any error."""
        from numba import cuda as _cuda

        cfg = next((c for c in self._variants if c["name"] == vname), None)
        if cfg is None:
            return None

        if self._clean_src is not None:
            tsrc = _apply_transform(self._clean_src, cfg)
        elif not cfg.get("transform"):
            tsrc = None
        else:
            return None

        if tsrc is None and cfg.get("transform"):
            return None

        try:
            if not cfg["transform"]:
                safe = {k: v for k, v in self._jit_kwargs.items()
                        if k != "device" and v is not None}
                disp = _cuda.jit(**safe)(self._py_func)
            else:
                disp = _build_dispatcher(tsrc, fname, glo, self._jit_kwargs)
            disp.compile(argtypes)
            disp.disable_compile()
            return disp
        except Exception:
            return None

    # ── autotuning core ───────────────────────────────────────────────────

    def _autotune(self, args: tuple, griddim, blockdim, stream,
                  sharedmem) -> None:
        from numba import cuda as _cuda, typeof

        argtypes = tuple(typeof(a) for a in args)
        fname    = self._py_func.__name__
        glo      = self._py_func.__globals__

        # Optionally benchmark on copies so live data is never corrupted
        bench_args = _copy_args(args) if self._bench_with_copy else args

        # ── cache lookup ─────────────────────────────────────────────────
        mgr    = self._get_cache_mgr()
        afp    = _argtypes_fingerprint(argtypes)
        bucket = _occupancy_bucket(griddim, blockdim) if mgr else 0

        if mgr is not None:
            cached = mgr.lookup(afp, bucket)
            if cached is not None:
                vname = cached["winner"]
                if self._verbose:
                    print(f"\n[jit_autotune] Cache hit for {fname!r}  "
                          f"(bucket={bucket:+d}, winner={vname!r})")
                disp = self._compile_variant(vname, fname, glo, argtypes)
                if disp is not None:
                    self._winner        = disp
                    self._winner_name   = vname
                    self._timing_results = [
                        (r[0], r[1]) for r in cached.get("timing_ms", [])
                    ]
                    self._tuned = True
                    return
                # Winner failed to rebuild – fall through to full retune
                if self._verbose:
                    print(f"  (cached winner {vname!r} failed to rebuild "
                          f"– retuning)")

        # ── full benchmark ────────────────────────────────────────────────
        if self._verbose:
            print(f"\n[jit_autotune] Tuning {fname!r}  (bucket={bucket:+d}) ...")

        results: list[tuple[str, Any, float]] = []
        base_ms: float | None = None
        _fatal_error: _FatalCUDAError | None = None

        for cfg in self._variants:
            vname = cfg["name"]

            if self._clean_src is not None:
                tsrc = _apply_transform(self._clean_src, cfg)
            elif not cfg["transform"]:
                tsrc = None
            else:
                if self._verbose:
                    print(f"  {vname:<24} SKIP  (no source)")
                continue

            if tsrc is None and cfg["transform"]:
                if self._verbose:
                    print(f"  {vname:<24} SKIP  (transform raised)")
                continue

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

            try:
                ms, err = _bench_variant(
                    disp, argtypes, griddim, blockdim, stream, sharedmem,
                    bench_args, self._warmup, self._reps,
                )
            except _FatalCUDAError as exc:
                if self._verbose:
                    print(f"  {vname:<24} ABORT (fatal CUDA error: {exc})")
                    print(f"  Stopping benchmark – CUDA context is poisoned.")
                _fatal_error = exc
                break  # stop trying more variants; use whatever we have so far
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
            if _fatal_error is not None:
                raise RuntimeError(
                    f"jit_autotune: CUDA context poisoned before any variant "
                    f"of {fname!r} could be timed. Restart the process."
                ) from _fatal_error
            raise RuntimeError(
                f"jit_autotune: every variant failed for {fname!r}."
            )

        bname, bdisp, bms = min(results, key=lambda t: t[2])
        timing = [(n, ms) for n, _, ms in results]

        if self._verbose:
            print(f"  => winner: {bname!r}  ({bms:.4f} ms)\n")

        # ── cache store ───────────────────────────────────────────────────
        if mgr is not None:
            mgr.store(afp, bucket, bname, timing, griddim, blockdim)
            if self._verbose:
                print(f"  [cache] saved to {mgr.path}")

        self._winner         = bdisp
        self._winner_name    = bname
        self._timing_results = timing
        self._tuned          = True

        if _fatal_error is not None:
            raise RuntimeError(
                f"jit_autotune: CUDA context poisoned while benchmarking "
                f"{fname!r} (winner {bname!r} cached for next run). "
                f"Restart the process to continue."
            ) from _fatal_error


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
    cache:    bool | str | Path = False,
    bench_with_copy: bool = False,
    source: str | None = None,
    # cuda.jit pass-through options
    debug:         bool | None = None,
    opt:           bool  = True,
    fastmath:      bool  = False,
    lineinfo:      bool  = False,
    link:          tuple = (),
    launch_bounds        = None,
    lto:           bool  = False,
    max_registers: int | None = None,
    **extra_jit_kwargs,
) -> "AutotuneDispatcher":
    """Decorator – compile & benchmark uint32 variants; use the fastest.

    Can be used with or without parentheses::

        @cuda.jit_autotune
        def kernel(...): ...

        @cuda.jit_autotune(cache=True, fastmath=True, reps=20)
        def kernel(...): ...

    Parameters
    ----------
    warmup : int
        Warm-up launches before timing (default 3).
    reps : int
        Timed launches per variant (default 10).
    variants : list[dict] | None
        Override :data:`DEFAULT_VARIANTS`.
    verbose : bool
        Print per-variant timing and the chosen winner.
    cache : bool | str | Path
        ``False``  – no caching (default).
        ``True``   – cache in ``.autotune_cache/`` next to the working dir.
        ``"path"`` – cache in the given directory.
    bench_with_copy : bool
        If ``True``, device arrays are copied before benchmarking each variant
        so that live simulation data is not corrupted by transformed kernels.
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
            cache=cache,
            bench_with_copy=bench_with_copy,
            source=source,
        )

    if func is not None:          # @jit_autotune  (no parentheses)
        return _wrap(func)
    return _wrap                  # @jit_autotune(...)  (with parentheses)

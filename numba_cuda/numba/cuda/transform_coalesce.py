#!/usr/bin/env python3
"""Detect and fix non-coalesced array access patterns in Numba CUDA kernels.

In C-order (row-major) NumPy arrays the **last** index is contiguous in memory.
CUDA warps are formed along ``threadIdx.x``, so consecutive warp threads differ
only in the x-component of ``cuda.grid()``.  For coalesced global memory access
the x-thread variable must therefore map to the **last** array subscript index.

A common mistake is writing ``array[x, y]`` or ``array[ii, jj, kk]`` where
``ii`` comes from the x-dimension of ``cuda.grid()`` — this causes each warp
thread to access memory ``Ny * Nz`` elements apart instead of 1 element apart.

This module provides:

* **Analysis** (``analyse_kernel``): returns a list of suspicious array
  accesses without modifying the source.
* **Transformation** (``transform_coalesce``): rewrites every multi-dimensional
  subscript so that the x-thread-derived index becomes the **last** subscript.
  Callers must transpose the corresponding host arrays to match.

Two common patterns are recognised:

1. Direct unpack::

       col, row = cuda.grid(2)          # col is x-thread (fast)
       B[col, row] = ...                # -> B[row, col]

2. Grid-stride range loop::

       global_idx, global_idy = cuda.grid(2)
       for ii in range(global_idx, Nx, gridDimX):   # ii derived from x-thread
           for jj in range(global_idy, Ny, gridDimY):
               B[ii, jj] = ...          # -> B[jj, ii]

Usage (CLI)::

    python transform_coalesce.py kernel.py            # analyse only
    python transform_coalesce.py kernel.py --fix      # rewrite + print
    python transform_coalesce.py kernel.py --fix out.py  # write to file
"""
from __future__ import annotations

import ast
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class AccessIssue(NamedTuple):
    lineno: int
    array_name: str
    original_indices: list[str]
    fixed_indices: list[str]
    x_var: str          # the variable flagged as x-thread derived


class LaunchIssue(NamedTuple):
    lineno: int
    kernel_name: str
    original_grid: str   # e.g. "(Nx, Ny, Nz)"
    fixed_grid: str      # e.g. "(Nz, Ny, Nx)"
    original_block: str  # e.g. "(2, 2, 32)"
    fixed_block: str     # e.g. "(32, 2, 2)"  — x gets the largest value
    ndim: int
    block_was_literal: bool  # False when block is a variable


@dataclass
class _ThreadVarCollector(ast.NodeVisitor):
    """Forward data-flow pass: collect variables derived from the x-thread.

    A variable is classified as *x-thread derived* if it is:
    - directly assigned from ``cuda.grid(1)``
    - the FIRST element of a tuple assigned from ``cuda.grid(N)``
    - the FIRST element of a tuple assigned from ``cuda.gridsize(N)``
      (gridsize x = total threads in x = also the stride for x-loops)
    - the loop variable of ``for V in range(x_var, ...)``
    - assigned from an expression whose *only* non-constant operand is
      already an x-thread variable (propagation through + and *)
    """
    x_vars: set[str] = field(default_factory=set)
    y_vars: set[str] = field(default_factory=set)
    z_vars: set[str] = field(default_factory=set)

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _is_grid_or_gridsize(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("grid", "gridsize")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cuda"
        )

    def _expr_is_x_derived(self, node: ast.expr) -> bool:
        """Return True if *node* is provably x-thread derived."""
        if isinstance(node, ast.Name):
            return node.id in self.x_vars
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            return self._expr_is_x_derived(node.left) or self._expr_is_x_derived(node.right)
        # cuda.grid(1) scalar form
        if self._is_grid_or_gridsize(node):
            return True
        return False

    # ---- visitors ----------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if not node.targets:
            self.generic_visit(node)
            return
        target = node.targets[0]

        # col, row = cuda.grid(2)  or  gx, gy, gz = cuda.gridsize(3)
        if isinstance(target, ast.Tuple) and self._is_grid_or_gridsize(node.value):
            elts = target.elts
            var_lists = [self.x_vars, self.y_vars, self.z_vars]
            for i, elt in enumerate(elts):
                if isinstance(elt, ast.Name) and i < len(var_lists):
                    var_lists[i].add(elt.id)
            self.generic_visit(node)
            return

        # scalar: global_id = cuda.grid(1)
        if isinstance(target, ast.Name):
            if self._is_grid_or_gridsize(node.value):
                self.x_vars.add(target.id)
            elif self._expr_is_x_derived(node.value):
                self.x_vars.add(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # for VAR in range(x_var, STOP, STEP) → VAR is x-thread derived
        if (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            args = node.iter.args
            if args and self._expr_is_x_derived(args[0]):
                self.x_vars.add(node.target.id)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

def _collect_thread_vars(func_node: ast.FunctionDef) -> _ThreadVarCollector:
    col = _ThreadVarCollector()
    col.visit(func_node)
    return col


def _subscript_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id
    return None


def _index_repr(node: ast.expr) -> str:
    return ast.unparse(node)


def _indices_of(node: ast.Subscript) -> list[ast.expr]:
    sl = node.slice
    if isinstance(sl, ast.Tuple):
        return list(sl.elts)
    return [sl]


def _x_var_position(indices: list[ast.expr], x_vars: set[str]) -> int | None:
    """Return the index position that is x-thread derived, or None."""
    for i, idx in enumerate(indices):
        names = {n.id for n in ast.walk(idx) if isinstance(n, ast.Name)}
        if names & x_vars:
            return i
    return None


def analyse_kernel(source: str) -> dict[str, list[AccessIssue]]:
    """Return a mapping from function-name to list of suspicious accesses.

    Does not modify *source*.
    """
    tree = ast.parse(textwrap.dedent(source))
    results: dict[str, list[AccessIssue]] = {}

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        tvars = _collect_thread_vars(func)
        if not tvars.x_vars:
            continue   # no CUDA thread variables found

        issues: list[AccessIssue] = []
        for node in ast.walk(func):
            if not isinstance(node, ast.Subscript):
                continue
            arr_name = _subscript_name(node)
            if arr_name is None:
                continue
            indices = _indices_of(node)
            if len(indices) < 2:
                continue  # 1-D: always coalesced

            x_pos = _x_var_position(indices, tvars.x_vars)
            if x_pos is None:
                continue  # no x-thread variable in subscript
            last_pos = len(indices) - 1
            if x_pos == last_pos:
                continue  # already in the correct (last) position

            # Find the x-thread variable name for reporting
            x_var = next(
                n.id for idx in (indices[x_pos],)
                for n in ast.walk(idx) if isinstance(n, ast.Name) and n.id in tvars.x_vars
            )

            # Build fixed index list: move x-position element to last
            fixed = list(indices)
            fixed.append(fixed.pop(x_pos))

            issues.append(AccessIssue(
                lineno=node.lineno,
                array_name=arr_name,
                original_indices=[_index_repr(i) for i in indices],
                fixed_indices=[_index_repr(i) for i in fixed],
                x_var=x_var,
            ))

        if issues:
            results[func.name] = issues

    return results


# ---------------------------------------------------------------------------
# AST transformer
# ---------------------------------------------------------------------------

class _CoalesceTransformer(ast.NodeTransformer):
    """Rewrite array subscripts so that x-thread indices move to the last slot."""

    def __init__(self, x_vars: set[str]) -> None:
        self._x_vars = x_vars

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        self.generic_visit(node)  # recurse first

        indices = _indices_of(node)
        if len(indices) < 2:
            return node

        x_pos = _x_var_position(indices, self._x_vars)
        if x_pos is None or x_pos == len(indices) - 1:
            return node  # already correct

        # Move the x-thread index to last position
        new_indices = list(indices)
        new_indices.append(new_indices.pop(x_pos))

        new_slice = (
            ast.Tuple(elts=new_indices, ctx=ast.Load())
            if len(new_indices) > 1
            else new_indices[0]
        )
        return ast.Subscript(value=node.value, slice=new_slice, ctx=node.ctx)


def _kernel_name(node: ast.expr) -> str | None:
    """Return the kernel function name from a launch expression, or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _grid_tuple(node: ast.expr) -> ast.Tuple | None:
    """Return the grid tuple node from a ``kernel[(grid, block)]`` subscript."""
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if not isinstance(sl, ast.Tuple) or len(sl.elts) < 2:
        return None
    grid = sl.elts[0]
    if isinstance(grid, ast.Tuple) and len(grid.elts) in (2, 3):
        return grid
    return None


def _block_tuple(node: ast.expr) -> ast.Tuple | None:
    """Return the block tuple node from a ``kernel[(grid, block)]`` subscript."""
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if not isinstance(sl, ast.Tuple) or len(sl.elts) < 2:
        return None
    block = sl.elts[1]
    if isinstance(block, ast.Tuple) and len(block.elts) in (2, 3):
        return block
    return None


def _swap_block_for_coalescing(block: ast.Tuple) -> ast.Tuple | None:
    """Rearrange block dims so that the largest value goes to x (threadIdx.x).

    CUDA warp threads are consecutive in threadIdx.x.  For coalesced access the
    x-dimension of the block must be ≥ 32 (preferably exactly 32) so that a
    full warp maps to consecutive thread IDs with the same (y, z) coordinates.

    If the block is already largest in x, or contains no integer literals,
    returns None (no change needed / cannot determine statically).
    """
    elts = block.elts
    # Only handle all-literal blocks (most common in benchmarks / kernels)
    vals: list[int] = []
    for e in elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, int):
            vals.append(e.value)
        else:
            return None   # non-literal element

    max_val = max(vals)
    if vals[0] == max_val:
        return None   # x already has the largest value

    # Move the largest value to the x position; keep the rest in order
    max_idx = vals.index(max_val)
    new_vals = list(vals)
    new_vals[0], new_vals[max_idx] = new_vals[max_idx], new_vals[0]
    new_elts = [ast.Constant(value=v) for v in new_vals]
    return ast.Tuple(elts=new_elts, ctx=ast.Load())


def analyse_launches(source: str,
                     coalescing_funcs: set[str]) -> list[LaunchIssue]:
    """Find launch sites for *coalescing_funcs* where grid or block is wrong.

    Two issues are detected:

    1. **Grid**: x-grid covers the slow (first) array dimension.
       Fix: swap first and last grid elements so x covers the fast (last) dim.

    2. **Block**: the largest thread count is not in the x-dimension.
       CUDA warp threads are consecutive in threadIdx.x, so ``blockDim.x``
       must be ≥ 32 for a full warp to form along x and get coalesced access.
       A common mistake is ``block=(2,2,32)`` — the 32 threads in z do NOT
       form a contiguous warp in x.
       Fix: move the largest block dimension to x.
    """
    tree = ast.parse(textwrap.dedent(source))
    issues: list[LaunchIssue] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_expr = node.func
        if not isinstance(func_expr, ast.Subscript):
            continue
        name = _kernel_name(func_expr.value)
        if name not in coalescing_funcs:
            continue

        grid  = _grid_tuple(func_expr)
        block = _block_tuple(func_expr)

        if grid is None:
            orig_grid  = "<variable>"
            fixed_grid = "<swap first and last elements manually>"
            ndim = 0
        else:
            elts = list(grid.elts)
            ndim = len(elts)
            elts[0], elts[-1] = elts[-1], elts[0]
            orig_grid  = ast.unparse(grid)
            fixed_grid = ast.unparse(ast.Tuple(elts=elts, ctx=ast.Load()))

        if block is None:
            orig_block   = "<variable>"
            fixed_block  = "<move largest dim to x manually>"
            block_literal = False
        else:
            new_block = _swap_block_for_coalescing(block)
            orig_block   = ast.unparse(block)
            if new_block is None:
                fixed_block  = orig_block   # already correct
            else:
                fixed_block  = ast.unparse(new_block)
            block_literal = True

        # Only emit an issue if something actually needs changing
        grid_changed  = (orig_grid != fixed_grid)
        block_changed = block_literal and (orig_block != fixed_block)
        if not grid_changed and not block_changed:
            continue

        issues.append(LaunchIssue(
            lineno=node.lineno,
            kernel_name=name,
            original_grid=orig_grid,
            fixed_grid=fixed_grid,
            original_block=orig_block,
            fixed_block=fixed_block,
            ndim=ndim,
            block_was_literal=block_literal,
        ))

    return issues


class _LaunchGridFixer(ast.NodeTransformer):
    """Swap first↔last grid dims and move largest block dim to x."""

    def __init__(self, coalescing_funcs: set[str]) -> None:
        self._funcs = coalescing_funcs

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Subscript):
            return node
        name = _kernel_name(node.func.value)
        if name not in self._funcs:
            return node

        sl = node.func.slice
        if not isinstance(sl, ast.Tuple) or len(sl.elts) < 2:
            return node

        new_elts = list(sl.elts)

        # Fix grid: swap first ↔ last
        grid = _grid_tuple(node.func)
        if grid is not None:
            g_elts = list(grid.elts)
            if len(g_elts) >= 2 and ast.dump(g_elts[0]) != ast.dump(g_elts[-1]):
                g_elts[0], g_elts[-1] = g_elts[-1], g_elts[0]
                new_elts[0] = ast.Tuple(elts=g_elts, ctx=ast.Load())

        # Fix block: move largest literal dim to x
        block = _block_tuple(node.func)
        if block is not None:
            new_block = _swap_block_for_coalescing(block)
            if new_block is not None:
                new_elts[1] = new_block

        node.func = ast.Subscript(
            value=node.func.value,
            slice=ast.Tuple(elts=new_elts, ctx=sl.ctx),
            ctx=node.func.ctx,
        )
        return node


def transform_coalesce(source: str) -> str:
    """Return *source* with non-coalesced subscripts **and** launch grids rewritten.

    For every kernel function that contains ``cuda.grid()`` assignments the
    transformer:

    1. Moves the x-thread-derived index to the **last** position in every
       multi-dimensional array subscript.
    2. Swaps the first and last elements of the grid tuple in every
       ``kernel[(grid, block)](...)`` call to that kernel, so that the
       x-dimension of the grid covers the now-last (fast) array dimension.

    .. warning::
        This only fixes the kernel source and call sites in the **same file**.
        The host code must also transpose (or allocate) every modified array
        so that the formerly-first dimension is now the last dimension.
        Call sites in other files must be updated manually.
    """
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # Pass 1: fix kernel subscripts and collect affected function names
    coalescing_funcs: set[str] = set()
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tvars = _collect_thread_vars(func)
        if not tvars.x_vars:
            continue
        # Check if there are any issues before marking as affected
        has_issue = any(
            len(_indices_of(n)) >= 2
            and _x_var_position(_indices_of(n), tvars.x_vars) not in (None, len(_indices_of(n)) - 1)
            for n in ast.walk(func)
            if isinstance(n, ast.Subscript) and _subscript_name(n) is not None
        )
        if has_issue:
            coalescing_funcs.add(func.name)
        _CoalesceTransformer(tvars.x_vars).visit(func)

    # Pass 2: fix launch sites in the same file
    if coalescing_funcs:
        _LaunchGridFixer(coalescing_funcs).visit(tree)

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(access_results: dict[str, list[AccessIssue]],
                  launch_results: list[LaunchIssue],
                  source: str) -> None:
    lines = source.splitlines()

    n_access  = sum(len(v) for v in access_results.values())
    n_unique  = sum(
        len({i.array_name for i in v}) for v in access_results.values()
    )
    n_launches = len(launch_results)

    if not n_access and not n_launches:
        print("No non-coalesced array accesses or launch configurations detected.")
        return

    if n_access:
        # Deduplicate: report each (array_name, original_indices) once per function
        print(f"Found non-coalesced array accesses in "
              f"{len(access_results)} kernel(s)  "
              f"({n_unique} distinct array(s)):\n")
        for fname, issues in access_results.items():
            # Show one representative issue per (array, subscript shape)
            seen: set[tuple] = set()
            print(f"  Kernel: {fname!r}")
            for iss in issues:
                key = (iss.array_name, tuple(iss.original_indices))
                if key in seen:
                    continue
                seen.add(key)
                orig  = ", ".join(iss.original_indices)
                fixed = ", ".join(iss.fixed_indices)
                src_line = (lines[iss.lineno - 1].strip()
                            if iss.lineno <= len(lines) else "")
                print(f"    line {iss.lineno:4d}  {iss.array_name}[{orig}]")
                print(f"             x-thread: {iss.x_var!r}  ->  fix: {iss.array_name}[{fixed}]")
            print()

    if n_launches:
        print(f"Found {n_launches} launch site(s) with wrong grid/block config:\n")
        for li in launch_results:
            grid_changed  = li.original_grid  != li.fixed_grid
            block_changed = li.block_was_literal and li.original_block != li.fixed_block
            print(f"  line {li.lineno:4d}  {li.kernel_name}")
            if grid_changed:
                print(f"    grid:   {li.original_grid}")
                print(f"    fix:    {li.fixed_grid}")
            if block_changed:
                print(f"    block:  {li.original_block}")
                print(f"    fix:    {li.fixed_block}"
                      f"  ← move largest dim to x so blockDim.x fills a warp")
            elif not li.block_was_literal:
                print(f"    block:  {li.original_block}"
                      f"  ← ensure blockDim.x ≥ 32 (largest dim) manually")
        print()

    print("NOTE: after applying --fix, also transpose/reshape host arrays so that")
    print("      the formerly-first dimension becomes the last dimension.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Python source file to analyse")
    ap.add_argument("output", nargs="?", help="Output file for --fix (default: stdout)")
    ap.add_argument("--fix", action="store_true",
                    help="Rewrite kernel subscripts and launch grid tuples")
    args = ap.parse_args()

    src = Path(args.input).read_text(encoding="utf-8")

    if args.fix:
        result = transform_coalesce(src)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            print(f"Written to {args.output}", file=sys.stderr)
        else:
            print(result)
    else:
        access_issues  = analyse_kernel(src)
        coalescing_fns = set(access_issues.keys())
        launch_issues  = analyse_launches(src, coalescing_fns)
        _print_report(access_issues, launch_issues, src)


if __name__ == "__main__":
    main()

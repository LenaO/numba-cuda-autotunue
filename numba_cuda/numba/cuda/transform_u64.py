#!/usr/bin/env python3
"""Transform Numba CUDA kernels to use uint64 indices for non-negative values.

Analyses function bodies with a forward data-flow pass to detect provably
non-negative values:
  - CUDA intrinsics: cuda.threadIdx.{x,y,z}, cuda.blockIdx, cuda.blockDim, cuda.gridDim
  - Array shape accesses: arr.shape[i]
  - cuda.grid() return values
  - Non-negative integer literals (>= 0)
  - Derived values: non-neg + non-neg, non-neg * non-neg, non-neg // X, non-neg % X
  - Loop counters in range(STOP) where STOP is non-negative

Two transformation modes:
  'range' : For loops keep range(); nb.uint64(VAR) is inserted at loop body top.
  'while' : For loops are replaced with while loops using uint64 counters (faster PTX).

In both modes, BinOp expressions used directly as subscript indices (e.g. ``tid + i``)
are hoisted to named temporaries before the subscript statement.  This prevents Numba
from incorrectly inferring float64 for uint64 arithmetic inside ``[]``.

Usage:
    python transform_u64.py input.py                    # prints to stdout (while mode)
    python transform_u64.py input.py --mode range       # range mode
    python transform_u64.py input.py --no-hoist         # skip index hoisting
    python transform_u64.py input.py output.py          # writes to file
    python transform_u64.py input.py output.py --mode range
"""

import ast
import copy
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Pass 1: forward data-flow analysis — collect non-negative variable names
# ---------------------------------------------------------------------------

class _NonNegCollector(ast.NodeVisitor):
    """Walk a function and record every variable that is provably non-negative."""

    _CUDA_STRUCTS = frozenset({"threadIdx", "blockIdx", "blockDim", "gridDim"})
    _CUDA_DIMS = frozenset({"x", "y", "z"})
    _UNSIGNED_TYPES = frozenset({"uint8", "uint16", "uint32", "uint64"})

    def __init__(self, seeds: "set[str] | None" = None) -> None:
        self.non_neg: set[str] = set(seeds) if seeds else set()

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def _is_non_neg_expr(self, node: ast.expr) -> bool:
        """Return True if *node* is statically provably >= 0."""
        # Non-negative integer constant (exclude bool to avoid False/True noise)
        if isinstance(node, ast.Constant):
            return (isinstance(node.value, int)
                    and not isinstance(node.value, bool)
                    and node.value >= 0)

        # Variable already known to be non-negative
        if isinstance(node, ast.Name):
            return node.id in self.non_neg

        # cuda.threadIdx.x / cuda.blockIdx.y / cuda.blockDim.z etc.
        if isinstance(node, ast.Attribute):
            if (node.attr in self._CUDA_DIMS
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr in self._CUDA_STRUCTS
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "cuda"):
                return True

        # arr.shape[i]  — always non-negative
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"
                    and isinstance(node.value.value, ast.Name)):
                return True

        # nb.uint64(...) / uint64(...) / nb.uint32(...) etc. — unsigned, always non-negative
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in self._UNSIGNED_TYPES:
                return True
            if isinstance(func, ast.Name) and func.id in self._UNSIGNED_TYPES:
                return True
            # cuda.grid() returns are handled via the assignment visitor,
            # but recognise the call itself as non-neg for completeness.
            if (isinstance(func, ast.Attribute)
                    and func.attr == "grid"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "cuda"):
                return True

        # Binary arithmetic: propagate non-negativity
        if isinstance(node, ast.BinOp):
            l_nn = self._is_non_neg_expr(node.left)
            r_nn = self._is_non_neg_expr(node.right)
            if isinstance(node.op, (ast.Add, ast.Mult)):
                return l_nn and r_nn
            # non-neg // anything  or  non-neg % anything  stays non-neg
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                return l_nn

        return False

    @staticmethod
    def _is_cuda_grid_call(node: ast.expr) -> bool:
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "grid"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cuda")

    # ------------------------------------------------------------------
    # Visitors  (forward order: assignments processed top-to-bottom)
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if not node.targets:
            return
        target = node.targets[0]

        # x, y = cuda.grid(2)  — all tuple elements are non-negative
        if isinstance(target, ast.Tuple) and self._is_cuda_grid_call(node.value):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.non_neg.add(elt.id)
            return

        # VAR = <non-neg-expr>
        if isinstance(target, ast.Name) and self._is_non_neg_expr(node.value):
            self.non_neg.add(target.id)

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # Infer non-negativity of the loop counter from range() arguments.
        if (isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"):
            args = node.iter.args
            if len(args) == 1:
                # range(stop): counter in [0, stop)
                if self._is_non_neg_expr(args[0]):
                    self.non_neg.add(node.target.id)
            elif len(args) >= 2:
                # range(start, stop[, step]): non-neg if start >= 0, step >= 0
                start = args[0]
                step = args[2] if len(args) == 3 else None
                if self._is_non_neg_expr(start) and (
                        step is None or self._is_non_neg_expr(step)):
                    self.non_neg.add(node.target.id)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Helper: detect break/continue in a loop body (not inside nested loops)
# ---------------------------------------------------------------------------

class _BreakContinueFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_Break(self, node: ast.Break) -> None:
        self.found = True

    def visit_Continue(self, node: ast.Continue) -> None:
        self.found = True

    # Don't descend into nested loops — their break/continue is independent.
    def visit_For(self, node: ast.For) -> None:
        pass

    def visit_While(self, node: ast.While) -> None:
        pass


# ---------------------------------------------------------------------------
# Pass 2: AST transformer — insert nb.uint64() casts
# ---------------------------------------------------------------------------

class _U64Transformer(ast.NodeTransformer):
    """Rewrite a function body inserting nb.uint64() casts for non-neg variables."""

    def __init__(
        self,
        non_neg: set[str],
        mode: str,
        multistep: bool = True,
        dtype: str = "uint64",
        unroll: int = 1,
    ) -> None:
        self.non_neg = non_neg
        self.mode = mode        # 'range' or 'while'
        self.multistep = multistep  # False → skip while-conversion for step != 1
        self.dtype = dtype          # 'uint64' or 'uint32'
        self.unroll = unroll        # manual unroll factor for step-1 while loops

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_u64(self, arg: ast.expr) -> ast.Call:
        """Build  nb.<dtype>(<arg>)."""
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="nb", ctx=ast.Load()),
                attr=self.dtype,
                ctx=ast.Load(),
            ),
            args=[arg],
            keywords=[],
        )

    @staticmethod
    def _is_already_u64(node: ast.expr) -> bool:
        """True if *node* is already any unsigned-integer cast call."""
        _UNSIGNED = frozenset({"uint8", "uint16", "uint32", "uint64"})
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # nb.uint64(...) / numba.uint32(...) etc.
        if isinstance(func, ast.Attribute) and func.attr in _UNSIGNED:
            return True
        # bare uint64(...) / uint32(...) etc.
        if isinstance(func, ast.Name) and func.id in _UNSIGNED:
            return True
        return False

    _CUDA_STRUCTS = frozenset({"threadIdx", "blockIdx", "blockDim", "gridDim"})
    _UNSIGNED_TYPES = frozenset({"uint8", "uint16", "uint32", "uint64"})

    def _is_non_neg_expr(self, node: ast.expr) -> bool:
        """True if *node* is statically provably >= 0 (mirrors _NonNegCollector)."""
        if isinstance(node, ast.Constant):
            return (isinstance(node.value, int)
                    and not isinstance(node.value, bool)
                    and node.value >= 0)
        if isinstance(node, ast.Name):
            return node.id in self.non_neg
        if isinstance(node, ast.Attribute):
            if (node.attr in self._CUDA_DIMS
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr in self._CUDA_STRUCTS
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "cuda"):
                return True
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"
                    and isinstance(node.value.value, ast.Name)):
                return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in self._UNSIGNED_TYPES:
                return True
            if isinstance(func, ast.Name) and func.id in self._UNSIGNED_TYPES:
                return True
            if (isinstance(func, ast.Attribute)
                    and func.attr == "grid"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "cuda"):
                return True
        if isinstance(node, ast.BinOp):
            l_nn = self._is_non_neg_expr(node.left)
            r_nn = self._is_non_neg_expr(node.right)
            if isinstance(node.op, (ast.Add, ast.Mult)):
                return l_nn and r_nn
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                return l_nn
        return False

    @staticmethod
    def _is_cuda_grid_call(node: ast.expr) -> bool:
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "grid"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cuda")

    @staticmethod
    def _has_break_continue(stmts: list) -> bool:
        finder = _BreakContinueFinder()
        for stmt in stmts:
            finder.visit(stmt)
            if finder.found:
                return True
        return False

    # blockDim and gridDim are always >= 1 (can't launch with 0 threads/blocks).
    _CUDA_POSITIVE_STRUCTS = frozenset({"blockDim", "gridDim"})
    _CUDA_DIMS = frozenset({"x", "y", "z"})

    def _is_positive(self, node: ast.expr) -> bool:
        """True if *node* is provably > 0 (needed to validate a range() step).

        Positive integer literals are accepted directly.  Non-constant
        non-negative expressions (variables, unsigned casts) are trusted to be
        non-zero — a step of 0 is a Python ValueError anyway.
        cuda.blockDim.{x,y,z} and cuda.gridDim.{x,y,z} are always >= 1.
        """
        if isinstance(node, ast.Constant):
            return (isinstance(node.value, int)
                    and not isinstance(node.value, bool)
                    and node.value > 0)
        if isinstance(node, ast.Name):
            return node.id in self.non_neg
        # cuda.blockDim.x/y/z  and  cuda.gridDim.x/y/z  — always >= 1
        if (isinstance(node, ast.Attribute)
                and node.attr in self._CUDA_DIMS
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in self._CUDA_POSITIVE_STRUCTS
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "cuda"):
            return True
        _UNSIGNED = frozenset({"uint8", "uint16", "uint32", "uint64"})
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _UNSIGNED:
                return True
            if isinstance(func, ast.Name) and func.id in _UNSIGNED:
                return True
        return False

    # ------------------------------------------------------------------
    # Visitors
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Do not recurse into nested function definitions; they are handled
        # independently by the outer U64FunctionTransformer if needed.
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> "ast.stmt | list[ast.stmt]":
        if not node.targets:
            return node
        target = node.targets[0]

        # ── x, y = cuda.grid(2) ──────────────────────────────────────
        # Keep the original assignment and append nb.uint64 casts for each name.
        if isinstance(target, ast.Tuple) and self._is_cuda_grid_call(node.value):
            casts = []
            for elt in target.elts:
                if isinstance(elt, ast.Name) and elt.id in self.non_neg:
                    cast = ast.Assign(
                        targets=[ast.Name(id=elt.id, ctx=ast.Store())],
                        value=self._make_u64(ast.Name(id=elt.id, ctx=ast.Load())),
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    )
                    casts.append(cast)
            return [node] + casts

        # ── VAR = <expr>  →  VAR = nb.uint64(<expr>) ─────────────────
        # Only wrap when the RHS is also provably non-negative; this prevents
        # nonsense like nb.uint32(-1) when a variable is non-neg in one branch
        # but assigned a negative value in another.
        if (isinstance(target, ast.Name)
                and target.id in self.non_neg
                and not self._is_already_u64(node.value)
                and self._is_non_neg_expr(node.value)):
            node.value = self._make_u64(node.value)

        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        """Rewrite  VAR += <non-neg-expr>  →  VAR += nb.uint32(<expr>).

        Only the Add operator is handled: subtraction could underflow and is
        already excluded from non_neg via _SignSensitiveFinder.  The target
        must be a plain Name that is in the non-negative set, and the RHS must
        itself be provably non-negative.
        """
        if (isinstance(node.target, ast.Name)
                and node.target.id in self.non_neg
                and isinstance(node.op, ast.Add)
                and not self._is_already_u64(node.value)
                and self._is_non_neg_expr(node.value)):
            node.value = self._make_u64(node.value)
        return node

    def visit_For(self, node: ast.For) -> "ast.stmt | list[ast.stmt]":
        # Recurse into the body first so nested loops are transformed.
        self.generic_visit(node)

        # Basic structural eligibility: range() call, single variable,
        # no else clause, no break/continue, counter known non-negative.
        if not (isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and node.target.id in self.non_neg
                and not node.orelse
                and not self._has_break_continue(node.body)):
            return node

        args = node.iter.args
        nargs = len(args)
        step_is_one = True  # updated in the nargs==3 branch

        if nargs == 1:
            # range(stop): start = 0, step = 1
            start_expr = ast.Constant(value=0)
            stop_expr  = args[0]
            step_expr  = ast.Constant(value=1)
        elif nargs == 2:
            # range(start, stop): step = 1; start non-neg guaranteed by collector
            start_expr = args[0]
            stop_expr  = args[1]
            step_expr  = ast.Constant(value=1)
        elif nargs == 3:
            # range(start, stop, step): step must be provably positive.
            # If multistep is disabled, only convert when step is literally 1.
            start_expr = args[0]
            stop_expr  = args[1]
            step_expr  = args[2]
            step_is_one = (isinstance(step_expr, ast.Constant)
                           and step_expr.value == 1)
            if not self._is_positive(step_expr):
                return node
            if not self.multistep and not step_is_one:
                return node
        else:
            return node

        var = node.target.id
        ln, col = node.lineno, node.col_offset

        if self.mode == "range":
            # Prepend  VAR = nb.uint64(VAR)  to the loop body.
            cast_stmt = ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=self._make_u64(ast.Name(id=var, ctx=ast.Load())),
                lineno=ln,
                col_offset=col,
            )
            node.body.insert(0, cast_stmt)
            return node

        # mode == 'while'
        # ── VAR = nb.uint64(START) ───────────────────────────────────
        init = ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=self._make_u64(start_expr),
            lineno=ln,
            col_offset=col,
        )
        # ── VAR += nb.uint64(STEP) ──────────────────────────────────
        increment = ast.AugAssign(
            target=ast.Name(id=var, ctx=ast.Store()),
            op=ast.Add(),
            value=self._make_u64(step_expr),
            lineno=ln,
            col_offset=col,
        )

        # ── Manual unrolling for step-1 loops ───────────────────────
        if self.unroll >= 2 and step_is_one:
            stop_var   = f"_u4stop_{var}"
            unroll_var = f"_u4lim_{var}"

            # _u4stop_VAR = nb.uint64(STOP)
            stop_assign = ast.Assign(
                targets=[ast.Name(id=stop_var, ctx=ast.Store())],
                value=self._make_u64(stop_expr),
                lineno=ln, col_offset=col,
            )
            # _u4lim_VAR = _u4stop_VAR - (_u4stop_VAR - VAR) % nb.uint64(UNROLL)
            # This is the largest multiple of UNROLL reachable from START,
            # i.e. START + floor((STOP - START) / UNROLL) * UNROLL.
            unroll_assign = ast.Assign(
                targets=[ast.Name(id=unroll_var, ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Name(id=stop_var, ctx=ast.Load()),
                    op=ast.Sub(),
                    right=ast.BinOp(
                        left=ast.BinOp(
                            left=ast.Name(id=stop_var, ctx=ast.Load()),
                            op=ast.Sub(),
                            right=ast.Name(id=var, ctx=ast.Load()),
                        ),
                        op=ast.Mod(),
                        right=self._make_u64(ast.Constant(value=self.unroll)),
                    ),
                ),
                lineno=ln, col_offset=col,
            )

            # Main loop: UNROLL copies of (body + increment) per iteration.
            unrolled_body: list[ast.stmt] = []
            for _ in range(self.unroll):
                for stmt in node.body:
                    unrolled_body.append(copy.deepcopy(stmt))
                unrolled_body.append(ast.AugAssign(
                    target=ast.Name(id=var, ctx=ast.Store()),
                    op=ast.Add(),
                    value=self._make_u64(ast.Constant(value=1)),
                    lineno=ln, col_offset=col,
                ))
            main_while = ast.While(
                test=ast.Compare(
                    left=ast.Name(id=var, ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Name(id=unroll_var, ctx=ast.Load())],
                ),
                body=unrolled_body,
                orelse=[],
                lineno=ln, col_offset=col,
            )

            # Tail loop: handles the remaining < UNROLL iterations.
            tail_while = ast.While(
                test=ast.Compare(
                    left=ast.Name(id=var, ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Name(id=stop_var, ctx=ast.Load())],
                ),
                body=node.body + [increment],
                orelse=[],
                lineno=ln, col_offset=col,
            )

            return [init, stop_assign, unroll_assign, main_while, tail_while]

        # ── Standard (non-unrolled) while loop ──────────────────────
        cond = ast.Compare(
            left=ast.Name(id=var, ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[self._make_u64(stop_expr)],
        )
        while_node = ast.While(
            test=cond,
            body=node.body + [increment],
            orelse=[],
            lineno=ln,
            col_offset=col,
        )
        return [init, while_node]


# ---------------------------------------------------------------------------
# Pass 3: hoist BinOp subscript-index expressions to named temporaries
# ---------------------------------------------------------------------------

class _IndexHoister:
    """Hoist BinOp index expressions out of subscript brackets.

    Transforms::

        arr[tid + i]  →  _idx_0 = tid + i; arr[_idx_0]

    Prevents Numba from incorrectly promoting uint64 arithmetic to float64
    when the expression appears directly inside ``[]``.  Only BinOp slices
    (or BinOp elements within a Tuple slice) that involve at least one
    provably non-negative variable are hoisted; everything else is left alone.
    """

    def __init__(self, non_neg: "set[str]") -> None:
        self.non_neg = non_neg
        self._counter = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fresh(self) -> str:
        name = f"_idx_{self._counter}"
        self._counter += 1
        return name

    def _involves_non_neg(self, node: ast.expr) -> bool:
        return any(
            isinstance(n, ast.Name) and n.id in self.non_neg
            for n in ast.walk(node)
        )

    def _maybe_hoist(
        self, sl: ast.expr, ref_node: ast.AST, pending: "list[ast.stmt]"
    ) -> ast.expr:
        """Return a Name node for *sl* if it is a hoistable BinOp, else *sl*."""
        if isinstance(sl, ast.BinOp) and self._involves_non_neg(sl):
            name = self._fresh()
            assign = ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=sl,
                lineno=getattr(ref_node, "lineno", 0),
                col_offset=getattr(ref_node, "col_offset", 0),
            )
            ast.fix_missing_locations(assign)
            pending.append(assign)
            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), ref_node)
        return sl

    # ------------------------------------------------------------------
    # Expression visitor
    # ------------------------------------------------------------------

    def _visit_expr(self, node: ast.expr, pending: "list[ast.stmt]") -> ast.expr:
        """Walk *node*, hoisting qualifying BinOp subscript slices into *pending*."""
        if isinstance(node, ast.Subscript):
            node.value = self._visit_expr(node.value, pending)
            sl = self._visit_expr(node.slice, pending)   # recurse into slice first
            if isinstance(sl, ast.Tuple):
                sl.elts = [self._maybe_hoist(e, node, pending) for e in sl.elts]
                node.slice = sl
            else:
                node.slice = self._maybe_hoist(sl, node, pending)
            return node
        if isinstance(node, ast.BinOp):
            node.left = self._visit_expr(node.left, pending)
            node.right = self._visit_expr(node.right, pending)
            return node
        if isinstance(node, ast.UnaryOp):
            node.operand = self._visit_expr(node.operand, pending)
            return node
        if isinstance(node, ast.BoolOp):
            node.values = [self._visit_expr(v, pending) for v in node.values]
            return node
        if isinstance(node, ast.Compare):
            node.left = self._visit_expr(node.left, pending)
            node.comparators = [self._visit_expr(c, pending) for c in node.comparators]
            return node
        if isinstance(node, ast.Call):
            node.args = [self._visit_expr(a, pending) for a in node.args]
            for kw in node.keywords:
                kw.value = self._visit_expr(kw.value, pending)
            return node
        if isinstance(node, ast.Attribute):
            node.value = self._visit_expr(node.value, pending)
            return node
        if isinstance(node, (ast.Tuple, ast.List)):
            node.elts = [self._visit_expr(e, pending) for e in node.elts]
            return node
        if isinstance(node, ast.IfExp):
            node.test = self._visit_expr(node.test, pending)
            node.body = self._visit_expr(node.body, pending)
            node.orelse = self._visit_expr(node.orelse, pending)
            return node
        # Constants, Names, and anything else: return unchanged
        return node

    # ------------------------------------------------------------------
    # Statement visitor
    # ------------------------------------------------------------------

    def hoist_stmts(self, stmts: "list[ast.stmt]") -> "list[ast.stmt]":
        """Return a new statement list with qualifying subscript indices hoisted."""
        result: list[ast.stmt] = []
        for stmt in stmts:
            pending: list[ast.stmt] = []
            stmt = self._visit_stmt(stmt, pending)
            result.extend(pending)
            result.append(stmt)
        return result

    def _visit_stmt(
        self, stmt: ast.stmt, pending: "list[ast.stmt]"
    ) -> ast.stmt:
        if isinstance(stmt, ast.Assign):
            stmt.value = self._visit_expr(stmt.value, pending)
            stmt.targets = [self._visit_expr(t, pending) for t in stmt.targets]
        elif isinstance(stmt, ast.AugAssign):
            stmt.target = self._visit_expr(stmt.target, pending)
            stmt.value = self._visit_expr(stmt.value, pending)
        elif isinstance(stmt, ast.Expr):
            stmt.value = self._visit_expr(stmt.value, pending)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                stmt.value = self._visit_expr(stmt.value, pending)
        elif isinstance(stmt, ast.If):
            stmt.test = self._visit_expr(stmt.test, pending)
            stmt.body = self.hoist_stmts(stmt.body)
            stmt.orelse = self.hoist_stmts(stmt.orelse)
        elif isinstance(stmt, ast.While):
            stmt.test = self._visit_expr(stmt.test, pending)
            stmt.body = self.hoist_stmts(stmt.body)
            stmt.orelse = self.hoist_stmts(stmt.orelse)
        elif isinstance(stmt, ast.For):
            stmt.iter = self._visit_expr(stmt.iter, pending)
            stmt.body = self.hoist_stmts(stmt.body)
        # FunctionDef and anything else: leave unchanged
        return stmt


# ---------------------------------------------------------------------------
# Pass 3b: find non-neg variables unsafe to cast (used in signed arithmetic)
# ---------------------------------------------------------------------------

class _SignSensitiveFinder(ast.NodeVisitor):
    """Collect non-neg variables that must NOT be cast to unsigned.

    A variable is *sign-sensitive* if it appears:

    * as the right-hand operand (or anywhere in the right subtree) of a
      subtraction: ``expr - var``  →  casting *var* to uint would wrap/underflow.
    * inside a unary negation: ``-var``  →  ``-uint32(var)`` wraps to a large
      unsigned value instead of a negative signed one.

    These variables are excluded from the cast set so signed arithmetic is
    preserved.  Non-negativity information (e.g. for range-loop inference) is
    unaffected — only the actual uint cast is skipped.
    """

    def __init__(self, non_neg: "set[str]") -> None:
        self.non_neg = non_neg
        self.sign_sensitive: set[str] = set()

    def _collect_names(self, node: ast.expr) -> None:
        """Add every non-neg Name in *node*'s subtree to sign_sensitive."""
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in self.non_neg:
                self.sign_sensitive.add(n.id)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Sub):
            self._collect_names(node.right)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.USub):
            self._collect_names(node.operand)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Top-level transformer: processes each function independently
# ---------------------------------------------------------------------------

class U64FunctionTransformer(ast.NodeTransformer):
    """Entry point: transforms every FunctionDef in a module."""

    def __init__(
        self,
        mode: str = "while",
        multistep: bool = True,
        dtype: str = "uint64",
        unroll: int = 1,
        hoist: bool = True,
    ) -> None:
        if dtype not in ("uint32", "uint64"):
            raise ValueError(f"dtype must be 'uint32' or 'uint64', got {dtype!r}")
        self.mode = mode
        self.multistep = multistep
        self.dtype = dtype
        self.unroll = unroll
        self.hoist = hoist

    @staticmethod
    def _parse_sig_string(sig_str: str) -> list[str]:
        """Split a Numba signature string into individual type strings.

        Handles ``"void(uint8[:,:], uint8[:,:], uint32)"`` as well as bare
        ``"uint8[:,:], uint32"`` forms.  Returns one stripped string per
        positional argument, e.g. ``["uint8[:,:]", "uint8[:,:]", "uint32"]``.
        """
        # Strip optional leading return type: "void(...)" -> "..."
        paren = sig_str.find("(")
        if paren >= 0:
            close = sig_str.rfind(")")
            inner = sig_str[paren + 1 : close]
        else:
            inner = sig_str
        # Split by commas while respecting brackets so "uint8[:,:]" stays whole.
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in inner:
            if ch in "([":
                depth += 1
                current.append(ch)
            elif ch in ")]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return parts

    @staticmethod
    def _unsigned_params_from_decorator(
        node: "ast.FunctionDef | ast.AsyncFunctionDef",
    ) -> list[str]:
        """Return param names typed as unsigned in a @cuda.jit() / @jit() decorator.

        Handles both the string form:
            @cuda.jit("void(uint8[:,:], uint8[:,:], uint32)")
        and the AST-node form:
            @cuda.jit(uint32, float32[:, :])
        """
        _UNSIGNED = frozenset({"uint8", "uint16", "uint32", "uint64"})
        param_names = [arg.arg for arg in node.args.args]
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            func = deco.func
            is_jit = (
                (isinstance(func, ast.Name) and func.id == "jit")
                or (isinstance(func, ast.Attribute) and func.attr == "jit")
            )
            if not is_jit:
                continue

            # ── String signature: @cuda.jit("void(uint8[:,:], uint32)") ──────
            if (len(deco.args) == 1
                    and isinstance(deco.args[0], ast.Constant)
                    and isinstance(deco.args[0].value, str)):
                type_strs = U64FunctionTransformer._parse_sig_string(
                    deco.args[0].value
                )
                return [
                    param_names[i]
                    for i, ts in enumerate(type_strs)
                    if i < len(param_names) and ts in _UNSIGNED
                ]

            # ── AST-node signature: @cuda.jit(uint32, float32[:, :]) ─────────
            result = []
            for i, type_arg in enumerate(deco.args):
                if i >= len(param_names):
                    break
                if isinstance(type_arg, ast.Name) and type_arg.id in _UNSIGNED:
                    result.append(param_names[i])
                elif isinstance(type_arg, ast.Attribute) and type_arg.attr in _UNSIGNED:
                    result.append(param_names[i])
            return result
        return []

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Collect module-level positive integer constants before visiting functions.

        Simple assignments of the form  NAME = <positive-int>  or
        NAME: <type> = <positive-int>  at module scope are recorded so they
        can be used as seeds for the non-negativity analysis inside each
        function (e.g. ``TPB = 16``).
        """
        pos: set[str] = set()
        for stmt in node.body:
            # Plain assignment:  TPB = 16
            if (isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, int)
                    and not isinstance(stmt.value.value, bool)
                    and stmt.value.value > 0):
                pos.add(stmt.targets[0].id)
            # Annotated assignment:  TPB: int = 16
            elif (isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.value is not None
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, int)
                    and not isinstance(stmt.value.value, bool)
                    and stmt.value.value > 0):
                pos.add(stmt.target.id)
        self._module_pos_consts = pos
        return self.generic_visit(node)

    def _transform_function(
        self, node: "ast.FunctionDef | ast.AsyncFunctionDef"
    ) -> "ast.FunctionDef | ast.AsyncFunctionDef":
        # Pre-pass: for parameters declared as unsigned in the @cuda.jit / @jit
        # signature, insert  VAR = nb.uint64(VAR)  at the top of the body.
        # This runs *before* the collector so those names are picked up naturally
        # via the existing visit_Assign path.
        unsigned_params = self._unsigned_params_from_decorator(node)
        if unsigned_params:
            ln, col = node.lineno, node.col_offset
            casts = [
                ast.Assign(
                    targets=[ast.Name(id=p, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="nb", ctx=ast.Load()),
                            attr=self.dtype,
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=p, ctx=ast.Load())],
                        keywords=[],
                    ),
                    lineno=ln,
                    col_offset=col,
                )
                for p in unsigned_params
            ]
            ast.fix_missing_locations(node)
            node.body = casts + node.body

        # Pass 1: collect non-negative variables via forward data-flow analysis.
        # Seed with module-level positive constants (e.g. TPB = 16) if available.
        seeds = getattr(self, "_module_pos_consts", set())
        collector = _NonNegCollector(seeds=seeds)
        collector.visit(node)

        # Pass 1b: remove variables that appear on the RHS of subtraction or
        # inside unary negation — casting them to unsigned would wrap/underflow
        # (e.g. ``1 - row`` or ``-row * 2`` with row typed as uint32).
        sign_finder = _SignSensitiveFinder(collector.non_neg)
        sign_finder.visit(node)
        safe_non_neg = collector.non_neg - sign_finder.sign_sensitive

        # Pass 2 (optional): hoist BinOp index expressions out of subscript
        # brackets before the uint transformation so that introduced temporaries
        # are visible to the subsequent collector pass and can receive casts.
        # Only hoist expressions that involve variables that will actually be
        # cast (safe_non_neg), so mixed int/uint temporaries are not created.
        if self.hoist:
            node.body = _IndexHoister(safe_non_neg).hoist_stmts(node.body)

            # Pass 2b: re-run collector + sign finder on the hoisted body so
            # that newly introduced _idx_N temporaries are included (or
            # excluded) correctly.
            collector = _NonNegCollector(seeds=seeds)
            collector.visit(node)
            sign_finder = _SignSensitiveFinder(collector.non_neg)
            sign_finder.visit(node)
            safe_non_neg = collector.non_neg - sign_finder.sign_sensitive

        # Pass 3: rewrite the function body with uint casts.
        transformer = _U64Transformer(
            safe_non_neg, self.mode, self.multistep, self.dtype, self.unroll
        )
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            result = transformer.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            elif result is not None:
                new_body.append(result)
        node.body = new_body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._transform_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._transform_function(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_source_u64(
    source: str,
    mode: str = "while",
    multistep: bool = True,
    dtype: str = "uint64",
    unroll: int = 1,
    hoist: bool = True,
) -> str:
    """Parse *source*, apply unsigned-integer promotion, and return unparsed Python.

    :param source:     Source code string containing one or more function definitions.
    :param mode:       ``'while'`` (default) converts eligible ``for`` loops to
                       ``while`` loops with unsigned counters.  ``'range'`` keeps the
                       ``for`` loop and inserts a cast at the top of the body.
    :param multistep:  When ``True`` (default), ``range(a, b, step)`` loops are
                       also converted as long as *step* is provably positive.
                       Set to ``False`` to leave any loop with a step other than
                       the literal ``1`` as a ``for`` loop (useful for benchmarking
                       strided loops such as tiled shared-memory loads).
    :param dtype:      Unsigned integer type to use for casts: ``'uint64'`` (default)
                       or ``'uint32'``.
    :param unroll:     Manual unroll factor for step-1 while loops (default ``1``,
                       i.e. no unrolling).  A value of ``4`` emits four copies of
                       the loop body per while-iteration plus a tail loop for the
                       remaining ``< unroll`` iterations.  Only applied when
                       ``mode='while'`` and the step is the literal ``1``.
    :param hoist:      When ``True`` (default), BinOp subscript-index expressions
                       involving non-negative variables are hoisted to named
                       temporaries before the uint cast pass.  Set to ``False`` to
                       skip hoisting (e.g. when the downstream compiler handles
                       mixed-type subscript arithmetic correctly).
    """
    if mode not in ("range", "while"):
        raise ValueError(f"mode must be 'range' or 'while', got {mode!r}")
    tree = ast.parse(source)
    new_tree = U64FunctionTransformer(
        mode, multistep=multistep, dtype=dtype, unroll=unroll, hoist=hoist
    ).visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = "while"
    multistep = True
    dtype = "uint64"
    unroll = 1
    hoist = True
    positional: list[str] = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--dtype" and i + 1 < len(sys.argv):
            dtype = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--unroll" and i + 1 < len(sys.argv):
            unroll = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no-multistep":
            multistep = False
            i += 1
        elif sys.argv[i] == "--no-hoist":
            hoist = False
            i += 1
        else:
            positional.append(sys.argv[i])
            i += 1

    if not positional:
        print(__doc__)
        sys.exit(1)

    input_path = Path(positional[0])
    source = input_path.read_text(encoding="utf-8")
    result = transform_source_u64(source, mode=mode, multistep=multistep, dtype=dtype, unroll=unroll, hoist=hoist)

    if len(positional) >= 2:
        output_path = Path(positional[1])
        output_path.write_text(result, encoding="utf-8")
        print(f"Transformed output written to: {output_path}")
    else:
        print(result)


if __name__ == "__main__":
    main()

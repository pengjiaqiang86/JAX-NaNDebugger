"""Locate the first operation that produces a NaN in a JAX forward run.

Strategy: trace the function to a jaxpr, then re-evaluate it one primitive at
a time in Python (un-jitted), checking every intermediate value for NaNs.
The first equation whose *inputs are NaN-free* but whose *output contains
NaNs* is the true origin; everything after it is just propagation.

Sub-computations (jit/pjit, custom_jvp/vjp, remat, named scopes) are
recursed into so the report points at the innermost responsible primitive.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import re
import sys
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

try:  # JAX >= 0.6: jaxpr classes live in jax.extend.core
    from jax.extend import core
except ImportError:  # pragma: no cover
    from jax import core

try:  # source_info_util is internal; degrade gracefully if it moves.
    from jax._src import source_info_util

    _OWN_DIR = os.path.dirname(__file__)

    def _summarize_source(eqn: core.JaxprEqn) -> str:
        frame = source_info_util.user_frame(eqn.source_info.traceback)
        if frame is None:
            return "<unknown location>"
        return f"{frame.file_name}:{frame.start_line} in {frame.function_name}"

    def _call_stack(eqn: core.JaxprEqn) -> list[str]:
        """User call chain to the equation, outermost first (like a Python
        traceback). JAX-internal frames and this package's own frames are
        dropped."""
        frames = [
            f
            for f in source_info_util.user_frames(eqn.source_info.traceback)
            if not f.file_name.startswith(_OWN_DIR)
        ]
        return [
            f"{f.file_name}:{f.start_line} in {f.function_name}"
            for f in reversed(frames)
        ]
except ImportError:  # pragma: no cover

    def _summarize_source(eqn: core.JaxprEqn) -> str:
        return "<source info unavailable in this JAX version>"

    def _call_stack(eqn: core.JaxprEqn) -> list[str]:
        return []


def _has_nan(x: Any) -> bool:
    if not isinstance(x, jax.Array) and not hasattr(x, "dtype"):
        return False
    if not jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact):
        return False
    return bool(jnp.any(jnp.isnan(x)))


def _stats(x: Any) -> str:
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        return f"{x.dtype}{list(x.shape)}"
    n_nan = int(jnp.sum(jnp.isnan(x)))
    n_inf = int(jnp.sum(jnp.isinf(x)))
    finite = x[jnp.isfinite(x)] if n_nan or n_inf else x.ravel()
    rng = (
        f"range=[{float(jnp.min(finite)):.4g}, {float(jnp.max(finite)):.4g}]"
        if finite.size
        else "all non-finite"
    )
    return f"{x.dtype}{list(x.shape)} {rng} nans={n_nan} infs={n_inf}"


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _Style:
    """Tiny ANSI styler; a no-op when color is disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.enabled else s

    def title(self, s):
        return self._w("1;31", s)  # bold red

    def prim(self, s):
        return self._w("1;35", s)  # bold magenta

    def path(self, s):
        return self._w("36", s)  # cyan

    def fn(self, s):
        return self._w("1", s)  # bold

    def dim(self, s):
        return self._w("2", s)

    def bad(self, s):
        return self._w("1;31", s)

    def warn(self, s):
        return self._w("33", s)

    def good(self, s):
        return self._w("1;32", s)


def _style_frame(frame: str, st: _Style) -> str:
    # Frames are formatted as "file:line in func" by _call_stack.
    loc, sep, fn = frame.rpartition(" in ")
    if not sep:
        return frame
    return f"{st.path(loc)} in {st.fn(fn)}"


def _style_stats(stats: str, st: _Style) -> str:
    stats = re.sub(r"nans=([1-9]\d*)", lambda m: st.bad(m.group(0)), stats)
    stats = re.sub(r"infs=([1-9]\d*)", lambda m: st.warn(m.group(0)), stats)
    return stats


def _one_line(s: str, limit: int = 100) -> str:
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclasses.dataclass
class NanSite:
    """One equation at which NaNs first appeared."""

    primitive: str
    source: str
    eqn_repr: str
    input_stats: list[str]
    output_stats: list[str]
    inputs_had_nan: bool
    call_stack: list[str] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        return self.render()

    def render(self, color: bool | None = None) -> str:
        st = _Style(_color_enabled() if color is None else color)
        kind = (
            "NaNs propagated from inputs of"
            if self.inputs_had_nan
            else "NaNs first produced by"
        )
        lines = [st.title(f"✘ {kind}: ") + st.prim(self.primitive)]
        if self.call_stack:
            lines.append("│ " + st.dim("call stack (most recent call last):"))
            for i, frame in enumerate(self.call_stack):
                mark = "▶" if i == len(self.call_stack) - 1 else " "
                lines.append(f"│   {st.title(mark) if mark == '▶' else mark} "
                             + _style_frame(frame, st))
        else:
            lines.append(f"│ {st.title('▶')} " + _style_frame(self.source, st))
        lines.append("│ " + st.dim("eqn:") + f" {_one_line(self.eqn_repr)}")
        lines.append("│ " + st.dim("inputs:"))
        for i, s in enumerate(self.input_stats):
            lines.append(f"│   [{i}] " + _style_stats(s, st))
        lines.append("╰ " + st.dim("outputs:"))
        for i, s in enumerate(self.output_stats):
            lines.append(f"    [{i}] " + _style_stats(s, st))
        return "\n".join(lines)


@dataclasses.dataclass
class NanReport:
    """Result of a forward NaN scan."""

    nan_in_inputs: bool
    first_site: NanSite | None
    outputs: Any

    @property
    def found(self) -> bool:
        return self.nan_in_inputs or self.first_site is not None

    def __str__(self) -> str:
        return self.render()

    def render(self, color: bool | None = None) -> str:
        st = _Style(_color_enabled() if color is None else color)
        if self.nan_in_inputs:
            return st.title("✘ NaNs are already present in the function inputs.")
        if self.first_site is None:
            return st.good("✔ No NaNs found in any intermediate or output value.")
        return self.first_site.render(color)


# Params that hold a sub-jaxpr we should recurse into for a precise location.
_SUBJAXPR_PARAMS = ("jaxpr", "call_jaxpr", "fun_jaxpr")


class _NanFound(Exception):
    def __init__(self, site: NanSite):
        self.site = site


@dataclasses.dataclass
class _ScanState:
    """Tracks the first place a NaN was merely *carried*, not produced.

    JAX library code intentionally feeds NaN literals through `where`-style
    guards (e.g. ``jnp.std`` contains ``where(n - ddof > 0, var, nan)``), so
    a NaN flowing through an equation is not evidence of a bug. Only an
    equation with NaN-free inputs and NaN outputs is raised as the origin;
    this state is the fallback when no such origin exists.
    """

    first_propagated: NanSite | None = None


def _eval_eqn(
    eqn: core.JaxprEqn, invals: Sequence[Any], state: _ScanState
) -> Sequence[Any]:
    """Evaluate one equation, recursing into sub-jaxprs when present."""
    for name in _SUBJAXPR_PARAMS:
        sub = eqn.params.get(name)
        if isinstance(sub, core.ClosedJaxpr):
            return _eval_jaxpr_checked(sub.jaxpr, sub.consts, *invals, state=state)
        if isinstance(sub, core.Jaxpr):
            return _eval_jaxpr_checked(sub, (), *invals, state=state)
    out = eqn.primitive.bind(*invals, **eqn.params)
    return out if eqn.primitive.multiple_results else (out,)


def _check_eqn(
    eqn: core.JaxprEqn,
    invals: Sequence[Any],
    outvals: Sequence[Any],
    state: _ScanState,
) -> None:
    if not any(_has_nan(v) for v in outvals):
        return
    site = NanSite(
        primitive=str(eqn.primitive),
        source=_summarize_source(eqn),
        eqn_repr=str(eqn),
        input_stats=[_stats(v) for v in invals],
        output_stats=[_stats(v) for v in outvals],
        inputs_had_nan=any(_has_nan(v) for v in invals),
        call_stack=_call_stack(eqn),
    )
    if not site.inputs_had_nan:
        raise _NanFound(site)
    if state.first_propagated is None:
        state.first_propagated = site


def _eval_jaxpr_checked(
    jaxpr: core.Jaxpr, consts: Sequence[Any], *args: Any, state: _ScanState
) -> list[Any]:
    env: dict[core.Var, Any] = {}

    def read(v):
        return v.val if isinstance(v, core.Literal) else env[v]

    def write(v, val):
        env[v] = val

    for var, c in zip(jaxpr.constvars, consts):
        write(var, c)
    for var, a in zip(jaxpr.invars, args):
        write(var, a)

    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        try:
            outvals = _eval_eqn(eqn, invals, state)
        except _NanFound:
            raise
        except Exception:
            # Primitive not evaluable outside its normal context (e.g. some
            # control-flow internals); fall back to opaque bind so the scan
            # can continue at this granularity.
            out = eqn.primitive.bind(*invals, **eqn.params)
            outvals = out if eqn.primitive.multiple_results else (out,)
        _check_eqn(eqn, invals, outvals, state)
        for var, val in zip(eqn.outvars, outvals):
            if not isinstance(var, core.DropVar):
                write(var, val)

    return [read(v) for v in jaxpr.outvars]


def find_nan_source(fn: Callable, *args: Any, **kwargs: Any) -> NanReport:
    """Run ``fn(*args, **kwargs)`` un-jitted, checking each primitive for NaNs.

    Returns a :class:`NanReport` whose ``first_site`` identifies the first
    equation that produced NaNs (with source location), or ``None`` if the
    run is clean. If the inputs themselves contain NaNs, ``nan_in_inputs``
    is set and no equation is blamed.
    """
    flat_in, _ = jax.tree.flatten((args, kwargs))
    if any(_has_nan(x) for x in flat_in):
        return NanReport(nan_in_inputs=True, first_site=None, outputs=None)

    closed = jax.make_jaxpr(functools.partial(fn, **kwargs))(*args)
    flat_args = jax.tree.leaves(args)

    state = _ScanState()
    try:
        flat_out = _eval_jaxpr_checked(
            closed.jaxpr, closed.consts, *flat_args, state=state
        )
    except _NanFound as e:
        return NanReport(nan_in_inputs=False, first_site=e.site, outputs=None)

    out_tree = jax.tree.structure(jax.eval_shape(functools.partial(fn, **kwargs), *args))
    outputs = jax.tree.unflatten(out_tree, flat_out)
    if any(_has_nan(x) for x in flat_out):
        # No equation produced a NaN from clean inputs, so the NaN entered
        # via a constant/literal; blame the first place it was carried.
        return NanReport(
            nan_in_inputs=False, first_site=state.first_propagated, outputs=outputs
        )
    return NanReport(nan_in_inputs=False, first_site=None, outputs=outputs)


def nan_trace(fn: Callable) -> Callable:
    """Decorator: run ``fn`` normally, but on a NaN in the output re-run it
    through :func:`find_nan_source` and raise with the located origin.

    Adds no overhead to clean runs beyond one output NaN check.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        if any(_has_nan(x) for x in jax.tree.leaves(out)):
            report = find_nan_source(fn, *args, **kwargs)
            raise FloatingPointError(
                f"NaN detected in output of {fn.__name__}:\n{report}"
            )
        return out

    return wrapped

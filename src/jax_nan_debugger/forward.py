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
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

try:  # JAX >= 0.6: jaxpr classes live in jax.extend.core
    from jax.extend import core
except ImportError:  # pragma: no cover
    from jax import core

try:  # source_info_util is internal; degrade gracefully if it moves.
    from jax._src import source_info_util

    def _summarize_source(eqn: core.JaxprEqn) -> str:
        frame = source_info_util.user_frame(eqn.source_info.traceback)
        if frame is None:
            return "<unknown location>"
        return f"{frame.file_name}:{frame.start_line} in {frame.function_name}"
except ImportError:  # pragma: no cover

    def _summarize_source(eqn: core.JaxprEqn) -> str:
        return "<source info unavailable in this JAX version>"


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


@dataclasses.dataclass
class NanSite:
    """One equation at which NaNs first appeared."""

    primitive: str
    source: str
    eqn_repr: str
    input_stats: list[str]
    output_stats: list[str]
    inputs_had_nan: bool

    def __str__(self) -> str:
        kind = (
            "NaNs propagated from inputs of"
            if self.inputs_had_nan
            else "NaNs FIRST PRODUCED by"
        )
        lines = [
            f"{kind}: {self.primitive}",
            f"  at {self.source}",
            f"  eqn: {self.eqn_repr}",
            "  inputs:",
            *(f"    [{i}] {s}" for i, s in enumerate(self.input_stats)),
            "  outputs:",
            *(f"    [{i}] {s}" for i, s in enumerate(self.output_stats)),
        ]
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
        if self.nan_in_inputs:
            return "NaNs are already present in the function inputs."
        if self.first_site is None:
            return "No NaNs found in any intermediate or output value."
        return str(self.first_site)


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

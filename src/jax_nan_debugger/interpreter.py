"""NaN-checked jaxpr evaluation: the engine behind the forward scan.

Evaluates a jaxpr one primitive at a time in Python (un-jitted), checking
every intermediate value. The first equation whose *inputs are NaN-free*
but whose *output contains NaNs* is the true origin; everything after it
is just propagation. Sub-computations (jit/pjit, custom_jvp/vjp, remat)
are recursed into so the blame lands on the innermost primitive.

The inverse debugger can reuse this engine by feeding it a transformed
(e.g. VJP) jaxpr.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from ._jax_compat import call_stack, core, summarize_source
from .report import NanSite, has_nan, value_stats

# Params that hold a sub-jaxpr we should recurse into for a precise location.
_SUBJAXPR_PARAMS = ("jaxpr", "call_jaxpr", "fun_jaxpr")


class NanFound(Exception):
    """Raised by the interpreter at the first NaN-producing equation."""

    def __init__(self, site: NanSite):
        self.site = site


@dataclasses.dataclass
class ScanState:
    """Tracks the first place a NaN was merely *carried*, not produced.

    JAX library code intentionally feeds NaN literals through `where`-style
    guards (e.g. ``jnp.std`` contains ``where(n - ddof > 0, var, nan)``), so
    a NaN flowing through an equation is not evidence of a bug. Only an
    equation with NaN-free inputs and NaN outputs is raised as the origin;
    this state is the fallback when no such origin exists.
    """

    first_propagated: NanSite | None = None


def _eval_eqn(
    eqn: core.JaxprEqn, invals: Sequence[Any], state: ScanState
) -> Sequence[Any]:
    """Evaluate one equation, recursing into sub-jaxprs when present.

    Call-like primitives (``pjit``, ``custom_jvp_call``, ``remat`` ...)
    carry their body as a jaxpr in ``eqn.params``; evaluating that body
    through :func:`eval_jaxpr_checked` instead of binding the primitive
    opaquely is what lets the scan blame the innermost responsible
    operation rather than the whole call.

    Args:
        eqn: The equation to evaluate.
        invals: Concrete input values, in ``eqn.invars`` order.
        state: Propagation bookkeeping threaded through the recursion.

    Returns:
        The equation's outputs as a sequence (single-result primitives
        are wrapped in a 1-tuple, mirroring ``eqn.outvars``).

    Raises:
        NanFound: Propagated from a nested :func:`eval_jaxpr_checked` when
            the origin lies inside a sub-jaxpr.
    """
    for name in _SUBJAXPR_PARAMS:
        sub = eqn.params.get(name)
        if isinstance(sub, core.ClosedJaxpr):
            return eval_jaxpr_checked(sub.jaxpr, sub.consts, *invals, state=state)
        if isinstance(sub, core.Jaxpr):
            return eval_jaxpr_checked(sub, (), *invals, state=state)
    out = eqn.primitive.bind(*invals, **eqn.params)
    return out if eqn.primitive.multiple_results else (out,)


def _check_eqn(
    eqn: core.JaxprEqn,
    invals: Sequence[Any],
    outvals: Sequence[Any],
    state: ScanState,
) -> None:
    """Classify an equation's NaNs as produced or merely carried.

    Args:
        eqn: The equation that was just evaluated.
        invals: Its concrete input values.
        outvals: Its concrete output values.
        state: Receives the first carried-NaN site as a fallback.

    Raises:
        NanFound: When the outputs contain NaNs but the inputs do not —
            this equation is the origin and the scan stops. Equations
            whose inputs already carried NaNs are recorded in ``state``
            instead, because they may be intentional (see
            :class:`ScanState`).
    """
    if not any(has_nan(v) for v in outvals):
        return
    site = NanSite(
        primitive=str(eqn.primitive),
        source=summarize_source(eqn),
        eqn_repr=str(eqn),
        input_stats=[value_stats(v) for v in invals],
        output_stats=[value_stats(v) for v in outvals],
        inputs_had_nan=any(has_nan(v) for v in invals),
        call_stack=call_stack(eqn),
    )
    if not site.inputs_had_nan:
        raise NanFound(site)
    if state.first_propagated is None:
        state.first_propagated = site


def eval_jaxpr_checked(
    jaxpr: core.Jaxpr, consts: Sequence[Any], *args: Any, state: ScanState
) -> list[Any]:
    """Evaluate a jaxpr equation by equation, checking each result for NaNs.

    This is a plain Python interpreter over the jaxpr (the same shape as
    ``jax.core.eval_jaxpr``) with a NaN check after every equation. It is
    orders of magnitude slower than compiled execution and intended only
    for debugging; see :func:`jax_nan_debugger.nan_trace` for a wrapper
    that pays this cost only when a NaN is actually present.

    Equations that cannot be evaluated outside their normal context (some
    control-flow internals) fall back to an opaque ``primitive.bind``, so
    the scan continues at whole-primitive granularity rather than failing.

    Args:
        jaxpr: The (open) jaxpr to evaluate.
        consts: Values for ``jaxpr.constvars``.
        *args: Values for ``jaxpr.invars``, already flattened.
        state: Shared bookkeeping for carried NaNs; create one
            :class:`ScanState` per scan and thread it through.

    Returns:
        The values of ``jaxpr.outvars``, as a list.

    Raises:
        NanFound: At the first equation that produces NaNs from NaN-free
            inputs, carrying the fully populated :class:`NanSite`.
    """
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
        except NanFound:
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

"""Public API for locating NaN origins in a JAX forward run."""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax

from .interpreter import NanFound, ScanState, eval_jaxpr_checked
from .report import NanReport, has_nan


def find_nan_source(fn: Callable, *args: Any, **kwargs: Any) -> NanReport:
    """Locate the operation that first produces a NaN in ``fn(*args, **kwargs)``.

    The function is traced to a jaxpr with ``jax.make_jaxpr`` and then
    re-evaluated one primitive at a time, un-jitted, with a NaN check
    after every step. The first equation whose inputs are NaN-free but
    whose output contains NaNs is reported as the origin, together with
    the user call stack and input/output value statistics. ``jit`` /
    ``pjit`` / ``custom_jvp`` / ``custom_vjp`` / remat sub-computations
    are recursed into, so the blame lands on the innermost primitive.

    Three outcomes are possible — see :class:`NanReport`:

    - the inputs already contain NaNs (``report.nan_in_inputs``);
    - an origin equation was found (``report.first_site``);
    - the run is clean (``report.found`` is False, ``report.outputs``
      holds the result).

    Args:
        fn: A JAX-traceable callable. It will *not* be executed compiled;
            side effects such as ``jax.debug.print`` fire under tracing
            semantics.
        *args: Positional arguments, passed through ``jax.make_jaxpr`` —
            arbitrary pytrees of arrays/scalars are fine.
        **kwargs: Keyword arguments; bound via ``functools.partial`` before
            tracing, so they may be non-array Python values as well.

    Returns:
        A :class:`NanReport`. Print it for a human-readable, colorized
        diagnosis, or inspect ``report.first_site`` programmatically.

    Example:
        >>> def f(x):
        ...     return jnp.sum(jnp.log(x - 2.0))
        >>> report = find_nan_source(f, jnp.array([1.0, 3.0]))
        >>> report.first_site.primitive
        'log'

    Note:
        Evaluation is op-by-op Python interpretation — orders of magnitude
        slower than a compiled run. Use :func:`nan_trace` to pay this cost
        only when a NaN actually occurs. Also note XLA fusion may reorder
        floating-point operations, so in rare cases the compiled run NaNs
        where the un-jitted scan does not (or vice versa).
    """
    flat_in, _ = jax.tree.flatten((args, kwargs))
    if any(has_nan(x) for x in flat_in):
        return NanReport(nan_in_inputs=True, first_site=None, outputs=None)

    closed = jax.make_jaxpr(functools.partial(fn, **kwargs))(*args)
    flat_args = jax.tree.leaves(args)

    state = ScanState()
    try:
        flat_out = eval_jaxpr_checked(
            closed.jaxpr, closed.consts, *flat_args, state=state
        )
    except NanFound as e:
        return NanReport(nan_in_inputs=False, first_site=e.site, outputs=None)

    out_tree = jax.tree.structure(jax.eval_shape(functools.partial(fn, **kwargs), *args))
    outputs = jax.tree.unflatten(out_tree, flat_out)
    if any(has_nan(x) for x in flat_out):
        # No equation produced a NaN from clean inputs, so the NaN entered
        # via a constant/literal; blame the first place it was carried.
        return NanReport(
            nan_in_inputs=False, first_site=state.first_propagated, outputs=outputs
        )
    return NanReport(nan_in_inputs=False, first_site=None, outputs=outputs)


def nan_trace(fn: Callable) -> Callable:
    """Decorator: run ``fn`` at full speed, locating NaN origins on demand.

    The wrapped function executes normally (jitted code stays jitted).
    After each call the outputs are checked for NaNs; on a hit, the call
    is re-run through :func:`find_nan_source` and a ``FloatingPointError``
    is raised whose message contains the full located report. Clean calls
    pay only one NaN reduction over the outputs.

    This is the right tool for loops over many batches/steps: the slow
    op-by-op scan runs only for the one call that actually fails, and the
    raised error tells you both *which call* (from your own loop context)
    and *which operation* (from the report).

    Args:
        fn: A JAX-traceable callable; both its normal execution and the
            diagnostic re-run receive the same arguments.

    Returns:
        A wrapped callable with the same signature, name, and docstring
        (via ``functools.wraps``).

    Raises:
        FloatingPointError: When the output contains a NaN. Raised by the
            *wrapper* at call time, not by this decorator itself.

    Example:
        >>> @nan_trace
        ... def loss(p):
        ...     return -jnp.sum(p * jnp.log(p))
        >>> loss(jnp.array([0.5, 0.5]))     # runs at full speed
        >>> loss(jnp.array([1.0, 0.0]))     # raises with the located `mul`

    Caveat:
        The diagnostic re-run assumes ``fn`` is deterministic for the
        given arguments; a function whose NaN depends on external state
        may not reproduce it during the re-scan.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        if any(has_nan(x) for x in jax.tree.leaves(out)):
            report = find_nan_source(fn, *args, **kwargs)
            raise FloatingPointError(
                f"NaN detected in output of {fn.__name__}:\n{report}"
            )
        return out

    return wrapped

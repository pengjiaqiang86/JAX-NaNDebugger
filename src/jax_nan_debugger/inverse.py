"""Public API for locating NaN origins in a JAX backward (gradient) run.

Backward-pass NaNs are nastier than forward ones because the offending
operation never appears in the user's source: it is synthesized by the
VJP transform. The classic traps share a shape — the *forward* value is
fine, but the local derivative is infinite at the evaluated point and
meets a zero cotangent (``0 * inf = NaN``):

- ``sqrt(x)`` at ``x == 0``: derivative ``0.5/sqrt(x)`` is inf;
- ``jnp.where(x > 0, jnp.log(x), 0.0)``: the masked branch still gets a
  gradient of ``1/x = inf`` at 0, multiplied by a zero cotangent;
- ``jnp.linalg.norm`` / RMSE-style losses at a perfect fit: gradient is
  ``err / norm = 0/0``.

:func:`find_grad_nan_source` scans the forward pass first (a forward NaN
would contaminate every gradient, so it must be ruled out), then scans
the VJP-transformed computation with the same checked interpreter. JAX
attributes backward equations to the forward line that generated them,
so the report's call stack points at your code either way; the headline
says which pass the NaN was born in.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .forward import find_nan_source
from .report import NanReport


def find_grad_nan_source(
    fn: Callable, *args: Any, cotangents: Any = None, **kwargs: Any
) -> NanReport:
    """Locate the operation that first produces a NaN in ``grad(fn)``.

    The scan runs in two stages:

    1. The forward pass is scanned with :func:`find_nan_source`. If it is
       already contaminated, that origin is returned (tagged
       ``phase="forward"``) — a backward scan would only re-discover the
       same NaN after it propagated through the VJP.
    2. Otherwise the full VJP computation ``vjp(fn)(cotangents)`` is
       traced and scanned equation by equation; the origin is tagged
       ``phase="backward"``.

    Args:
        fn: A JAX-differentiable callable. With the default cotangents it
            must return a single scalar (like a loss); for non-scalar
            outputs pass explicit ``cotangents``.
        *args: Positional arguments (arbitrary pytrees). Gradients are
            taken with respect to *all* of them, so NaNs in e.g.
            data-gradients are found too.
        cotangents: Output cotangent(s) to pull back, matching the output
            structure of ``fn``. Defaults to 1.0 for a scalar output —
            i.e. ordinary ``jax.grad`` semantics.
        **kwargs: Keyword arguments; treated as non-differentiated
            constants, mirroring ``jax.grad``.

    Returns:
        A :class:`NanReport`. ``report.first_site.phase`` tells you which
        pass produced the NaN; for clean runs ``report.outputs`` holds the
        gradients (a tuple, one entry per positional argument).

    Raises:
        ValueError: If ``fn`` does not return a single scalar and no
            explicit ``cotangents`` were provided.

    Example:
        >>> def loss(x):
        ...     return jnp.sum(jnp.sqrt(jax.nn.relu(x)))
        >>> report = find_grad_nan_source(loss, jnp.array([0.0, 4.0]))
        >>> report.first_site.phase
        'backward'

    Note:
        Like the forward scan, evaluation is op-by-op and debug-speed
        only. The backward jaxpr includes the forward equations (VJP
        needs the primal intermediates), but stage 1 guarantees they are
        clean, so any site blamed in stage 2 belongs to the gradient
        computation.
    """
    fn_bound = functools.partial(fn, **kwargs)

    fwd = find_nan_source(fn, *args, **kwargs)
    if fwd.found:
        if fwd.first_site is not None:
            fwd.first_site.phase = "forward"
        return fwd

    if cotangents is None:
        out_shape = jax.eval_shape(fn_bound, *args)
        leaves = jax.tree.leaves(out_shape)
        if len(leaves) != 1 or leaves[0].shape != ():
            raise ValueError(
                "fn does not return a single scalar; pass explicit "
                "`cotangents` matching its output structure to "
                "find_grad_nan_source."
            )
        cotangents = jnp.ones((), dtype=leaves[0].dtype)

    def vjp_fn(*inner_args):
        _, pullback = jax.vjp(fn_bound, *inner_args)
        return pullback(cotangents)

    report = find_nan_source(vjp_fn, *args)
    if report.first_site is not None:
        report.first_site.phase = "backward"
    return report

"""find_nan_source composes with any JAX transform.

``jax.make_jaxpr`` traces through arbitrary transform compositions, so
wrapping the *transformed* function is a fully supported pattern:
``find_nan_source(jax.grad(f), x)``, jvp/vjp/linearize/linear_transpose,
and nested transforms all scan fine. ``find_grad_nan_source`` remains the
convenience wrapper that adds forward/backward phase tagging.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_nan_debugger import find_nan_source, nan_trace


def loss(x):
    # Forward-clean, backward-NaN at x < 0 (sqrt'(0)=inf meets max-grad 0).
    return jnp.sum(jnp.sqrt(jnp.maximum(x, 0.0)))


X = jnp.array([-1.0, 0.0, 4.0])


def test_wrapping_jax_grad():
    report = find_nan_source(jax.grad(loss), X)
    assert report.found
    # Backward eqns carry the forward line's source info.
    assert "in loss" in report.first_site.source


def test_wrapping_jax_jvp():
    def jvp_fn(x, t):
        return jax.jvp(loss, (x,), (t,))[1]

    report = find_nan_source(jvp_fn, X, jnp.ones(3))
    assert report.found


def test_wrapping_vjp_pullback():
    _, pullback = jax.vjp(loss, X)
    report = find_nan_source(pullback, jnp.float32(1.0))
    assert report.found


def test_wrapping_linearize():
    _, lin = jax.linearize(loss, X)
    report = find_nan_source(lin, jnp.ones(3))
    assert report.found


def test_wrapping_linear_transpose():
    linear = lambda x: jnp.sum(x * jnp.array([1.0, jnp.inf, 1.0]))
    transposed = jax.linear_transpose(linear, X)
    report = find_nan_source(transposed, jnp.float32(0.0))  # 0 * inf
    assert report.found


def test_wrapping_grad_of_grad():
    def f(y):
        return jnp.sqrt(jnp.maximum(y, 0.0))

    report = find_nan_source(jax.grad(jax.grad(f)), jnp.float32(-1.0))
    assert report.found


def test_nan_trace_composes_with_grad():
    guarded = nan_trace(jax.grad(loss))
    with pytest.raises(FloatingPointError):
        guarded(X)
    # Clean inputs pass through untouched.
    assert not jnp.any(jnp.isnan(guarded(jnp.array([1.0, 4.0]))))

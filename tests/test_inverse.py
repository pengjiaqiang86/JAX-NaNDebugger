import jax
import jax.numpy as jnp
import pytest

from jax_nan_debugger import find_grad_nan_source


def test_sqrt_at_zero_blamed_in_backward():
    # Forward is clean (sqrt(0) = 0); the NaN is inf * 0 in the VJP.
    def loss(x):
        return jnp.sum(jnp.sqrt(jnp.maximum(x, 0.0)))

    x = jnp.array([-1.0, 0.0, 4.0])
    assert jnp.any(jnp.isnan(jax.grad(loss)(x)))  # the symptom
    report = find_grad_nan_source(loss, x)
    assert report.found
    assert report.first_site.phase == "backward"
    assert not report.first_site.inputs_had_nan


def test_where_log_trap():
    # The classic: where() masks log(0) in the forward pass, but the
    # masked branch still gets cotangent * 1/x = 0 * inf in the backward.
    def loss(x):
        return jnp.sum(jnp.where(x > 0, jnp.log(x), 0.0))

    report = find_grad_nan_source(loss, jnp.array([0.0, 1.0]))
    assert report.first_site.phase == "backward"
    assert report.first_site.primitive == "div"


def test_relu_custom_jvp_is_clean():
    # jax.nn.relu defines a custom JVP exactly to avoid the maximum trap.
    def loss(x):
        return jnp.sum(jnp.sqrt(jax.nn.relu(x)))

    report = find_grad_nan_source(loss, jnp.array([-1.0, 0.0, 4.0]))
    assert not report.found


def test_clean_grad_returns_gradients():
    def loss(x):
        return jnp.sum(x ** 2)

    report = find_grad_nan_source(loss, jnp.array([1.0, 2.0]))
    assert not report.found
    (grad_x,) = report.outputs
    assert jnp.allclose(grad_x, jnp.array([2.0, 4.0]))


def test_forward_nan_reported_with_forward_phase():
    def loss(x):
        return jnp.sum(jnp.log(x))

    report = find_grad_nan_source(loss, jnp.array([-1.0]))
    assert report.first_site.phase == "forward"
    assert report.first_site.primitive == "log"


def test_phase_shown_in_rendered_report():
    def loss(x):
        return jnp.sum(jnp.sqrt(jnp.maximum(x, 0.0)))

    # Needs a negative element: there sqrt'(0)=inf meets maximum-grad 0.
    # (At exactly x=0 the tie-gradient is 0.5, giving inf, not NaN.)
    report = find_grad_nan_source(loss, jnp.array([-1.0]))
    assert "(backward pass)" in report.render(color=False)


def test_nonscalar_output_requires_cotangents():
    with pytest.raises(ValueError, match="cotangents"):
        find_grad_nan_source(lambda x: x * 2.0, jnp.ones(3))


def test_explicit_cotangents_for_nonscalar_output():
    def f(x):
        return jnp.sqrt(jnp.maximum(x, 0.0))  # element-wise, non-scalar

    report = find_grad_nan_source(
        f, jnp.array([-1.0, 4.0]), cotangents=jnp.ones(2)
    )
    assert report.first_site.phase == "backward"


def test_pytree_params_supported():
    params = {"w": jnp.array([[1.0, 0.0], [0.0, 1.0]]), "b": jnp.zeros(2)}

    def loss(params, x):
        return jnp.sqrt(jnp.sum((x @ params["w"] + params["b"]) ** 2))

    # Norm of an exactly-zero vector: backward 0/0.
    report = find_grad_nan_source(loss, params, jnp.zeros((1, 2)))
    assert report.first_site.phase == "backward"

    # Non-zero input: clean, gradients returned with pytree structure.
    report = find_grad_nan_source(loss, params, jnp.ones((1, 2)))
    assert not report.found
    grad_params, grad_x = report.outputs
    assert set(grad_params) == {"w", "b"}

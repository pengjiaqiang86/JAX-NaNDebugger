import jax
import jax.numpy as jnp
import pytest

from jax_nan_debugger import find_nan_source, nan_trace


def test_locates_log_of_negative():
    def f(x):
        return jnp.sum(jnp.log(x - 2.0) ** 2)

    report = find_nan_source(f, jnp.array([1.0, 3.0]))
    assert report.found
    assert report.first_site.primitive == "log"
    assert not report.first_site.inputs_had_nan


def test_blames_origin_not_propagation():
    def f(x):
        bad = jnp.sqrt(x)        # NaN origin (x < 0)
        return jnp.exp(bad) + 1  # propagation, must not be blamed

    report = find_nan_source(f, jnp.array([-1.0]))
    assert report.first_site.primitive == "sqrt"


def test_recurses_into_jit():
    @jax.jit
    def f(x):
        return jnp.log(x)

    report = find_nan_source(f, jnp.array([-1.0]))
    assert report.first_site.primitive == "log"


def test_clean_run_returns_outputs():
    def f(x):
        return jnp.sum(x ** 2)

    report = find_nan_source(f, jnp.array([1.0, 2.0]))
    assert not report.found
    assert jnp.allclose(report.outputs, 5.0)


def test_nan_in_inputs_flagged():
    report = find_nan_source(jnp.sum, jnp.array([jnp.nan, 1.0]))
    assert report.nan_in_inputs
    assert report.first_site is None


def test_zero_div_zero():
    def f(x):
        return x / jnp.sum(x)

    report = find_nan_source(f, jnp.array([0.0, 0.0]))
    assert report.first_site.primitive == "div"


def test_internal_nan_literal_not_blamed():
    # jnp.std contains where(n - ddof > 0, var, nan): an intentional NaN
    # literal that gets selected away. The real origin here is the 0/0 div
    # from the constant column, not std's internal plumbing.
    @jax.jit
    def standardize(x):
        return (x - jnp.mean(x, axis=0)) / jnp.std(x, axis=0)

    x = jnp.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    report = find_nan_source(standardize, x)
    assert report.first_site.primitive == "div"
    assert not report.first_site.inputs_had_nan


def test_clean_run_with_internal_nan_literal():
    # No constant column: std's internal NaN literal must not be reported.
    def f(x):
        return (x - jnp.mean(x)) / jnp.std(x)

    report = find_nan_source(f, jnp.array([1.0, 2.0, 3.0]))
    assert not report.found


def test_nan_trace_decorator():
    @nan_trace
    def f(x):
        return jnp.sqrt(x)

    assert f(jnp.array([4.0])) == 2.0
    with pytest.raises(FloatingPointError, match="sqrt"):
        f(jnp.array([-1.0]))


def test_kwargs_supported():
    def f(x, shift=0.0):
        return jnp.log(x + shift)

    report = find_nan_source(f, jnp.array([0.5]), shift=-1.0)
    assert report.first_site.primitive == "log"

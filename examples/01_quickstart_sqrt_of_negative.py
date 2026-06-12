"""Demo: locate the NaN source in a small forward computation."""

import jax
import jax.numpy as jnp

from jax_nan_debugger import find_nan_source, nan_trace


def model(x, w):
    h = jnp.tanh(x @ w)
    h = jnp.sqrt(h)          # <-- NaN here: tanh output can be negative
    h = h * 3.0 + 1.0        # NaN propagates silently...
    return jnp.mean(h)       # ...and surfaces far away, as a NaN mean.


@jax.jit
def jitted_model(x, w):
    return model(x, w)


def main():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 8))
    w = jax.random.normal(jax.random.split(key)[1], (8, 8))

    print("Plain run gives:", jitted_model(x, w), "\n")

    print("=== find_nan_source ===")
    print(find_nan_source(model, x, w))

    print("\n=== find_nan_source through jit (recurses into pjit) ===")
    print(find_nan_source(jitted_model, x, w))

    print("\n=== nan_trace decorator ===")
    try:
        nan_trace(model)(x, w)
    except FloatingPointError as e:
        print(f"FloatingPointError:\n{e}")

    print("\n=== clean run ===")
    print(find_nan_source(model, jnp.abs(x) * 0.01, jnp.abs(w) * 0.01))


if __name__ == "__main__":
    main()

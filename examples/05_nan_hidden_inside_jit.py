"""NaNs inside jitted code are the worst to debug by hand.

With `jax.config.update("jax_debug_nans", True)`, a NaN inside a @jax.jit
function is reported against the compiled XLA computation as a whole — no
user line number. find_nan_source recurses into the pjit sub-jaxpr and
points at the exact primitive and source line inside the jitted function.

The bug: standardizing a constant feature column gives std = 0, so the
division is 0/0.
"""

import jax
import jax.numpy as jnp

from jax_nan_debugger import find_nan_source


@jax.jit
def standardize(x):
    return (x - jnp.mean(x, axis=0)) / jnp.std(x, axis=0)


def main():
    # Second column is constant -> std = 0 -> 0/0 in the div.
    x = jnp.array([[1.0, 5.0],
                   [2.0, 5.0],
                   [3.0, 5.0]])

    print("Jitted run gives:\n", standardize(x), "\n")
    print(find_nan_source(standardize, x))


if __name__ == "__main__":
    main()

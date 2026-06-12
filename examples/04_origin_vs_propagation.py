"""The NaN you observe is rarely where the NaN was born.

Here a `log` goes NaN in the very first layer, then flows through three
matmuls and a relu before surfacing as a NaN loss. Checking only the loss
tells you nothing; the report walks the computation in order and blames
the first equation with finite inputs but NaN output — the `log`.
"""

import jax
import jax.numpy as jnp

from jax_nan_debugger import find_nan_source


def deep_model(x, ws):
    h = jnp.log(1.0 - x)            # <-- NaN born here when any x > 1
    for w in ws:                    # ... then silently carried through
        h = jax.nn.relu(h @ w)      #     three layers of matmul+relu ...
    return jnp.mean(h ** 2)         # ... and observed only in the loss.


def main():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 8)) * 2.0  # some entries exceed 1
    ws = [jax.random.normal(jax.random.key(i), (8, 8)) for i in range(3)]

    print("Loss value:", deep_model(x, ws), "\n")
    print(find_nan_source(deep_model, x, ws))


if __name__ == "__main__":
    main()

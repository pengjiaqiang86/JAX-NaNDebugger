"""Sometimes no operation is to blame — the NaN came in with your data.

find_nan_source checks the inputs first and reports that case separately,
so you don't waste time staring at the model when the dataset (a corrupt
sensor reading, a bad join) is the real culprit.
"""

import jax.numpy as jnp

from jax_nan_debugger import find_nan_source


def model(x):
    return jnp.mean(x ** 2)


def main():
    clean = jnp.array([1.0, 2.0, 3.0])
    corrupt = jnp.array([1.0, jnp.nan, 3.0])

    print("=== corrupt input ===")
    print(find_nan_source(model, corrupt))

    print("\n=== clean input ===")
    print(find_nan_source(model, clean))


if __name__ == "__main__":
    main()

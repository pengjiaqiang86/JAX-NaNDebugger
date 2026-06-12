"""A weighted mean where all weights happen to be zero: 0/0 = NaN.

This is a common silent failure when a mask selects no elements (empty
batch, fully padded sequence). The report blames the `div` and the input
stats show both numerator and denominator are exactly zero.
"""

import jax.numpy as jnp

from jax_nan_debugger import find_nan_source


def masked_mean(x, mask):
    return jnp.sum(x * mask) / jnp.sum(mask)


def main():
    x = jnp.array([1.0, 2.0, 3.0])

    print("=== mask selects nothing -> 0/0 ===")
    print(find_nan_source(masked_mean, x, jnp.zeros(3)))

    print("\n=== non-empty mask is fine ===")
    print(find_nan_source(masked_mean, x, jnp.array([1.0, 1.0, 0.0])))


if __name__ == "__main__":
    main()

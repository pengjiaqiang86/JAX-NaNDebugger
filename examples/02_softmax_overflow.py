"""Naively implemented softmax overflows for large logits.

exp(1000) = inf, and inf/inf = NaN. The report blames the `div` and its
input stats show the infs that caused it — the classic hint to subtract
the max logit first (or just use jax.nn.softmax).
"""

import jax.numpy as jnp

from jax_nan_debugger import find_nan_source


def naive_softmax(logits):
    e = jnp.exp(logits)          # overflows to inf for large logits
    return e / jnp.sum(e)        # inf / inf -> NaN


def stable_softmax(logits):
    e = jnp.exp(logits - jnp.max(logits))
    return e / jnp.sum(e)


def main():
    logits = jnp.array([10.0, 500.0, 1000.0])

    print("=== naive softmax ===")
    print(find_nan_source(naive_softmax, logits))

    print("\n=== stable softmax ===")
    print(find_nan_source(stable_softmax, logits))


if __name__ == "__main__":
    main()

"""Use @nan_trace to guard a loop over many batches at full speed.

The scan in find_nan_source runs op-by-op and is slow, so you don't want
it on every iteration. @nan_trace runs the function normally and only
re-scans when the output actually contains a NaN — then raises a
FloatingPointError telling you which batch and which operation.

The bug: batch 3 contains a probability of exactly 0, and the entropy
term 0 * log(0) = 0 * (-inf) = NaN.
"""

import jax.numpy as jnp

from jax_nan_debugger import nan_trace


@nan_trace
def entropy(p):
    return -jnp.sum(p * jnp.log(p))


def main():
    batches = [
        jnp.array([0.5, 0.5]),
        jnp.array([0.9, 0.1]),
        jnp.array([0.3, 0.3, 0.4]),
        jnp.array([1.0, 0.0]),   # <-- p=0 here: 0 * log(0) = NaN
        jnp.array([0.2, 0.8]),
    ]

    for i, batch in enumerate(batches):
        try:
            print(f"batch {i}: entropy = {entropy(batch):.4f}")
        except FloatingPointError as e:
            print(f"batch {i}: FAILED\n{e}")
            break


if __name__ == "__main__":
    main()

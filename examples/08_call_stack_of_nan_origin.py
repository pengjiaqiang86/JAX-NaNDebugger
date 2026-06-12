"""The report shows the full call chain leading to the NaN, not just a line.

In real code the offending primitive sits deep inside nested helpers, and
the same helper may be called from many places. The call stack tells you
*which* call path produced the NaN: here, `attention` -> `softmax_weights`
-> the `div` — reached from `transformer_block`, not from anywhere else.
"""

import jax
import jax.numpy as jnp

from jax_nan_debugger import find_nan_source


def softmax_weights(scores):
    e = jnp.exp(scores)              # overflows for large scores
    return e / jnp.sum(e, axis=-1, keepdims=True)


def attention(q, k, v):
    scores = q @ k.T * 1000.0        # oops: forgot 1/sqrt(d) scaling
    return softmax_weights(scores) @ v


def transformer_block(x, wq, wk, wv):
    return x + attention(x @ wq, x @ wk, x @ wv)


def main():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 8))
    wq, wk, wv = (jax.random.normal(jax.random.key(i), (8, 8)) for i in range(3))

    print(find_nan_source(transformer_block, x, wq, wk, wv))


if __name__ == "__main__":
    main()

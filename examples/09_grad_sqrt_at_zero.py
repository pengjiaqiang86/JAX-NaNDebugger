"""The most common backward-pass trap: an infinite derivative meets zero.

`sqrt(maximum(x, 0))` is perfectly fine in the forward pass — sqrt(0) is 0.
But the derivative of sqrt at 0 is `0.5/sqrt(0) = inf`, and the chain rule
multiplies it by the derivative of `maximum`, which is 0 for x < 0:
`inf * 0 = NaN`. The loss is finite, the gradient is NaN.

find_grad_nan_source scans the forward pass first (clean), then the VJP
computation, and blames the backward `mul` — with the call stack pointing
at the *forward* line that generated it, since that is the line to fix.

Also shown: `jax.nn.relu` instead of `jnp.maximum` is clean, because relu
defines a custom JVP precisely to avoid this trap.
"""

import jax
import jax.numpy as jnp

from jax_nan_debugger import find_grad_nan_source


def loss_naive(x):
    return jnp.sum(jnp.sqrt(jnp.maximum(x, 0.0)))


def loss_with_relu(x):
    return jnp.sum(jnp.sqrt(jax.nn.relu(x)))


def main():
    x = jnp.array([-1.0, 0.0, 4.0])

    print("loss:", loss_naive(x), "  grad:", jax.grad(loss_naive)(x), "\n")

    print("=== naive maximum(x, 0) ===")
    print(find_grad_nan_source(loss_naive, x))

    print("\n=== jax.nn.relu (custom JVP) ===")
    print(find_grad_nan_source(loss_with_relu, x))


if __name__ == "__main__":
    main()

"""The infamous `where` gradient trap.

`jnp.where(x > 0, jnp.log(x), 0.0)` looks safe: the log of a non-positive
value is masked out and the forward pass contains no NaN. But the VJP
still differentiates *both* branches — the masked branch receives
cotangent `0 * d(log)/dx = 0 * (1/0) = 0 * inf = NaN` at x = 0.

The report blames the backward `div` (the `1/x` of log's derivative) and
points at the `jnp.log` line. The standard fix is to sanitize the *input*
with an inner where, so the dangerous value never reaches log at all:
`jnp.log(jnp.where(x > 0, x, 1.0))`.
"""

import jax
import jax.numpy as jnp

from jax_nan_debugger import find_grad_nan_source


def loss_trap(x):
    return jnp.sum(jnp.where(x > 0, jnp.log(x), 0.0))


def loss_fixed(x):
    safe_x = jnp.where(x > 0, x, 1.0)
    return jnp.sum(jnp.where(x > 0, jnp.log(safe_x), 0.0))


def main():
    x = jnp.array([0.0, 1.0, 2.0])

    print("forward values are identical:")
    print("  trap :", loss_trap(x), " grad:", jax.grad(loss_trap)(x))
    print("  fixed:", loss_fixed(x), " grad:", jax.grad(loss_fixed)(x), "\n")

    print("=== outer where only (trap) ===")
    print(find_grad_nan_source(loss_trap, x))

    print("\n=== inner where sanitizes the input (fixed) ===")
    print(find_grad_nan_source(loss_fixed, x))


if __name__ == "__main__":
    main()

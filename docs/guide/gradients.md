# Debugging gradients

Backward-pass NaNs are the nastiest kind: the loss is finite, the forward
pass is healthy, and the operation that produced the NaN was synthesized by
the VJP transform — it appears nowhere in your source.

## The shape of the problem

Almost every backward NaN is the same pattern: **the forward value is fine,
but the local derivative is infinite at the evaluated point, and the chain
rule multiplies that infinity by zero.**

| Trap | Forward | Backward |
|---|---|---|
| `sqrt(maximum(x, 0))` at `x < 0` | `sqrt(0) = 0` ✓ | `sqrt'(0) = inf`, times `maximum`'s gradient `0` → NaN |
| `where(x > 0, log(x), 0)` at `x = 0` | masked ✓ | the dead branch still gets `0 * (1/0)` → NaN |
| RMSE loss at a perfect fit | `sqrt(0) = 0` ✓ | `err / rmse = 0/0` → NaN |

## `find_grad_nan_source`

```python
import jax.numpy as jnp
from jax_nan_debugger import find_grad_nan_source

def loss(x):
    return jnp.sum(jnp.where(x > 0, jnp.log(x), 0.0))

report = find_grad_nan_source(loss, jnp.array([0.0, 1.0]))
print(report)
```

```text
✘ NaNs first produced by: div  (backward pass)
│ call stack (most recent call last):
│   ▶ script.py:5 in loss
│ eqn: a:f32[2] = div b c
...
```

Two things to notice:

1. The headline says **`(backward pass)`** — the scan checked the forward
   pass first and found it clean, so this NaN is purely a gradient artifact.
2. The call stack points at the **`jnp.log` line** even though the blamed
   `div` is in the backward pass. JAX attributes backward equations to the
   forward line that generated them — which is exactly the line you need
   to fix.

## How the two-stage scan works

1. The forward pass is scanned with `find_nan_source`. A forward NaN would
   contaminate every gradient, so if one exists it is reported (tagged
   `phase="forward"`) and the backward scan is skipped.
2. Otherwise the full `jax.vjp` pullback is traced and scanned equation by
   equation; the origin is tagged `phase="backward"`.

Gradients are taken with respect to **all positional arguments** (arbitrary
pytrees), so NaNs in data-gradients are found too. Keyword arguments are
non-differentiated constants, mirroring `jax.grad`.

## Non-scalar outputs

By default the function must return a scalar (like a loss) and the scan uses
ordinary `jax.grad` semantics. For non-scalar outputs, pass the cotangents to
pull back:

```python
report = find_grad_nan_source(f, x, cotangents=jnp.ones_like(f_output))
```

## Fixing the traps

- **`sqrt` / `norm` at zero** — add an epsilon *inside* the sqrt
  (`jnp.sqrt(x + eps)`), or use a squared loss (MSE instead of RMSE).
- **The `where` trap** — sanitize the *input*, not the output, with an inner
  `where`, so the dangerous value never reaches the dangerous op:
  `jnp.log(jnp.where(x > 0, x, 1.0))`.
- **Use the library versions** — `jax.nn.relu` defines a custom JVP exactly
  so that `sqrt(relu(x))` does *not* NaN where `sqrt(maximum(x, 0))` does.
  Likewise prefer `jax.nn.softmax`, `jax.nn.log_softmax`,
  `jax.nn.logsumexp` over hand-rolled versions.

!!! tip "A subtle one for test writers"
    At exactly `x = 0`, `jnp.maximum(x, 0)`'s tie-gradient is 0.5 — the
    chain gives `inf * 0.5 = inf`, an **inf, not a NaN**. You need a
    strictly negative element (gradient factor exactly 0) to produce
    `0 * inf = NaN`.

## A realistic failure: RMSE on a perfectly-fit batch

The feed-forward network example
([`examples/11_grad_feedforward_network.py`](../examples.md)) trains an MLP
with `sqrt(mean((pred - y)²))`. On a batch the network fits exactly, the
loss is a clean `0.0` — and every gradient in the network is NaN, because
the derivative of RMSE is `err / (n · rmse) = 0/0`. One such batch silently
poisons all parameters. The report blames the backward `div` and walks you
to the `jnp.sqrt` in the loss.

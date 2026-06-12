# jax-nan-debugger

**Pinpoint the exact operation that first produces a NaN in a JAX computation — forward or backward.**

## The problem

`jax.config.update("jax_debug_nans", True)` tells you *that* a NaN appeared, but:

- inside jitted code it points at an opaque XLA computation, not your source line;
- by the time a NaN is observed it has propagated far from its origin — the
  symptom (a NaN loss) is rarely the cause (a `log` three layers earlier);
- backward-pass NaNs are worse: the offending operation was synthesized by
  the VJP transform and never appears in your code at all.

## The approach

This package traces your function to a jaxpr and re-evaluates it **one
primitive at a time**, checking every intermediate value. The first equation
whose inputs are NaN-free but whose output contains NaNs is the true origin —
reported with your call stack, the primitive, and value statistics:

```text
✘ NaNs first produced by: div  (backward pass)
│ call stack (most recent call last):
│       train.py:88 in train_step
│     ▶ train.py:41 in rmse_loss
│ eqn: a:f32[32,1] = div b c
│ inputs:
│   [0] float32[32, 1] range=[0, 0] nans=0 infs=0
│   [1] float32[32, 1] range=[0, 0] nans=0 infs=0
╰ outputs:
    [0] float32[32, 1] all non-finite nans=32 infs=0
```

## Install

```bash
git clone https://github.com/pengjiaqiang86/JAX-NaNDebugger
cd JAX-NaNDebugger
pip install -e .
```

Requires Python ≥ 3.10 and JAX ≥ 0.4.30.

## 30-second tour

```python
import jax.numpy as jnp
from jax_nan_debugger import find_nan_source, find_grad_nan_source, nan_trace

def loss(x):
    return jnp.sum(jnp.where(x > 0, jnp.log(x), 0.0))

# Forward scan: this one is clean (the where masks log(0)).
print(find_nan_source(loss, jnp.array([0.0, 1.0])))
# ✔ No NaNs found in any intermediate or output value.

# Gradient scan: the infamous where-trap is caught and located.
print(find_grad_nan_source(loss, jnp.array([0.0, 1.0])))
# ✘ NaNs first produced by: div  (backward pass)  ...

# Production guard: full speed until a NaN actually appears.
guarded = nan_trace(loss)
```

## Where to go next

- [Debugging forward runs](guide/forward.md) — `find_nan_source` and `nan_trace`
- [Debugging gradients](guide/gradients.md) — `find_grad_nan_source` and the classic backward traps
- [Reading the report](guide/report.md) — every field of the output explained
- [Wrapping any transform](guide/transforms.md) — `jvp`, `vjp`, `linearize`, higher-order grads
- [Examples](examples.md) — eleven runnable scripts, one per failure mode
- [API reference](api.md)

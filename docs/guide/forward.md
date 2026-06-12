# Debugging forward runs

## One-shot scan: `find_nan_source`

Hand it the function and the arguments that produce a NaN:

```python
import jax.numpy as jnp
from jax_nan_debugger import find_nan_source

def model(x, w):
    h = jnp.tanh(x @ w)
    h = jnp.sqrt(h)          # NaN here: tanh output can be negative
    return jnp.mean(h * 3.0)

report = find_nan_source(model, x, w)
print(report)
```

```text
✘ NaNs first produced by: sqrt
│ call stack (most recent call last):
│   ▶ model.py:6 in model
│ eqn: a:f32[4,8] = sqrt b
│ inputs:
│   [0] float32[4, 8] range=[-1, 1] nans=0 infs=0
╰ outputs:
    [0] float32[4, 8] range=[0.9583, 1] nans=21 infs=0
```

The input stats already tell the story: `sqrt` received values down to `-1`.

Arguments can be arbitrary pytrees; keyword arguments are supported and
treated as constants. If the run is clean, `report.found` is `False` and
`report.outputs` holds the function's result.

## Origin, not propagation

The scan blames the **first equation whose inputs are NaN-free but whose
output contains NaNs**. Downstream operations that merely carry the NaN are
never blamed — so a NaN born in layer 1 and observed in the loss is reported
at layer 1.

Two refinements of this rule matter in practice:

!!! note "Intentional NaN literals are not blamed"
    JAX library code routes NaN literals through `where`-style guards on
    purpose — `jnp.std`, for instance, contains
    `where(n - ddof > 0, var, nan)`. A NaN flowing *through* an equation is
    therefore not treated as evidence of a bug; only producing a NaN from
    clean inputs is. If a NaN ever enters purely via a constant and reaches
    the output, the report falls back to the first place it was carried.

!!! note "NaNs already in your inputs"
    If the arguments themselves contain NaNs, the report says exactly that
    instead of blaming the first operation that touched them — check your
    data pipeline, not the model.

## Inside `jit`

The scan recurses into `jit`/`pjit`, `custom_jvp`/`custom_vjp`, and remat
sub-jaxprs, so a NaN inside a jitted function is attributed to the precise
primitive and source line, not to an opaque compiled call:

```python
@jax.jit
def standardize(x):
    return (x - jnp.mean(x, axis=0)) / jnp.std(x, axis=0)

print(find_nan_source(standardize, x_with_constant_column))
# ✘ NaNs first produced by: div   ← the 0/0, at the line above
```

## Guarding hot loops: `nan_trace`

The op-by-op scan is slow (it is a Python interpreter over the jaxpr). For
training loops, decorate the function instead:

```python
from jax_nan_debugger import nan_trace

@nan_trace
def loss_fn(params, batch):
    ...

for step, batch in enumerate(data):
    loss = loss_fn(params, batch)   # full speed; raises on the bad batch
```

The wrapped function runs normally — jitted code stays jitted — and only when
an output actually contains a NaN is the call re-run through the scanner. The
raised `FloatingPointError` carries the full report, and your own traceback
tells you which step/batch triggered it.

## Performance and caveats

- The scan is **debug-speed only**: orders of magnitude slower than compiled
  execution. Use `nan_trace` so you pay only on failure.
- XLA fusion may reorder floating-point operations, so in rare cases the
  compiled run NaNs where the un-jitted scan does not (or vice versa). The
  blamed site is still almost always the right place to look.
- Control-flow bodies (`lax.scan`, `lax.cond`, `lax.while_loop`) are checked
  at the granularity of the whole control-flow primitive.

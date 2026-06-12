# jax-nan-debugger

Pinpoint the exact operation that first produces a NaN in a JAX computation.

`jax.config.update("jax_debug_nans", True)` tells you *that* a NaN appeared, but
inside jitted code it often points at an opaque XLA computation, and by the
time you see the NaN it has propagated far from its origin. This package
re-evaluates your function's jaxpr one primitive at a time and blames the
**first** equation whose inputs are finite but whose output contains NaN —
with the primitive name, your source file/line, and value statistics.

## Install

```bash
pip install -e .
```

## Usage

### One-shot scan

```python
import jax.numpy as jnp
from jax_nan_debugger import find_nan_source

def model(x):
    y = jnp.log(x - 2.0)   # NaN for x < 2
    return jnp.sum(y ** 2)

report = find_nan_source(model, jnp.array([1.0, 3.0]))
print(report)
# ✘ NaNs first produced by: log
# │ call stack (most recent call last):
# │       your_script.py:12 in main
# │     ▶ your_script.py:5 in model
# │ eqn: b:f32[2] = log a
# │ inputs:
# │   [0] float32[2] range=[-1, 1] nans=0 infs=0
# ╰ outputs:
#     [0] float32[2] range=[0, 0] nans=1 infs=0
```

The call stack is the full chain of *your* functions leading to the
offending primitive (JAX internals are filtered out), so when the same
helper is called from many places you can see which call path produced
the NaN. On a terminal the report is colorized — the origin frame, the
primitive, and any non-zero `nans=`/`infs=` counts are highlighted.
Color is disabled automatically when output is piped, or explicitly via
the `NO_COLOR` env var (`FORCE_COLOR` forces it on); `report.render(color=...)`
gives explicit control.

### Gradient (backward-pass) NaNs

```python
from jax_nan_debugger import find_grad_nan_source

def loss(x):
    return jnp.sum(jnp.where(x > 0, jnp.log(x), 0.0))  # forward: fine
                                                       # grad: NaN at x=0

report = find_grad_nan_source(loss, jnp.array([0.0, 1.0]))
print(report)
# ✘ NaNs first produced by: div  (backward pass)
# │   ▶ your_script.py:4 in loss
# ...
```

The forward pass is scanned first (a forward NaN would contaminate every
gradient); if it is clean, the VJP computation is scanned and the origin
is tagged ``(backward pass)``. Backward equations are attributed to the
*forward* line that generated them — i.e. the line you need to fix.
Non-scalar outputs are supported via ``cotangents=...``; gradients are
taken with respect to all positional arguments (arbitrary pytrees).

### Works with any transform

`find_nan_source` only needs a JAX-traceable callable, and `jax.make_jaxpr`
traces through arbitrary transform compositions — so wrapping the
*transformed* function is fully supported:

```python
find_nan_source(jax.grad(f), x)                  # reverse-mode
find_nan_source(lambda x, t: jax.jvp(f, (x,), (t,))[1], x, t)   # forward-mode
_, pullback = jax.vjp(f, x);   find_nan_source(pullback, ct)
_, f_lin    = jax.linearize(f, x);   find_nan_source(f_lin, t)
find_nan_source(jax.linear_transpose(g, x), ct)
find_nan_source(jax.grad(jax.grad(f)), x)        # higher-order too
nan_trace(jax.grad(f))                           # the decorator composes the same way
```

Compared to this generic pattern, `find_grad_nan_source` adds two things:
the forward/backward `phase` tag in the report, and a whole-pipeline scan.
The latter matters for pre-computed `vjp`/`linearize` pullbacks: their
residuals were evaluated *before* the scan, so a NaN born in the forward
pass enters the pullback as an opaque constant and only the weaker
"propagated" report is possible. When in doubt, hand the *primal* function
to `find_grad_nan_source`.

### Decorator (zero-cost until a NaN appears)

```python
from jax_nan_debugger import nan_trace

@nan_trace
def model(x): ...

model(x)  # raises FloatingPointError with the located origin if output has NaN
```

The scan recurses into `jit`/`pjit`, `custom_jvp`/`custom_vjp` and remat
sub-jaxprs, so the report points at the innermost responsible primitive, not
an opaque call.

If the NaN is already in your *inputs*, the report says so instead of blaming
an operation.

## Examples

Each script in [examples/](examples/) is a self-contained NaN scenario:

| Script | Scenario |
|---|---|
| `01_quickstart_sqrt_of_negative.py` | Tour of the API: `sqrt` of a negative intermediate, scan through jit, the decorator, a clean run |
| `02_softmax_overflow.py` | Naive softmax: `exp` overflow makes `inf/inf` |
| `03_zero_divided_by_zero.py` | Masked mean where the mask selects nothing: `0/0` |
| `04_origin_vs_propagation.py` | NaN born in layer 1, observed in the loss — the origin gets blamed, not the symptom |
| `05_nan_hidden_inside_jit.py` | NaN inside `@jax.jit`, located at its source line (also exercises `jnp.std`'s intentional internal NaN literal, which is correctly *not* blamed) |
| `06_nan_trace_in_data_loop.py` | `@nan_trace` guarding a loop over batches at full speed |
| `07_nan_already_in_inputs.py` | NaN already present in the data, reported as such |
| `08_call_stack_of_nan_origin.py` | NaN deep in nested helpers — the report shows the full call chain (`transformer_block` → `attention` → `softmax_weights`) |
| `09_grad_sqrt_at_zero.py` | Backward trap: `sqrt'` is inf where the value is 0, and `inf * 0 = NaN` in the chain rule; `jax.nn.relu`'s custom JVP shown as the safe variant |
| `10_grad_where_log_trap.py` | The infamous `where` gradient trap — the forward pass masks `log(0)`, the backward pass still differentiates the dead branch; inner-`where` fix shown |
| `11_grad_feedforward_network.py` | Feed-forward NN with RMSE loss: a perfectly-fit batch gives a finite loss but all-NaN gradients (`0/0` in sqrt's VJP) |

Run them from the repo root after `pip install -e .`:

```bash
python examples/05_nan_hidden_inside_jit.py
```

## Notes & limitations

- The scan runs **un-jitted**, one primitive at a time — slow, for debugging
  only. Use `nan_trace` so the cost is only paid when a NaN actually occurs.
- The first NaN under op-by-op evaluation can occasionally differ from the
  compiled run (XLA fusion may rearrange floating-point operations), but the
  blamed site is almost always the right place to look.
- Control-flow bodies (`lax.scan`, `lax.cond`, `lax.while_loop`) are currently
  checked at the granularity of the whole control-flow primitive.

## Roadmap

- [x] Step 1: forward-run NaN source location
- [x] Step 2: inverse (gradient/VJP) NaN source location

# Reading the report

Every scan returns a [`NanReport`][jax_nan_debugger.NanReport]. Printing it
gives one of three messages.

## The three outcomes

```text
✘ NaNs are already present in the function inputs.
```
The NaN came in with your data — check the pipeline, not the model.
Programmatically: `report.nan_in_inputs` is `True`.

```text
✔ No NaNs found in any intermediate or output value.
```
Clean run. `report.found` is `False` and `report.outputs` holds the result
(for gradient scans: the gradients, one entry per positional argument).

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
An origin was located: `report.first_site` is a
[`NanSite`][jax_nan_debugger.NanSite].

## Anatomy of a located site

| Part | Meaning |
|---|---|
| headline | The primitive that produced the NaN. `(forward pass)`/`(backward pass)` appears for gradient scans. |
| call stack | The chain of *your* functions leading to the equation, outermost first; JAX internals and this package's own frames are filtered out. The `▶` marks the frame where the NaN was born. For backward sites this points at the forward line that generated the gradient — the line to fix. |
| `eqn` | The jaxpr equation, e.g. `a:f32[32,1] = div b c`. |
| `inputs` / `outputs` | One line per operand: dtype, shape, the range of *finite* elements, and NaN/inf counts. |

The input stats usually explain the *why* at a glance:

- `inf` counts on an input + NaN output → an `inf - inf`, `inf/inf`, or `0 * inf`;
- `range=[0, 0]` on both operands of a `div` → a `0/0`;
- a negative lower bound into `sqrt`/`log` → domain error.

## "first produced" vs "propagated"

The scan blames the first equation with **NaN-free inputs and NaN output**.
Equations that merely carry an existing NaN are never blamed — JAX library
code intentionally routes NaN literals through `where` guards (e.g. inside
`jnp.std`), and those must not be reported. In the rare case where a NaN
enters purely via a constant and reaches the output, the headline switches
to `NaNs propagated from inputs of: ...` as a fallback.

## Color

Reports are colorized when printed to an interactive terminal: the headline
and origin marker in red, the primitive in magenta, paths in cyan, and any
**non-zero** `nans=`/`infs=` counts highlighted so the cause jumps out.

- Output piped to a file or another program is automatically plain.
- `NO_COLOR=1` forces plain; `FORCE_COLOR=1` forces color.
- `report.render(color=True/False)` gives explicit programmatic control;
  `str(report)` auto-detects.

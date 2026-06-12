# Examples

Eleven runnable scripts live in
[`examples/`](https://github.com/pengjiaqiang86/JAX-NaNDebugger/tree/master/examples),
one per failure mode. Run any of them from the repo root after
`pip install -e .`:

```bash
python examples/11_grad_feedforward_network.py
```

## Forward-pass scenarios

| Script | Scenario |
|---|---|
| `01_quickstart_sqrt_of_negative.py` | Tour of the API: `sqrt` of a negative intermediate, scan through jit, the decorator, a clean run |
| `02_softmax_overflow.py` | Naive softmax: `exp` overflow makes `inf/inf` |
| `03_zero_divided_by_zero.py` | Masked mean where the mask selects nothing: `0/0` |
| `04_origin_vs_propagation.py` | NaN born in layer 1, observed in the loss — the origin gets blamed, not the symptom |
| `05_nan_hidden_inside_jit.py` | NaN inside `@jax.jit`, located at its source line; also shows that `jnp.std`'s intentional internal NaN literal is *not* blamed |
| `06_nan_trace_in_data_loop.py` | `@nan_trace` guarding a loop over batches at full speed; fails on the batch containing `0 * log(0)` |
| `07_nan_already_in_inputs.py` | NaN already present in the data, reported as such |
| `08_call_stack_of_nan_origin.py` | NaN deep in nested attention helpers — the report shows the full call chain |

## Backward-pass scenarios

| Script | Scenario |
|---|---|
| `09_grad_sqrt_at_zero.py` | `sqrt'` is inf where the value is 0; `inf * 0 = NaN` in the chain rule. `jax.nn.relu`'s custom JVP shown as the safe variant |
| `10_grad_where_log_trap.py` | The infamous `where` gradient trap — forward is masked, backward still differentiates the dead branch; inner-`where` fix shown |
| `11_grad_feedforward_network.py` | Feed-forward NN with RMSE loss: a perfectly-fit batch gives a finite loss but all-NaN gradients (`0/0` in sqrt's VJP) |

## Sample output

From the feed-forward network example — finite loss, all-NaN gradients:

```text
loss = 0.0  (finite!)   first-layer grad w: [nan nan nan] ...

=== RMSE loss, perfectly-fit batch ===
✘ NaNs first produced by: mul  (backward pass)
│ call stack (most recent call last):
│   ▶ examples/11_grad_feedforward_network.py:41 in rmse_loss
│ eqn: a:f32[32,1] = mul b c
│ inputs:
│   [0] float32[32, 1] all non-finite nans=0 infs=32
│   [1] float32[32, 1] range=[0, 0] nans=0 infs=0
╰ outputs:
    [0] float32[32, 1] all non-finite nans=32 infs=0

=== RMSE loss, normal batch ===
✔ No NaNs found in any intermediate or output value.

=== MSE loss, perfectly-fit batch (the fix) ===
✔ No NaNs found in any intermediate or output value.
```

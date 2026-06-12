# Wrapping any transform

`find_nan_source` only needs a JAX-traceable callable — and `jax.make_jaxpr`
traces through arbitrary compositions of transforms. So beyond the built-in
forward and gradient scans, you can hand it any transformed function
directly:

```python
import jax
from jax_nan_debugger import find_nan_source, nan_trace

find_nan_source(jax.grad(f), x)                    # reverse-mode
find_nan_source(lambda x, t: jax.jvp(f, (x,), (t,))[1], x, t)   # forward-mode

_, pullback = jax.vjp(f, x)
find_nan_source(pullback, cotangent)               # a vjp pullback

_, f_lin = jax.linearize(f, x)
find_nan_source(f_lin, tangent)                    # a linearized function

find_nan_source(jax.linear_transpose(g, x), ct)    # a transposed function

find_nan_source(jax.grad(jax.grad(f)), x)          # higher-order derivatives

nan_trace(jax.grad(f))                             # the decorator composes too
```

All of these scan correctly, and backward equations still carry the source
location of the forward line that generated them.

## When to prefer `find_grad_nan_source`

Compared to wrapping `jax.grad` yourself, the dedicated gradient scanner
adds two things:

**Phase tagging.** A scan of `jax.grad(f)` sees one flattened jaxpr; it
cannot tell you whether the blamed equation belongs to the primal or the
cotangent computation. `find_grad_nan_source` scans the forward pass
separately first, which is what makes the `(forward pass)` /
`(backward pass)` label possible.

**Pre-computed pullbacks hide forward NaNs.** When you call
`jax.vjp(f, x)` yourself, the residuals are evaluated eagerly — *before*
any scan. If the forward pass put a NaN into a residual, the pullback scan
sees that NaN entering via an opaque constant: no equation with clean
inputs ever produces it, so only the weaker "propagated" fallback report is
possible, with no true origin. `find_grad_nan_source` always scans the
whole pipeline from the primal inputs, so the forward origin is found.

!!! tip "Rule of thumb"
    For the everyday "my loss gradient is NaN", hand the **primal** function
    to `find_grad_nan_source`. Use generic wrapping for everything it does
    not cover: forward-mode `jvp`, transposes, higher-order derivatives,
    or any custom transform stack.

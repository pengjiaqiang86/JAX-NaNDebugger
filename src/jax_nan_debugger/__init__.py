"""jax-nan-debugger: pinpoint the operation that first produces NaNs in JAX code.

Why this exists
---------------
``jax.config.update("jax_debug_nans", True)`` reports *that* a NaN appeared,
but inside jitted code it points at an opaque XLA computation, and by the
time a NaN is observed it has usually propagated far from its origin. This
package instead traces the target function to a jaxpr and re-evaluates it
one primitive at a time, blaming the first equation whose inputs are
NaN-free but whose output contains NaNs — together with the user call
stack, the offending primitive, and value statistics for its inputs and
outputs.

Public API
----------
- :func:`find_nan_source` -- one-shot scan of a function call; returns a
  :class:`NanReport`.
- :func:`nan_trace` -- decorator that runs the function at full speed and
  only re-scans (then raises ``FloatingPointError``) when the output
  actually contains a NaN.
- :class:`NanReport` / :class:`NanSite` -- the structured result types,
  with colorized terminal rendering via ``str()`` or ``.render()``.

Quick start
-----------
    import jax.numpy as jnp
    from jax_nan_debugger import find_nan_source

    def model(x):
        return jnp.sum(jnp.log(x - 2.0) ** 2)   # NaN for x < 2

    report = find_nan_source(model, jnp.array([1.0, 3.0]))
    print(report)          # blames the `log`, with file:line and stats

Package layout
--------------
- ``forward.py`` -- public entry points for forward-run debugging.
- ``interpreter.py`` -- the NaN-checked jaxpr evaluator (the engine).
- ``report.py`` -- result types and terminal rendering.
- ``_jax_compat.py`` -- shims over JAX-internal APIs.

Inverse-run (gradient/VJP) debugging is planned as step 2 and will reuse
the same interpreter and report types.
"""

from .forward import find_nan_source, nan_trace
from .report import NanReport, NanSite

__all__ = ["find_nan_source", "nan_trace", "NanReport", "NanSite"]
__version__ = "0.1.0"

"""jax-nan-debugger: pinpoint the operation that first produces NaNs in JAX code.

Forward-run debugging (step 1):

    from jax_nan_debugger import find_nan_source

    report = find_nan_source(fn, *args)
    print(report)

Inverse-run debugging is planned as step 2.
"""

from .forward import NanReport, NanSite, find_nan_source, nan_trace

__all__ = ["find_nan_source", "nan_trace", "NanReport", "NanSite"]
__version__ = "0.1.0"

"""Shims over JAX internals that may move between versions.

Everything in the package that touches a non-public JAX API goes through
this module, so a JAX upgrade breaks (at most) one file. Two such APIs
are wrapped here:

- ``core``: the home of the jaxpr data structures (``Jaxpr``,
  ``ClosedJaxpr``, ``JaxprEqn``, ``Var``, ``Literal``, ``DropVar``).
  Public location since JAX 0.6 is ``jax.extend.core``; older versions
  exposed them as ``jax.core``.
- ``jax._src.source_info_util``: maps a jaxpr equation back to the user
  source code that created it. This is private API with no public
  equivalent; if it disappears, :func:`summarize_source` and
  :func:`call_stack` degrade to uninformative placeholders instead of
  breaking the package.
"""

from __future__ import annotations

import os

try:  # JAX >= 0.6: jaxpr classes live in jax.extend.core
    from jax.extend import core
except ImportError:  # pragma: no cover
    from jax import core

try:  # source_info_util is internal; degrade gracefully if it moves.
    from jax._src import source_info_util

    _OWN_DIR = os.path.dirname(__file__)

    def summarize_source(eqn: core.JaxprEqn) -> str:
        """Return the innermost user frame of an equation as one line.

        Args:
            eqn: A jaxpr equation carrying ``source_info`` recorded by JAX
                during tracing.

        Returns:
            A string of the form ``"path/to/file.py:42 in function_name"``,
            or ``"<unknown location>"`` when JAX recorded no user frame
            (e.g. for equations synthesized by transformations).
        """
        frame = source_info_util.user_frame(eqn.source_info.traceback)
        if frame is None:
            return "<unknown location>"
        return f"{frame.file_name}:{frame.start_line} in {frame.function_name}"

    def call_stack(eqn: core.JaxprEqn) -> list[str]:
        """Return the user call chain that created an equation.

        JAX stores a full traceback on every jaxpr equation. This extracts
        the *user* frames (JAX-internal frames are already excluded by
        ``source_info_util.user_frames``) and additionally drops frames
        from this package itself, so the chain starts at the caller's code.

        Args:
            eqn: A jaxpr equation carrying ``source_info``.

        Returns:
            Frames formatted as ``"file:line in function"``, ordered
            outermost first — i.e. like a Python traceback, "most recent
            call last". Empty if no user frames are available.
        """
        frames = [
            f
            for f in source_info_util.user_frames(eqn.source_info.traceback)
            if not f.file_name.startswith(_OWN_DIR)
        ]
        return [
            f"{f.file_name}:{f.start_line} in {f.function_name}"
            for f in reversed(frames)
        ]
except ImportError:  # pragma: no cover

    def summarize_source(eqn: core.JaxprEqn) -> str:
        """Fallback when ``source_info_util`` is unavailable: no location."""
        return "<source info unavailable in this JAX version>"

    def call_stack(eqn: core.JaxprEqn) -> list[str]:
        """Fallback when ``source_info_util`` is unavailable: empty stack."""
        return []

"""Report types and rendering shared by all NaN debuggers in this package.

Both the forward scan and the (future) inverse scan produce the same
:class:`NanReport` / :class:`NanSite`, so users learn one output format.

Rendering policy: ``str(report)`` colorizes with ANSI escape codes only
when stdout is an interactive terminal, honoring the ``NO_COLOR`` and
``FORCE_COLOR`` environment variables; piped or redirected output stays
plain text. Call ``report.render(color=True/False)`` for explicit control
(e.g. when writing to a log file that is later viewed with ``less -R``).

This module also hosts the two value-inspection helpers used by the
interpreter, :func:`has_nan` and :func:`value_stats`, since their output
feeds directly into the report.
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
from typing import Any

import jax
import jax.numpy as jnp


def has_nan(x: Any) -> bool:
    """Return True if ``x`` is a floating/complex array containing a NaN.

    Non-array values and arrays of non-inexact dtype (int, bool) can never
    hold NaNs and return False without forcing a device sync.

    Args:
        x: Any value that may appear in a jaxpr environment — a JAX array,
            a Python scalar, or an arbitrary object carried through.

    Returns:
        True only when ``x`` has an inexact dtype and at least one NaN
        element. Note this triggers a blocking device->host transfer for
        the reduction, which is acceptable in this debug-only code path.
    """
    if not isinstance(x, jax.Array) and not hasattr(x, "dtype"):
        return False
    if not jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact):
        return False
    return bool(jnp.any(jnp.isnan(x)))


def value_stats(x: Any) -> str:
    """Summarize an array as a one-line human-readable string.

    The summary is what the report prints for each input/output of the
    blamed equation, e.g.::

        float32[4, 8] range=[-1, 1] nans=0 infs=12

    The range covers only *finite* elements so a single inf does not wash
    out the useful information; when nothing is finite the range is
    replaced by ``"all non-finite"``. Non-inexact arrays show just dtype
    and shape, since NaN/inf counts would be meaningless.

    Args:
        x: An array-like value.

    Returns:
        A one-line summary: dtype, shape, finite range, NaN count, and
        inf count.
    """
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        return f"{x.dtype}{list(x.shape)}"
    n_nan = int(jnp.sum(jnp.isnan(x)))
    n_inf = int(jnp.sum(jnp.isinf(x)))
    finite = x[jnp.isfinite(x)] if n_nan or n_inf else x.ravel()
    rng = (
        f"range=[{float(jnp.min(finite)):.4g}, {float(jnp.max(finite)):.4g}]"
        if finite.size
        else "all non-finite"
    )
    return f"{x.dtype}{list(x.shape)} {rng} nans={n_nan} infs={n_inf}"


def _color_enabled() -> bool:
    """Decide whether auto-rendering should emit ANSI color codes.

    Precedence: ``NO_COLOR`` (any non-empty value) disables, then
    ``FORCE_COLOR`` enables, then fall back to whether stdout is a TTY.
    Evaluated at render time, not import time, so tests and pipes that
    swap ``sys.stdout`` behave correctly.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _Style:
    """Tiny ANSI styler; every method is a no-op when color is disabled.

    Each method wraps its argument in one SGR escape sequence and a reset.
    The methods are named for *roles* in the report rather than colors, so
    the palette can be retuned in one place: ``title`` (bold red) for the
    headline and origin marker, ``prim`` (bold magenta) for the primitive
    name, ``path`` (cyan) for file locations, ``fn`` (bold) for function
    names, ``dim`` for structural labels, ``bad`` (bold red) for non-zero
    NaN counts, ``warn`` (yellow) for non-zero inf counts, and ``good``
    (bold green) for the all-clear message.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.enabled else s

    def title(self, s):
        return self._w("1;31", s)  # bold red

    def prim(self, s):
        return self._w("1;35", s)  # bold magenta

    def path(self, s):
        return self._w("36", s)  # cyan

    def fn(self, s):
        return self._w("1", s)  # bold

    def dim(self, s):
        return self._w("2", s)

    def bad(self, s):
        return self._w("1;31", s)

    def warn(self, s):
        return self._w("33", s)

    def good(self, s):
        return self._w("1;32", s)


def _style_frame(frame: str, st: _Style) -> str:
    """Colorize one call-stack frame: cyan location, bold function name.

    Frames arrive as ``"file:line in func"`` from ``_jax_compat.call_stack``;
    a frame without the ``" in "`` separator is returned unchanged.
    """
    loc, sep, fn = frame.rpartition(" in ")
    if not sep:
        return frame
    return f"{st.path(loc)} in {st.fn(fn)}"


def _style_stats(stats: str, st: _Style) -> str:
    """Highlight non-zero ``nans=``/``infs=`` counts inside a stats line.

    Zero counts are left unstyled so the eye is drawn only to the values
    that explain the NaN (e.g. the ``infs=12`` input of an ``inf/inf``
    division).
    """
    stats = re.sub(r"nans=([1-9]\d*)", lambda m: st.bad(m.group(0)), stats)
    stats = re.sub(r"infs=([1-9]\d*)", lambda m: st.warn(m.group(0)), stats)
    return stats


def _one_line(s: str, limit: int = 100) -> str:
    """Collapse whitespace to single spaces and truncate with an ellipsis.

    Guards the report against multi-line equation reprs: when the opaque
    fallback blames a whole ``jit`` equation, ``str(eqn)`` would otherwise
    dump the entire sub-jaxpr into the report.
    """
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclasses.dataclass
class NanSite:
    """One jaxpr equation at which NaNs appeared, with its context.

    Attributes:
        primitive: Name of the JAX primitive (e.g. ``"div"``, ``"log"``).
        source: Innermost user source location as ``"file:line in func"``.
        eqn_repr: The jaxpr equation as a string, e.g.
            ``"a:f32[4,4] = div b c"``.
        input_stats: One :func:`value_stats` line per equation input, in
            positional order.
        output_stats: One :func:`value_stats` line per equation output.
        inputs_had_nan: False means this equation *produced* the NaN from
            NaN-free inputs (the true origin). True means the NaN was
            already present in an input and merely flowed through — only
            reported when no producing equation exists (e.g. a NaN that
            entered via a constant).
        call_stack: User call chain to this equation, outermost first;
            may be empty when JAX recorded no source info.
    """

    primitive: str
    source: str
    eqn_repr: str
    input_stats: list[str]
    output_stats: list[str]
    inputs_had_nan: bool
    call_stack: list[str] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        """Render with color auto-detected from the environment."""
        return self.render()

    def render(self, color: bool | None = None) -> str:
        """Render the site as a multi-line report block.

        Args:
            color: True forces ANSI colors, False forces plain text, None
                (default) auto-detects via :func:`_color_enabled`.

        Returns:
            A block shaped like::

                ✘ NaNs first produced by: div
                │ call stack (most recent call last):
                │       script.py:12 in main
                │     ▶ script.py:5 in model
                │ eqn: a:f32[2] = div b c
                │ inputs:
                │   [0] float32[2] range=[0, 0] nans=0 infs=0
                ╰ outputs:
                    [0] float32[2] range=[0, 0] nans=1 infs=0

            with the ``▶`` marking the frame where the NaN was born.
        """
        st = _Style(_color_enabled() if color is None else color)
        kind = (
            "NaNs propagated from inputs of"
            if self.inputs_had_nan
            else "NaNs first produced by"
        )
        lines = [st.title(f"✘ {kind}: ") + st.prim(self.primitive)]
        if self.call_stack:
            lines.append("│ " + st.dim("call stack (most recent call last):"))
            for i, frame in enumerate(self.call_stack):
                mark = "▶" if i == len(self.call_stack) - 1 else " "
                lines.append(f"│   {st.title(mark) if mark == '▶' else mark} "
                             + _style_frame(frame, st))
        else:
            lines.append(f"│ {st.title('▶')} " + _style_frame(self.source, st))
        lines.append("│ " + st.dim("eqn:") + f" {_one_line(self.eqn_repr)}")
        lines.append("│ " + st.dim("inputs:"))
        for i, s in enumerate(self.input_stats):
            lines.append(f"│   [{i}] " + _style_stats(s, st))
        lines.append("╰ " + st.dim("outputs:"))
        for i, s in enumerate(self.output_stats):
            lines.append(f"    [{i}] " + _style_stats(s, st))
        return "\n".join(lines)


@dataclasses.dataclass
class NanReport:
    """Result of a NaN scan; print it or inspect the fields programmatically.

    Exactly one of three situations is described:

    1. ``nan_in_inputs`` is True — the NaN came in with the caller's data;
       no equation is blamed and ``first_site`` is None.
    2. ``first_site`` is set — the scan located the equation responsible.
    3. Neither — the run was clean; ``outputs`` holds the function result.

    Attributes:
        nan_in_inputs: True when the function inputs already contained
            NaNs before any computation ran.
        first_site: The located :class:`NanSite`, or None for cases 1
            and 3.
        outputs: The function's outputs (with original pytree structure)
            when the scan completed, i.e. when no origin equation aborted
            it; None when the scan stopped at ``first_site``.
    """

    nan_in_inputs: bool
    first_site: NanSite | None
    outputs: Any

    @property
    def found(self) -> bool:
        """True if the scan detected NaNs anywhere (inputs or equations)."""
        return self.nan_in_inputs or self.first_site is not None

    def __str__(self) -> str:
        """Render with color auto-detected from the environment."""
        return self.render()

    def render(self, color: bool | None = None) -> str:
        """Render the report for the terminal.

        Args:
            color: True forces ANSI colors, False forces plain text, None
                (default) auto-detects via :func:`_color_enabled`.

        Returns:
            The rendered :class:`NanSite` block, or a one-line message for
            the NaN-in-inputs and all-clean cases.
        """
        st = _Style(_color_enabled() if color is None else color)
        if self.nan_in_inputs:
            return st.title("✘ NaNs are already present in the function inputs.")
        if self.first_site is None:
            return st.good("✔ No NaNs found in any intermediate or output value.")
        return self.first_site.render(color)

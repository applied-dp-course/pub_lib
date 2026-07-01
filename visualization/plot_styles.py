"""Shared line styles for colorblind-accessible multi-series plots."""

from __future__ import annotations

# Matplotlib
MPL_PRIMARY = "-"
MPL_SECONDARY = "--"
MPL_BOUND = "-."
MPL_REFERENCE = ":"
MPL_LINE_STYLE_CYCLE = (MPL_PRIMARY, MPL_SECONDARY, MPL_BOUND, MPL_REFERENCE)

# Plotly (semantic match to the matplotlib styles above)
PLOTLY_PRIMARY = "solid"
PLOTLY_SECONDARY = "dash"
PLOTLY_BOUND = "dashdot"
PLOTLY_REFERENCE = "dot"
PLOTLY_DASH_CYCLE = (PLOTLY_PRIMARY, PLOTLY_SECONDARY, PLOTLY_BOUND, PLOTLY_REFERENCE)


def mpl_line_style(index: int) -> str:
    return MPL_LINE_STYLE_CYCLE[index % len(MPL_LINE_STYLE_CYCLE)]


def plotly_dash(index: int) -> str:
    return PLOTLY_DASH_CYCLE[index % len(PLOTLY_DASH_CYCLE)]

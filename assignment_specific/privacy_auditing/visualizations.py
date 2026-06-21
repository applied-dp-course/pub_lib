"""Interactive binomial-bound figures used by the privacy-auditing lecture."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.special import xlogy

from libdpy.visualization.interactive import ControlSpec, InteractiveSpec
from libdpy.visualization.interactive_widgets import render_ipywidgets


def make_binomial_alpha_bounds_figure(
    p: float,
    log_beta: float,
    *,
    n_range: Sequence[int] = tuple(range(1, 1001)),
) -> go.Figure:
    """Return Markov, Chebyshev, and Hoeffding alpha bounds."""

    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    beta = float(np.exp(log_beta))
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    n = np.asarray(n_range, dtype=float)
    if n.ndim != 1 or len(n) == 0 or np.any(n <= 0):
        raise ValueError("n_range must contain positive sample sizes")

    bounds = {
        "Markov": np.minimum(1 - p, p / beta) * np.ones_like(n),
        "Chebyshev": np.minimum(1 - p, np.sqrt((p * (1 - p)) / (n * beta))),
        "Hoeffding": np.minimum(1 - p, np.sqrt(-np.log(beta) / (2 * n))),
    }
    figure = go.Figure(
        [go.Scatter(x=n, y=values, mode="lines", name=name) for name, values in bounds.items()]
    )
    figure.update_layout(
        title=f"Error Bounds (p={p:.2f}, β={beta:.2e})",
        xaxis_title="Sample Size (n)",
        yaxis_title="Error Bound (α)",
        width=800,
        height=500,
    )
    return figure


def make_binomial_beta_bounds_figure(
    p: float,
    alpha: float,
    *,
    n_range: Sequence[int] = tuple(range(100, 1001)),
    log_y: bool = False,
) -> go.Figure:
    """Return Chebyshev, Hoeffding, and Chernoff beta bounds."""

    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    n = np.asarray(n_range, dtype=float)
    if n.ndim != 1 or len(n) == 0 or np.any(n <= 0):
        raise ValueError("n_range must contain positive sample sizes")

    effective_alpha = min(alpha, 1 - p)
    chebyshev = np.minimum(
        1.0,
        (p * (1 - p)) / (n * max(effective_alpha, np.finfo(float).eps) ** 2),
    )
    hoeffding = np.exp(-2 * n * effective_alpha**2)

    chernoff_alpha = min(alpha, max(0.0, 1 - p - 1e-6))
    q = p + chernoff_alpha
    log_base = xlogy(q, np.divide(p, q, out=np.ones((), dtype=float), where=q != 0))
    complement = 1 - q
    log_base += xlogy(
        complement,
        np.divide(
            1 - p,
            complement,
            out=np.ones((), dtype=float),
            where=complement != 0,
        ),
    )
    chernoff = np.exp(n * log_base)

    figure = go.Figure(
        [
            go.Scatter(x=n, y=chebyshev, mode="lines", name="Chebyshev"),
            go.Scatter(x=n, y=hoeffding, mode="lines", name="Hoeffding"),
            go.Scatter(x=n, y=chernoff, mode="lines", name="Chernoff"),
        ]
    )
    figure.update_layout(
        title=f"Probability Bounds (p={p:.2f}, α={alpha:.2f})",
        xaxis_title="Sample Size (n)",
        yaxis_title="Error Probability (β)",
        yaxis_type="log" if log_y else "linear",
        width=800,
        height=500,
    )
    return figure


def binomial_alpha_bounds_spec() -> InteractiveSpec:
    return InteractiveSpec(
        name="binomial_alpha_bounds",
        artifact_name="binomial-alpha-bounds",
        controls=(
            ControlSpec("p", "slider", "p:", 0.5, min=0.0, max=1.0, step=0.01),
            ControlSpec(
                "log_beta",
                "slider",
                "log(β):",
                float(np.log(0.5)),
                min=float(np.log(1e-20)),
                max=float(np.log(0.5)),
                step=0.01,
            ),
        ),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets",),
        make_figure=make_binomial_alpha_bounds_figure,
        figure_factory=(
            "libdpy.assignment_specific.privacy_auditing.visualizations:"
            "make_binomial_alpha_bounds_figure"
        ),
    )


def binomial_beta_bounds_spec(*, log_y: bool = False) -> InteractiveSpec:
    return InteractiveSpec(
        name="binomial_beta_bounds_log" if log_y else "binomial_beta_bounds",
        artifact_name=("binomial-beta-bounds-log" if log_y else "binomial-beta-bounds"),
        controls=(
            ControlSpec("p", "slider", "p:", 0.5, min=0.0, max=1.0, step=0.01),
            ControlSpec(
                "alpha",
                "slider",
                "α:",
                0.1,
                min=0.001,
                max=1.0,
                step=0.01,
            ),
        ),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets",),
        make_figure=make_binomial_beta_bounds_figure,
        figure_factory=(
            "libdpy.assignment_specific.privacy_auditing.visualizations:"
            "make_binomial_beta_bounds_figure"
        ),
        fixed_kwargs={"log_y": log_y},
    )


def binomial_alpha_bounds_interactive():
    return render_ipywidgets(binomial_alpha_bounds_spec()).root


def binomial_beta_bounds_interactive(*, log_y: bool = False):
    return render_ipywidgets(binomial_beta_bounds_spec(log_y=log_y)).root

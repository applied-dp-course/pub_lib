"""Interactive binomial-bound figures used by the privacy-auditing lecture."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.special import xlogy

from libdpy.visualization.interactive import (
    AbstractInteractivePlot,
    ActionSpec,
    ControlSpec,
    InteractiveSpec,
)
from libdpy.visualization.interactive_widgets import render_ipywidgets
from libdpy.assignment_specific.privacy_auditing.utils import (
    run_repeated_gaussian_audits,
    true_epsilon_gaussian_threshold,
)


def make_binomial_alpha_bounds_figure(
    p: float,
    log_beta: float,
    *,
    n_range: Sequence[int] = tuple(range(1, 1001)),
) -> go.Figure:
    """Return the Hoeffding one-sided error bound as a function of ``n``."""

    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    beta = float(np.exp(log_beta))
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    n = np.asarray(n_range, dtype=float)
    if n.ndim != 1 or len(n) == 0 or np.any(n <= 0):
        raise ValueError("n_range must contain positive sample sizes")

    hoeffding = np.minimum(1 - p, np.sqrt(-np.log(beta) / (2 * n)))
    figure = go.Figure(
        [go.Scatter(x=n, y=hoeffding, mode="lines", name="Hoeffding")]
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
    """Return Hoeffding and Chernoff one-sided failure-probability bounds."""

    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    n = np.asarray(n_range, dtype=float)
    if n.ndim != 1 or len(n) == 0 or np.any(n <= 0):
        raise ValueError("n_range must contain positive sample sizes")

    effective_alpha = min(alpha, 1 - p)
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


def make_naive_safe_epsilon_histogram_figure(
    alpha_total: float,
    n_repeats: int,
    *,
    seed: int = 0,
    scale: float = 1.0,
    tau: float = 0.5,
    delta: float = 1e-2,
    n_audit: int = 500,
) -> go.Figure:
    """Compare plug-in and safe audit distributions against the analytic Gaussian truth."""

    if not 0 < alpha_total <= 1:
        raise ValueError("alpha_total must be in (0, 1]")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if n_audit <= 0:
        raise ValueError("n_audit must be positive")

    true_eps = true_epsilon_gaussian_threshold(tau, delta, scale=scale)
    plug_values, safe_values = run_repeated_gaussian_audits(
        tau,
        delta,
        scale=scale,
        n_audit=n_audit,
        n_repeats=int(n_repeats),
        alpha_total=float(alpha_total),
        seed=int(seed),
    )
    finite_plug = [value for value in plug_values if np.isfinite(value)]
    finite_safe = [value for value in safe_values if np.isfinite(value)]
    plug_failures = sum(value > true_eps for value in finite_plug)
    safe_failures = sum(value > true_eps for value in finite_safe)

    figure = go.Figure()
    if finite_plug:
        figure.add_trace(
            go.Histogram(
                x=finite_plug,
                name="plug-in ε",
                opacity=0.55,
                marker_color="#1f77b4",
                histnorm="",
            )
        )
    if finite_safe:
        figure.add_trace(
            go.Histogram(
                x=finite_safe,
                name="safe ε",
                opacity=0.55,
                marker_color="#ff7f0e",
                histnorm="",
            )
        )
    figure.add_vline(
        x=true_eps,
        line_color="#d62728",
        line_width=2,
        annotation_text=f"true ε = {true_eps:.3g}",
        annotation_position="top right",
    )
    figure.update_layout(
        title="Naive versus safe audit lower bounds",
        annotations=[
            dict(
                text=(
                    f"overshoots — plug-in: {plug_failures}/{len(finite_plug)}, "
                    f"safe: {safe_failures}/{len(finite_safe)} "
                    f"(target ≤ {alpha_total:.0%})"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.06,
                showarrow=False,
                font=dict(size=11, color="#555"),
                xanchor="center",
            )
        ],
        xaxis_title="ε",
        yaxis_title="frequency",
        barmode="overlay",
        width=820,
        height=480,
        margin=dict(t=70),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return figure


def naive_safe_epsilon_histogram_spec(
    *,
    scale: float = 1.0,
    tau: float = 0.5,
    delta: float = 1e-2,
    n_audit: int = 500,
    alpha_total: float = 0.05,
    n_repeats: int = 200,
    seed: int = 0,
) -> InteractiveSpec:
    def resample_action(state, _controls):
        state["seed"] = int(state.get("seed", seed)) + 1
        return None

    return InteractiveSpec(
        name="naive_safe_epsilon_histogram",
        artifact_name="naive-safe-epsilon-histogram",
        controls=(
            ControlSpec(
                name="alpha_total",
                kind="slider",
                label="failure prob. α",
                default=alpha_total,
                min=0.01,
                max=0.2,
                step=0.01,
                continuous=True,
            ),
            ControlSpec(
                name="n_repeats",
                kind="slider",
                label="repetitions",
                default=n_repeats,
                min=50,
                max=500,
                step=10,
                continuous=False,
            ),
        ),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets", "wasm-marimo"),
        fixed_kwargs={
            "scale": scale,
            "tau": tau,
            "delta": delta,
            "n_audit": n_audit,
        },
        make_figure=make_naive_safe_epsilon_histogram_figure,
        figure_factory=(
            "libdpy.assignment_specific.privacy_auditing.visualizations:"
            "make_naive_safe_epsilon_histogram_figure"
        ),
        actions=(
            ActionSpec(
                name="resample",
                label="Resample",
                handler=resample_action,
                button_style="info",
                state_updates={"seed": 1},
            ),
        ),
        initial_state={"seed": seed},
        description=(
            "Repeated Gaussian audits at a fixed threshold, comparing plug-in and safe estimates "
            "against the analytic true epsilon."
        ),
    )


def naive_safe_epsilon_histogram_interactive(
    *,
    scale: float = 1.0,
    tau: float = 0.5,
    delta: float = 1e-2,
    n_audit: int = 500,
    alpha_total: float = 0.05,
    n_repeats: int = 200,
    seed: int = 0,
):
    return render_ipywidgets(
        naive_safe_epsilon_histogram_spec(
            scale=scale,
            tau=tau,
            delta=delta,
            n_audit=n_audit,
            alpha_total=alpha_total,
            n_repeats=n_repeats,
            seed=seed,
        )
    ).root


class NaiveSafeEpsilonHistogram(AbstractInteractivePlot):
    """Embeddable wrapper for the naive-vs-safe epsilon histogram explorer.

    Use ``.show()`` in a live kernel (Colab/Jupyter) and ``.embed()`` for the static
    site build, mirroring the other interactive plot wrappers.
    """

    def __init__(
        self,
        *,
        scale: float = 1.0,
        tau: float = 0.5,
        delta: float = 1e-2,
        n_audit: int = 500,
        alpha_total: float = 0.05,
        n_repeats: int = 200,
        seed: int = 0,
    ):
        self._kwargs = dict(
            scale=scale,
            tau=tau,
            delta=delta,
            n_audit=n_audit,
            alpha_total=alpha_total,
            n_repeats=n_repeats,
            seed=seed,
        )

    def spec(self) -> InteractiveSpec:
        return naive_safe_epsilon_histogram_spec(**self._kwargs)

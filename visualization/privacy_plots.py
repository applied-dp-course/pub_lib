"""Privacy-specific visualization utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from statistics import NormalDist

import plotly.graph_objects as go
from .interactive import AbstractInteractivePlot, ControlSpec, InteractiveSpec
from .plot_styles import PLOTLY_BOUND, plotly_dash

_EPSILON_MAX = math.log(30)
_DEFAULT_EPSILON = 0.0
_DEFAULT_DELTA = 0.0


def _epsilon_control_spec(epsilon: float = _DEFAULT_EPSILON) -> ControlSpec:
    return ControlSpec(
        name="epsilon",
        kind="slider",
        label="ε",
        default=epsilon,
        min=0.0,
        max=_EPSILON_MAX,
        step=0.01,
        continuous=True,
    )


def _privacy_delta_control_spec(delta: float = _DEFAULT_DELTA) -> ControlSpec:
    return ControlSpec(
        name="delta",
        kind="slider",
        label="δ",
        default=delta,
        min=0.0,
        max=1.0,
        step=0.01,
        continuous=True,
    )


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [start + index * step for index in range(count)]


def calc_privacy_bound(
    log_slope: float, shift: float, res: int = 100
) -> tuple[list[float], list[float]]:
    """Return the piecewise-linear privacy bound for concrete parameters."""

    if res < 2:
        raise ValueError("res must be at least 2")
    if not 0 <= shift <= 1:
        raise ValueError("shift must be between 0 and 1")
    slope = math.exp(log_slope)
    midpoint = (1 - shift) / (1 + slope)
    x_1 = _linspace(0, midpoint, res // 2)
    y_1 = [slope * value + shift for value in x_1]
    x_2 = _linspace(midpoint, 1 - shift, res - res // 2)
    y_2 = [value / slope + 1 - (1 - shift) / slope for value in x_2]
    return x_1 + x_2, y_1 + y_2


def create_distribution(dist_type, mean, std):
    return dist_type(loc=mean, scale=std)


def draw_ROCs_and_diag_from_distributions(
    distributions, fig, show_diagonal=True, diagonal_offset=0, res=100
):
    import numpy as np

    colors = ["red", "green", "orange", "purple", "brown"]

    for i, (dist0, dist1) in enumerate(distributions):
        # Calculate ROC curve points
        thresholds = np.linspace(dist0.ppf(0.001), dist0.ppf(0.999), res)
        fpr = 1 - dist0.cdf(thresholds)
        tpr = 1 - dist1.cdf(thresholds)

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC {i+1}",
                line=dict(color=color, width=2, dash=plotly_dash(i)),
            )
        )

    if show_diagonal:
        diagonal_x = np.linspace(0, 1, res)
        diagonal_y = diagonal_x + diagonal_offset
        diagonal_y = np.clip(diagonal_y, 0, 1)
        fig.add_trace(
            go.Scatter(
                x=diagonal_x,
                y=diagonal_y,
                mode="lines",
                name="Diagonal",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )


def _distribution_name(distribution_type) -> str:
    if isinstance(distribution_type, str):
        name = distribution_type
    else:
        name = getattr(distribution_type, "name", None)
    if name is None:
        name = getattr(distribution_type, "__name__", "")
        if name.endswith("_gen"):
            name = name[:-4]
    if name not in {"norm", "laplace", "uniform"}:
        raise ValueError(f"unsupported scipy distribution type: {distribution_type!r}")
    return name


def _distribution_ppf(name: str, probability: float, mean: float, std: float) -> float:
    if name == "norm":
        return NormalDist(mean, std).inv_cdf(probability)
    if name == "laplace":
        if probability < 0.5:
            return mean + std * math.log(2 * probability)
        return mean - std * math.log(2 * (1 - probability))
    if name == "uniform":
        return mean + probability * std
    raise ValueError(f"unknown distribution: {name!r}")


def _distribution_cdf(name: str, value: float, mean: float, std: float) -> float:
    if name == "norm":
        return NormalDist(mean, std).cdf(value)
    if name == "laplace":
        if value < mean:
            return 0.5 * math.exp((value - mean) / std)
        return 1 - 0.5 * math.exp(-(value - mean) / std)
    if name == "uniform":
        return min(1.0, max(0.0, (value - mean) / std))
    raise ValueError(f"unknown distribution: {name!r}")


def _named_distribution_roc(
    name: str, sensitivity: float, std: float, res: int
) -> tuple[list[float], list[float]]:
    lower = _distribution_ppf(name, 0.001, 0, std)
    upper = _distribution_ppf(name, 0.999, 0, std)
    thresholds = _linspace(lower, upper, res)
    fpr = [1 - _distribution_cdf(name, threshold, 0, std) for threshold in thresholds]
    tpr = [1 - _distribution_cdf(name, threshold, sensitivity, std) for threshold in thresholds]
    return fpr, tpr


def make_privacy_bound_figure(
    log_slope: float,
    shift: float,
    *,
    distribution_names: Sequence[str] = ("norm",),
    sensitivity: float = 1.0,
    std: float = 1.5,
    res: int = 100,
) -> go.Figure:
    """Build one Plotly figure for one concrete privacy-plot state."""

    if sensitivity == 0:
        raise ValueError("sensitivity must be nonzero")
    if std <= 0:
        raise ValueError("std must be positive")

    figure = go.Figure()
    x, y = calc_privacy_bound(log_slope, shift, res)
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Privacy bound",
            line=dict(color="blue", width=4, dash=PLOTLY_BOUND),
        )
    )

    colors = ["red", "green", "orange", "purple", "brown"]
    for index, name in enumerate(distribution_names):
        fpr, tpr = _named_distribution_roc(name, sensitivity, std, res)
        figure.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC {index + 1}",
                line=dict(
                    color=colors[index % len(colors)],
                    width=2,
                    dash=plotly_dash(index),
                ),
            )
        )

    figure.update_layout(
        title={
            "text": f"ROC for distributions with std / sensitivity = {std/sensitivity:.2f}",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        autosize=True,
        height=600,
        margin={"t": 80, "b": 60, "l": 65, "r": 30},
    )
    return figure


def _privacy_plot_artifact_name(
    distribution_names: Sequence[str], sensitivity: float, std: float, res: int
) -> str:
    configuration = {
        "distribution_names": list(distribution_names),
        "sensitivity": sensitivity,
        "std": std,
        "res": res,
    }
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    distribution_slug = "-".join(distribution_names)
    return f"privacy-plot-{distribution_slug}-{digest}"


def _make_privacy_plot_interactive_figure(
    epsilon: float,
    delta: float,
    *,
    distribution_names: Sequence[str] = ("norm",),
    sensitivity: float = 1.0,
    std: float = 1.5,
    res: int = 100,
) -> go.Figure:
    """Interactive adapter: ``epsilon`` and ``delta`` map to ``log_slope`` and ``shift``."""

    return make_privacy_bound_figure(
        epsilon,
        delta,
        distribution_names=distribution_names,
        sensitivity=sensitivity,
        std=std,
        res=res,
    )


def privacy_plot_spec(
    distribution_types: Sequence = ("norm",),
    sensitivity: float = 1.0,
    std: float = 1.5,
    res: int = 100,
    epsilon: float = _DEFAULT_EPSILON,
    delta: float = _DEFAULT_DELTA,
) -> InteractiveSpec:
    """Return the backend-neutral specification for ``PrivacyPlot``."""

    if not 0.0 <= epsilon <= _EPSILON_MAX:
        raise ValueError(f"epsilon must be between 0 and {_EPSILON_MAX}")
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be between 0 and 1")

    distribution_names = tuple(_distribution_name(item) for item in distribution_types)
    fixed_kwargs = {
        "distribution_names": distribution_names,
        "sensitivity": sensitivity,
        "std": std,
        "res": res,
    }
    return InteractiveSpec(
        name="privacy_plot",
        artifact_name=_privacy_plot_artifact_name(
            distribution_names=distribution_names,
            sensitivity=sensitivity,
            std=std,
            res=res,
        ),
        controls=(
            _epsilon_control_spec(epsilon),
            _privacy_delta_control_spec(delta),
        ),
        preferred_backend="wasm-marimo",
        # Only advertise backends that have a working renderer. JupyterLite
        # (interactive.py's third Backend literal) is intentionally not listed:
        # no render_jupyterlite adapter exists yet, so claiming it here would
        # let the build pick a backend it cannot actually produce.
        allowed_backends=("ipywidgets", "wasm-marimo"),
        make_figure=_make_privacy_plot_interactive_figure,
        figure_factory=(
            "libdpy.visualization.privacy_plots:_make_privacy_plot_interactive_figure"
        ),
        fixed_kwargs=fixed_kwargs,
        description="Privacy bound explorer with ε and δ controls.",
    )


class PrivacyPlot(AbstractInteractivePlot):
    """Privacy-bound explorer with notebook and static-site adapters."""

    def __init__(
        self,
        distribution_types: Sequence,
        sensitivity: float,
        std: float,
        res: int = 100,
        epsilon: float = _DEFAULT_EPSILON,
        delta: float = _DEFAULT_DELTA,
    ):
        if not 0.0 <= epsilon <= _EPSILON_MAX:
            raise ValueError(f"epsilon must be between 0 and {_EPSILON_MAX}")
        if not 0.0 <= delta <= 1.0:
            raise ValueError("delta must be between 0 and 1")

        self.distribution_types = tuple(distribution_types)
        self.sensitivity = sensitivity
        self.std = std
        self.res = res
        self.epsilon = float(epsilon)
        self.delta = float(delta)

    def spec(self) -> InteractiveSpec:
        return privacy_plot_spec(
            distribution_types=self.distribution_types,
            sensitivity=self.sensitivity,
            std=self.std,
            res=self.res,
            epsilon=self.epsilon,
            delta=self.delta,
        )

    def show(self, **renderer_options):
        """Display the live notebook widget."""

        return super().show(**renderer_options)


def make_epsilon_delta_tradeoff_figure(
    epsilons, deltas, mechanism_names=None, title="Privacy Budget Trade-off"
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    if mechanism_names is None:
        mechanism_names = [f"Mechanism {i+1}" for i in range(len(epsilons))]

    for i, (eps_list, delta_list, name) in enumerate(zip(epsilons, deltas, mechanism_names)):
        ax.plot(eps_list, delta_list, "o-", label=name, linewidth=2, markersize=6)

    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Failure Probability (δ)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xscale("log")
    return fig


def make_privacy_loss_distribution_figure(privacy_losses, epsilon, title="Privacy Loss Distribution"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        privacy_losses,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
    )
    ax.axvline(epsilon, color="red", linestyle="--", linewidth=2, label=f"ε = {epsilon}")
    ax.axvline(-epsilon, color="red", linestyle="--", linewidth=2, label=f"-ε = {-epsilon}")
    ax.set_xlabel("Privacy Loss")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def make_roc_curves_figure(fpr_list, tpr_list, labels=None, title="ROC Curves Comparison"):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))

    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(fpr_list))]

    for fpr, tpr, label in zip(fpr_list, tpr_list, labels):
        auc_value = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc_value:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig


def make_privacy_accounting_figure(epsilons, steps, title="Privacy Budget Consumption"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, epsilons, "b-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative Privacy Budget (ε)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    privacy_levels = [0.1, 1.0, 10.0]
    colors = ["green", "orange", "red"]
    labels = ["Strong Privacy", "Moderate Privacy", "Weak Privacy"]

    for level, color, label in zip(privacy_levels, colors, labels):
        if level <= max(epsilons):
            ax.axhline(
                level,
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"{label} (ε={level})",
            )

    ax.legend()
    return fig

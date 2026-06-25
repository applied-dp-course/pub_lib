import hashlib
import json
import math
import re
from enum import Enum
from functools import lru_cache, partial
from typing import Any, Callable, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import cauchy, laplace, logistic, norm, t
from scipy.stats import rv_continuous as Distribution
from scipy.stats import uniform
from sklearn.metrics import auc, roc_curve

from .interactive import (
    AbstractInteractivePlot,
    ActionSpec,
    ControlSpec,
    InteractiveSpec,
    declarative_plotly_from_spec,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - exercised in browser-side Pyodide exports.

    def display(*_args, **_kwargs):
        return None


class ComparisonType(Enum):
    SAME_VAR = 'value threshold'
    SAME_MEAN = 'absolute value threshold'
    GENERAL = 'general'


def create_ROC_point_and_thresholds(
    desicion_rule_params: Callable,
    dist0: Distribution,
    dist1: Distribution,
    param: float,
    param_type: ComparisonType,
    res: int,
):
    line_height = np.max([dist0.pdf(dist0.mean()), dist1.pdf(dist1.mean())]) * 1.05
    dist_range = range_from_distribution(dist0, dist1, res)
    fpr, tpr, thresholds = desicion_rule_params(dist0, dist1, param, param_type)
    markers = [
        dict(
            type="circle",
            xref="x2",
            yref="y2",
            x0=fpr - 0.01,
            x1=fpr + 0.01,
            y0=tpr - 0.01,
            y1=tpr + 0.01,
            fillcolor="yellow",
            line_color="yellow",
        )
    ]
    for threshold in thresholds:
        markers.append(
            dict(
                type="line",
                x0=threshold,
                x1=threshold,
                y0=0,
                y1=line_height,
                line=dict(color="Black", width=2),
            )
        )
    if len(thresholds) == 1:
        # add a rectangle to color the area to the left of the threshold in light blue and the area to the right in light red
        markers.append(
            dict(
                type="rect",
                x0=thresholds[0],
                x1=np.max(dist_range),
                y0=0,
                y1=line_height,
                fillcolor="rgba(0, 0, 255, 0.2)",
            )
        )
        markers.append(
            dict(
                type="rect",
                x0=np.min(dist_range),
                x1=thresholds[0],
                y0=0,
                y1=line_height,
                fillcolor="rgba(255, 0, 0, 0.2)",
            )
        )
    else:
        # add rectangles to color the area between the thresholds in light blue and the area to the left of the left threshold and to the right of the right threshold in light red
        markers.append(
            dict(
                type="rect",
                x0=np.min(dist_range),
                x1=thresholds[0],
                y0=0,
                y1=line_height,
                fillcolor="rgba(0, 0, 255, 0.2)",
            )
        )
        markers.append(
            dict(
                type="rect",
                x0=thresholds[1],
                x1=np.max(dist_range),
                y0=0,
                y1=line_height,
                fillcolor="rgba(0, 0, 255, 0.2)",
            )
        )
        markers.append(
            dict(
                type="rect",
                x0=thresholds[0],
                x1=thresholds[1],
                y0=0,
                y1=line_height,
                fillcolor="rgba(255, 0, 0, 0.2)",
                line_color="rgba(0, 0, 255, 0.2)",
            )
        )
    return markers


def draw_two_distributions(dist0: Distribution, dist1: Distribution, fig, res: int):
    x_range = range_from_distribution(dist0, dist1, res)
    hist0 = go.Scatter(
        x=x_range,
        y=dist0.pdf(x_range),
        name=f"{dist0.dist.name}({dist0.args[0] :.1f}, {dist0.args[1]:.1f})",
        mode='lines',
        line=dict(width=2),
        marker_color='red',
    )
    hist1 = go.Scatter(
        x=x_range,
        y=dist1.pdf(x_range),
        name=f"{dist1.dist.name}({dist1.args[0] :.1f}, {dist1.args[1]:.1f})",
        mode='lines',
        line=dict(width=2),
        marker_color='blue',
    )
    fig.add_trace(hist0, row=1, col=1)
    fig.add_trace(hist1, row=1, col=1)


def draw_ROCs_and_diag_from_desicion_rule(
    desicion_rule_params: Callable,
    distributions: list[list[Distribution]],
    fig,
    by_threshold: bool,
    column: int,
    res: int,
):
    for dist_tuple in distributions:
        dist0 = dist_tuple[0]
        dist1 = dist_tuple[1]
        if by_threshold:
            FPR, TPR = calc_numerical_ROC(dist0, dist1, res)
        else:
            comparison_type = check_comparison_type(dist0, dist1)
            FPR = np.linspace(0.0001, 0.9999, res)
            TPR = [desicion_rule_params(dist0, dist1, fpr, comparison_type)[1] for fpr in FPR]  # type: ignore
        dist_name = ' - ' + dist0.dist.name if len(distributions) > 1 else ""
        if column == 0:
            fig.add_trace(go.Scatter(x=FPR, y=TPR, mode='lines', name='ROC Curve' + dist_name))
        else:
            fig.add_trace(
                go.Scatter(x=FPR, y=TPR, mode='lines', name='ROC Curve' + dist_name),
                row=1,
                col=column,
            )
    if column == 0:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='X=Y (Random Performance)',
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='X=Y (Random Performance)',
            ),
            row=1,
            col=column,
        )


def make_roc_decision_figure(
    decision_parameter: float,
    *,
    desicion_rule_params: Callable,
    dist0: Distribution,
    dist1: Distribution,
    by_threshold: bool,
    res: int = 100,
):
    """Build one distribution/ROC state for a concrete decision parameter."""

    if res < 2:
        raise ValueError("res must be at least 2")
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distributions PDF", "ROC Curve"),
        column_widths=[0.45, 0.45],
        horizontal_spacing=0.1,
    )
    draw_two_distributions(dist0, dist1, figure, res)
    draw_ROCs_and_diag_from_desicion_rule(
        desicion_rule_params, [[dist0, dist1]], figure, by_threshold, 2, res
    )
    parameter_type = ComparisonType.GENERAL if by_threshold else check_comparison_type(dist0, dist1)
    figure.update_layout(
        shapes=create_ROC_point_and_thresholds(
            desicion_rule_params,
            dist0,
            dist1,
            decision_parameter,
            parameter_type,
            res,
        ),
        height=500,
        autosize=True,
        margin=dict(t=90, b=40, l=60, r=30),
        title=dict(
            text="Two distributions and their ROC curve",
            y=0.97,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='black', family='Arial', weight='bold'),
        ),
        xaxis=dict(title=dict(text="Value"), domain=[0.0, 0.45]),
        yaxis=dict(title=dict(text="Probability Density")),
        xaxis2=dict(
            title=dict(text="False Positive Rate (FPR)"),
            domain=[0.55, 1.0],
        ),
        yaxis2=dict(title=dict(text="True Positive Rate (TPR)"), domain=[0, 1]),
    )
    return figure


def roc_decision_spec(
    desicion_rule_params: Callable,
    dist0: Distribution,
    dist1: Distribution,
    by_threshold: bool,
    res: int = 100,
) -> tuple[InteractiveSpec, np.ndarray]:
    """Return the one-control ROC specification and its finite display grid."""

    slider_states = min(res, 101)
    if by_threshold:
        parameter_values = range_from_distribution(dist0, dist1, slider_states)
        default = (dist0.mean() + dist1.mean()) / 2
        label = "Threshold"
    else:
        parameter_values = np.linspace(0.0001, 0.9999, slider_states)
        default = 0.5
        label = "FPR"
    spec = InteractiveSpec(
        name="roc_decision_explorer",
        artifact_name="roc-decision-explorer",
        controls=(
            ControlSpec(
                name="decision_parameter",
                kind="slider",
                label=label,
                default=float(default),
                min=float(parameter_values[0]),
                max=float(parameter_values[-1]),
                step=float(parameter_values[1] - parameter_values[0]),
            ),
        ),
        preferred_backend="plotly-declarative",
        allowed_backends=("ipywidgets", "plotly-declarative"),
        make_figure=make_roc_decision_figure,
        figure_factory="libdpy.visualization.roc_plots:make_roc_decision_figure",
        fixed_kwargs={
            "desicion_rule_params": desicion_rule_params,
            "dist0": dist0,
            "dist1": dist1,
            "by_threshold": by_threshold,
            "res": res,
        },
    )
    return spec, parameter_values


def ROC_and_distributions_visualization(
    desicion_rule_params: Callable,
    dist0: Distribution,
    dist1: Distribution,
    by_threshold: bool,
    res: int = 100,
):
    spec, parameter_values = roc_decision_spec(
        desicion_rule_params,
        dist0,
        dist1,
        by_threshold,
        res,
    )
    return declarative_plotly_from_spec(
        spec,
        {"decision_parameter": parameter_values},
        max_states=200,
        max_json_mb=5.0,
        assume_constant_data=True,
    )


def check_comparison_type(dist0: Distribution, dist1: Distribution) -> ComparisonType:
    if (
        dist0.dist.name == dist1.dist.name
        and dist0.std() == dist1.std()
        and dist1.mean() > dist0.mean()
    ):
        return ComparisonType.SAME_VAR
    elif (
        dist0.dist.name == dist1.dist.name
        and dist0.mean() == dist1.mean()
        and dist1.std() > dist0.std()
    ):
        return ComparisonType.SAME_MEAN
    else:
        raise ValueError("This setting is not supported")


def range_from_distribution(dist0: Distribution, dist1: Distribution, res: int) -> np.ndarray:
    min_range = min(dist0.mean() - 4 * dist0.std(), dist1.mean() - 4 * dist1.std())
    max_range = max(dist0.mean() + 4 * dist0.std(), dist1.mean() + 4 * dist1.std())
    return np.linspace(min_range, max_range, res)


def calc_numerical_ROC(
    dist0: Distribution, dist1: Distribution, res: int
) -> tuple[np.ndarray, np.ndarray]:
    x_range = range_from_distribution(dist0, dist1, res)
    y0 = dist0.pdf(x_range)
    y0 = y0 / np.sum(y0)
    y1 = dist1.pdf(x_range)
    y1 = y1 / np.sum(y1)
    prob_ratio = y1 / y0
    sorted_indices = np.argsort(prob_ratio)
    y0 = y0[sorted_indices] / np.sum(y0)
    y1 = y1[sorted_indices] / np.sum(y1)
    FPR = 1 - np.cumsum(y0)
    TPR = 1 - np.cumsum(y1)
    return FPR, TPR


def create_distribution(dist_type: Distribution, mu: float, std: float) -> Distribution:
    match dist_type.__class__.__name__:
        case norm.__class__.__name__:
            loc = mu
            scale = std
        case laplace.__class__.__name__:
            loc = mu
            scale = std / math.sqrt(2)
        case uniform.__class__.__name__:
            loc = mu - math.sqrt(3) * std
            scale = 2 * math.sqrt(3) * std
        case _:
            raise ValueError("Unsupported distribution type")
    return dist_type(loc, scale)


def calc_privacy_bound(log_slope: float, shift: float, res) -> tuple[np.ndarray, np.ndarray]:
    slope = np.exp(log_slope)
    x_1 = np.linspace(0, (1 - shift) / (1 + slope), np.floor(res / 2).astype(int))
    y_1 = slope * x_1 + shift
    x_2 = np.linspace((1 - shift) / (1 + slope), 1 - shift, np.ceil(res / 2).astype(int))
    y_2 = 1 / slope * x_2 + 1 - (1 - shift) / slope
    x = np.concatenate([x_1, x_2])
    y = np.concatenate([y_1, y_2])
    return x, y


def one_sided_privacy_bound(
    log_slope: float,
    shift: float,
    res: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-sided ROC bound ``TPR = exp(log_slope) * FPR + shift``."""

    if res < 2:
        raise ValueError("res must be at least 2")
    if not 0 <= shift <= 1:
        raise ValueError("shift must be between 0 and 1")
    if not np.isfinite(log_slope):
        return np.array([0.0]), np.array([1.0])

    slope = np.exp(max(0.0, log_slope))
    if slope <= 0:
        x_end = 1.0
    else:
        x_end = min(1.0, (1.0 - shift) / slope)
    x = np.linspace(0.0, x_end, res)
    y = slope * x + shift
    return x, y


# Distributions with a closed-form density: they can draw their theoretical PDFs
# and exact ROC curve.
_ANALYTIC_ROC_DISTRIBUTIONS = (
    "Laplace",
    "Gaussian",
    "Student t",
    "Cauchy",
    "Logistic",
)
# The difference of two i.i.d. lognormals has no elementary PDF/CDF, so it is
# sampling-only (see make_empirical_roc_figure, which skips its theory traces).
_LOGNORMAL_DIFFERENCE = "Lognormal difference"
_TWO_LOGISTIC_SUM = "Sum of logistic distributions"
_TWO_LOGISTIC_SCALE2 = 0.4
_EMPIRICAL_ROC_DISTRIBUTIONS = _ANALYTIC_ROC_DISTRIBUTIONS + (_LOGNORMAL_DIFFERENCE, _TWO_LOGISTIC_SUM)
_EMPIRICAL_ROC_STUDENT_T_DF = 3
_EMPIRICAL_ROC_CAUCHY_MARGIN = 6.0


def _empirical_distribution(name: str):
    """Resolve a display name and its scipy class.

    Returns ``(display_name, distribution_type)`` where ``distribution_type`` is
    the scipy class for analytic distributions, or ``None`` for sampling-only
    distributions (the lognormal difference) that have no closed-form density.
    """
    normalized = name.strip().lower()
    if normalized in {"gaussian", "normal", "norm"}:
        return "Gaussian", norm
    if normalized == "laplace":
        return "Laplace", laplace
    if normalized in {"student t", "student-t", "studentt", "t"}:
        return "Student t", t
    if normalized == "cauchy":
        return "Cauchy", cauchy
    if normalized == "logistic":
        return "Logistic", logistic
    if normalized in {
        "two logistic sum",
        "two-logistic-sum",
        "two_logistic_sum",
        "sum of logistic distributions",
        "sum of logistics",
        "sum-of-logistic-distributions",
    }:
        return _TWO_LOGISTIC_SUM, None
    if normalized in {"lognormal difference", "lognormal-difference", "lognormal_difference"}:
        return _LOGNORMAL_DIFFERENCE, None
    supported = ", ".join(f"'{name}'" for name in _EMPIRICAL_ROC_DISTRIBUTIONS)
    raise ValueError(f"distribution must be one of {supported}")


def _make_empirical_roc_dist(
    distribution_type: type[Distribution],
    loc: float,
    scale: float,
) -> Distribution:
    if distribution_type is t:
        return t(df=_EMPIRICAL_ROC_STUDENT_T_DF, loc=loc, scale=scale)
    return distribution_type(loc=loc, scale=scale)


def _sample_empirical_roc_dist(
    distribution_type: type[Distribution],
    loc: float,
    scale: float,
    size: int,
    random_state: np.random.Generator,
) -> np.ndarray:
    if distribution_type is t:
        return t.rvs(
            df=_EMPIRICAL_ROC_STUDENT_T_DF,
            loc=loc,
            scale=scale,
            size=size,
            random_state=random_state,
        )
    return distribution_type.rvs(
        loc=loc,
        scale=scale,
        size=size,
        random_state=random_state,
    )


_EMPIRICAL_ROC_TAIL_PROB = 1e-6
_EMPIRICAL_ROC_SCALE_MIN = 0.1
_EMPIRICAL_ROC_SCALE_MAX = 5.0


def _frozen_loc_scale(dist: Distribution) -> tuple[float, float]:
    if dist.kwds:
        return float(dist.kwds.get("loc", 0.0)), float(dist.kwds.get("scale", 1.0))
    if len(dist.args) >= 2:
        return float(dist.args[0]), float(dist.args[1])
    raise ValueError("distribution must provide loc and scale")


def _empirical_roc_pdf_x_range(
    dist_negative: Distribution,
    dist_positive: Distribution,
    *,
    tail_prob: float = _EMPIRICAL_ROC_TAIL_PROB,
) -> tuple[float, float]:
    """Return the PDF panel x-axis bounds from distribution tail quantiles."""

    if not 0 < tail_prob < 0.5:
        raise ValueError("tail_prob must be between 0 and 0.5")
    if not np.isfinite(dist_negative.std()) or not np.isfinite(dist_positive.std()):
        neg_loc, neg_scale = _frozen_loc_scale(dist_negative)
        pos_loc, pos_scale = _frozen_loc_scale(dist_positive)
        margin = _EMPIRICAL_ROC_CAUCHY_MARGIN
        return float(neg_loc - margin * neg_scale), float(pos_loc + margin * pos_scale)
    return float(dist_negative.ppf(tail_prob)), float(dist_positive.ppf(1 - tail_prob))


def _samples_pdf_x_range(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    tail_prob: float = 0.02,
) -> tuple[float, float]:
    """PDF panel x-axis bounds for sampling-only distributions, from sample quantiles.

    Heavy-tailed sampling-only distributions have no quantile function to query, so
    the readable range is taken from inner quantiles of the drawn samples.
    """
    if not 0 < tail_prob < 0.5:
        raise ValueError("tail_prob must be between 0 and 0.5")
    negatives = samples[labels == 0]
    positives = samples[labels == 1]
    lower = float(np.quantile(negatives, tail_prob))
    upper = float(np.quantile(positives, 1 - tail_prob))
    if not lower < upper:
        lower, upper = float(np.min(samples)), float(np.max(samples))
    return lower, upper


def _sample_lognormal_difference(
    loc: float,
    scale: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``loc + (A - B)`` with ``A, B`` i.i.d. ``LogNormal(0, scale)``.

    The difference of two i.i.d. lognormals is symmetric about ``loc`` but has no
    elementary PDF/CDF -- which is exactly why this distribution is sampling-only.
    """
    first = rng.lognormal(mean=0.0, sigma=scale, size=size)
    second = rng.lognormal(mean=0.0, sigma=scale, size=size)
    return loc + first - second


def _sample_two_logistic_sum(
    loc: float,
    scale1: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``loc + Z_1 + Z_2`` with independent logistics (sampling-only distribution)."""

    from libdpy.privacy_mechanisms.noise import two_logistic_noise

    return loc + two_logistic_noise(
        size=size,
        scale1=scale1,
        scale2=_TWO_LOGISTIC_SCALE2,
        rng=rng,
    )


def _generate_empirical_roc_samples(
    distribution: str,
    scale: float,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    display_name, distribution_type = _empirical_distribution(distribution)
    rng = np.random.default_rng(seed)
    if distribution_type is None:
        if display_name == _TWO_LOGISTIC_SUM:
            negative_samples = _sample_two_logistic_sum(0, scale, n_samples, rng)
            positive_samples = _sample_two_logistic_sum(1, scale, n_samples, rng)
        elif display_name == _LOGNORMAL_DIFFERENCE:
            negative_samples = _sample_lognormal_difference(0, scale, n_samples, rng)
            positive_samples = _sample_lognormal_difference(1, scale, n_samples, rng)
        else:
            raise ValueError(f"unsupported sampling-only distribution: {display_name}")
    else:
        negative_samples = _sample_empirical_roc_dist(
            distribution_type,
            loc=0,
            scale=scale,
            size=n_samples,
            random_state=rng,
        )
        positive_samples = _sample_empirical_roc_dist(
            distribution_type,
            loc=1,
            scale=scale,
            size=n_samples,
            random_state=rng,
        )
    labels = np.concatenate([np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=int)])
    return np.concatenate([negative_samples, positive_samples]), labels


# --- Modular ROC figure family -------------------------------------------------
#
# Rather than one figure builder switching on flags, the explorer is assembled
# from small panel builders that each add one layer to a shared two-panel canvas
# (PDF panel on the left, ROC panel on the right). The public factories below
# compose the layers they need, so the family builds up cleanly:
#
#   make_theory_roc_figure      -> theory PDFs + theoretical ROC          (analytic only)
#   make_empirical_roc_figure   -> the above (when analytic) + sampled
#                                  histograms + empirical ROC             (any distribution)
#
# Trace order is part of the contract (tests and the legend grouping depend on
# it): negative pdf, positive pdf, theoretical ROC, negative hist, positive hist,
# empirical ROC, random-classifier diagonal.

_EPSILON_FIGURE_ROC_COLOR = "#00897B"
_EPSILON_FIGURE_BOUND_COLOR = "#5E35B1"
_DELTA_MIN = 1e-15
_DELTA_DEFAULT_THEORY = 1e-6
_DELTA_DEFAULT_EMPIRICAL = 1e-2
_ROC_FIGURE_WIDTH = 1000
_ROC_FIGURE_MARGIN_L = 65
_ROC_FIGURE_MARGIN_R = 35
_ROC_SUBPLOT_DOMAINS = ((0.0, 0.46), (0.54, 1.0))


def roc_panel_slider_grid_template(
    *,
    figure_width: int = _ROC_FIGURE_WIDTH,
    margin_left: int = _ROC_FIGURE_MARGIN_L,
    margin_right: int = _ROC_FIGURE_MARGIN_R,
    subplot_domains: tuple[tuple[float, float], tuple[float, float]] = _ROC_SUBPLOT_DOMAINS,
) -> str:
    """Return a CSS grid template aligned with the ROC figure's two subplot panels."""

    plot_width = figure_width - margin_left - margin_right
    left_width = int(round((subplot_domains[0][1] - subplot_domains[0][0]) * plot_width))
    gap_width = int(round((subplot_domains[1][0] - subplot_domains[0][1]) * plot_width))
    right_width = int(round((subplot_domains[1][1] - subplot_domains[1][0]) * plot_width))
    return f"{margin_left}px {left_width}px {gap_width}px {right_width}px {margin_right}px"


def _new_roc_canvas() -> go.Figure:
    """Return the shared empty two-panel (PDF | ROC) subplot canvas."""

    return make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Probability Density Functions", "ROC Curves"),
        horizontal_spacing=0.08,
    )


def _add_theory_pdf_layer(
    figure: go.Figure,
    dist_negative: Distribution,
    dist_positive: Distribution,
    x: np.ndarray,
) -> None:
    """Add the analytic negative/positive PDF curves to the PDF panel."""

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dist_negative.pdf(x),
            mode="lines",
            name="Negative PDF",
            line={"color": "red"},
            legend="legend",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=dist_positive.pdf(x),
            mode="lines",
            name="Positive PDF",
            line={"color": "blue"},
            legend="legend",
        ),
        row=1,
        col=1,
    )


def _add_theory_roc_layer(
    figure: go.Figure,
    dist_negative: Distribution,
    dist_positive: Distribution,
    *,
    resolution: int,
    line_color: str = "purple",
    fpr: np.ndarray | None = None,
    tpr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the optimal (Neyman-Pearson) ROC curve to the ROC panel.

    See :func:`_optimal_roc_curve`: the curve orders outcomes by their likelihood
    ratio, which is the most powerful test and is correct even when the ratio is not
    monotone (a one-sided value threshold would be suboptimal there).
    """

    if fpr is None or tpr is None:
        fpr, tpr = _optimal_roc_curve(dist_negative, dist_positive, resolution=resolution)
    figure.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Theoretical ROC — AUC {auc(fpr, tpr):.3f}",
            line={"color": line_color},
            legend="legend2",
        ),
        row=1,
        col=2,
    )
    return fpr, tpr


def _add_sample_pdf_layer(
    figure: go.Figure,
    samples: np.ndarray,
    labels: np.ndarray,
    lower: float,
    upper: float,
    histogram_bins: int,
) -> None:
    """Add the negative/positive sample histograms to the PDF panel."""

    bin_size = (upper - lower) / histogram_bins
    common_bins = {"start": lower, "end": upper, "size": bin_size}
    figure.add_trace(
        go.Histogram(
            x=samples[labels == 0],
            histnorm="probability density",
            xbins=common_bins,
            opacity=0.4,
            name="Negative samples",
            marker_color="red",
            legend="legend",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=samples[labels == 1],
            histnorm="probability density",
            xbins=common_bins,
            opacity=0.4,
            name="Positive samples",
            marker_color="blue",
            legend="legend",
        ),
        row=1,
        col=1,
    )


def _sample_lr_scores(
    samples: np.ndarray,
    distribution_type: type[Distribution] | None,
    scale: float,
) -> np.ndarray:
    """Score samples by their log-likelihood ratio -- the optimal test statistic.

    Sorting samples by ``log f_positive - log f_negative`` is the empirical analogue
    of the Neyman-Pearson ROC, so the empirical curve converges to the optimal
    theoretical one (rather than to a suboptimal value-threshold curve). Sampling-only
    distributions have no closed-form density, so they fall back to the raw value.
    """

    if distribution_type is None:
        return samples
    dist_negative = _make_empirical_roc_dist(distribution_type, loc=0, scale=scale)
    dist_positive = _make_empirical_roc_dist(distribution_type, loc=1, scale=scale)
    return dist_positive.logpdf(samples) - dist_negative.logpdf(samples)


def _empirical_roc_points(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full step-function ROC used for privacy-bound calculations."""

    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return fpr, tpr


def _add_sample_roc_layer(
    figure: go.Figure,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    line_color: str = "green",
    line_dash: str = "dash",
) -> tuple[np.ndarray, np.ndarray]:
    """Add the empirical ROC curve (from sklearn ``roc_curve``) to the ROC panel."""

    fpr, tpr = _empirical_roc_points(labels, scores)
    figure.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Empirical ROC — AUC {auc(fpr, tpr):.3f}",
            line={"color": line_color, "dash": line_dash},
            legend="legend2",
        ),
        row=1,
        col=2,
    )
    return fpr, tpr


def _add_random_classifier_layer(figure: go.Figure) -> None:
    """Add the diagonal ``y = x`` reference line to the ROC panel."""

    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random classifier",
            line={"color": "black", "dash": "dash"},
            legend="legend2",
        ),
        row=1,
        col=2,
    )


def _finalize_roc_canvas(
    figure: go.Figure,
    title: str,
    lower: float,
    upper: float,
    *,
    top_margin: int | None = None,
) -> go.Figure:
    """Apply the shared layout, legends, and axis ranges to a populated canvas."""

    resolved_top_margin = top_margin if top_margin is not None else (90 if title else 55)
    figure.update_layout(
        title=title,
        width=_ROC_FIGURE_WIDTH,
        height=580,
        autosize=False,
        barmode="overlay",
        legend={
            "orientation": "v",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.78)",
            "bordercolor": "rgba(0,0,0,0.18)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        legend2={
            "orientation": "v",
            "x": 0.99,
            "xanchor": "right",
            "y": 0.03,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.78)",
            "bordercolor": "rgba(0,0,0,0.18)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        margin={"t": resolved_top_margin, "b": 70, "l": _ROC_FIGURE_MARGIN_L, "r": _ROC_FIGURE_MARGIN_R},
    )
    figure.update_xaxes(title_text="x", range=[lower, upper], row=1, col=1)
    figure.update_yaxes(title_text="Probability Density", row=1, col=1)
    figure.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=2)
    return figure


def make_theory_roc_figure(
    distribution: str,
    scale: float,
    *,
    resolution: int = 1000,
    figure_title: str | None = None,
) -> go.Figure:
    """Build the theory-only ROC explorer (PDF curves + exact ROC) for one distribution.

    Only analytic distributions are supported; sampling-only distributions (the
    lognormal difference) raise, since they have no closed-form density.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    display_name, distribution_type = _empirical_distribution(distribution)
    if distribution_type is None:
        raise ValueError(
            f"{display_name} has no closed-form density; use make_empirical_roc_figure with samples"
        )

    dist_negative = _make_empirical_roc_dist(distribution_type, loc=0, scale=scale)
    dist_positive = _make_empirical_roc_dist(distribution_type, loc=1, scale=scale)
    lower, upper, grid, fpr, tpr, _ = _get_analytic_roc_primitives(
        distribution_type,
        scale,
        resolution,
    )

    figure = _new_roc_canvas()
    _add_theory_pdf_layer(figure, dist_negative, dist_positive, grid)
    _add_theory_roc_layer(
        figure,
        dist_negative,
        dist_positive,
        resolution=resolution,
        fpr=fpr,
        tpr=tpr,
    )
    _add_random_classifier_layer(figure)
    resolved_title = (
        figure_title
        if figure_title is not None
        else f"{display_name} distributions: scale={scale:.2f}"
    )
    return _finalize_roc_canvas(figure, resolved_title, lower, upper)


def make_empirical_roc_figure(
    distribution: str,
    scale: float,
    n_samples: int,
    sample_seed: int | None = None,
    *,
    resolution: int = 1000,
    histogram_bins: int | None = None,
    compute_epsilon: bool = False,
    delta: float | None = None,
) -> go.Figure:
    """Build the empirical ROC explorer, layering samples on the theory when available.

    Analytic distributions also draw their theoretical PDFs and exact ROC curve.
    The lognormal difference has no closed form, so it is rendered from samples
    only and therefore requires ``sample_seed``.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if histogram_bins is not None and histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")
    resolved_delta = _DELTA_DEFAULT_EMPIRICAL if delta is None else delta
    if compute_epsilon and not _DELTA_MIN <= resolved_delta <= 1:
        raise ValueError(f"delta must be between {_DELTA_MIN} and 1")
    resolved_histogram_bins = (
        max(5, int(math.ceil(math.sqrt(n_samples)))) if histogram_bins is None else histogram_bins
    )

    display_name, distribution_type = _empirical_distribution(distribution)
    has_theory = distribution_type is not None

    samples = None
    labels = None
    if sample_seed is not None:
        samples, labels = _generate_empirical_roc_samples(
            display_name,
            scale,
            n_samples,
            sample_seed,
        )

    if not has_theory and samples is None:
        raise ValueError(
            f"{display_name} has no closed-form density; pass sample_seed to draw it from samples"
        )

    if has_theory:
        dist_negative = _make_empirical_roc_dist(distribution_type, loc=0, scale=scale)
        dist_positive = _make_empirical_roc_dist(distribution_type, loc=1, scale=scale)
        lower, upper, grid, theory_fpr, theory_tpr, _ = _get_analytic_roc_primitives(
            distribution_type,
            scale,
            resolution,
        )
    else:
        lower, upper = _samples_pdf_x_range(samples, labels)

    figure = _new_roc_canvas()
    fpr = tpr = None
    if has_theory:
        _add_theory_pdf_layer(figure, dist_negative, dist_positive, grid)
        _add_theory_roc_layer(
            figure,
            dist_negative,
            dist_positive,
            resolution=resolution,
            fpr=theory_fpr,
            tpr=theory_tpr,
        )
    if samples is not None and labels is not None:
        _add_sample_pdf_layer(figure, samples, labels, lower, upper, resolved_histogram_bins)
        scores = _sample_lr_scores(samples, distribution_type, scale)
        fpr, tpr = _add_sample_roc_layer(figure, scores, labels)
    _add_random_classifier_layer(figure)

    title = ""
    if compute_epsilon and fpr is not None and tpr is not None:
        from libdpy.assignment_specific.privacy_auditing.utils import (
            selected_threshold_from_empirical_roc,
        )

        samples_neg = samples[labels == 0]
        samples_pos = samples[labels == 1]
        tau_star, governing_point, epsilon = selected_threshold_from_empirical_roc(
            samples_neg,
            samples_pos,
            resolved_delta,
        )
        log_slope = epsilon if np.isfinite(epsilon) else float("inf")
        if np.isfinite(log_slope):
            _add_privacy_bound_layer(
                figure,
                log_slope,
                resolved_delta,
                resolution=resolution,
                line_color=_EPSILON_FIGURE_BOUND_COLOR,
                one_sided=True,
            )
            _add_roc_governing_point_layer(
                figure,
                fpr,
                tpr,
                resolved_delta,
                tau_star=tau_star,
                point=governing_point,
            )
    return _finalize_roc_canvas(figure, title, lower, upper)


def _format_scale_readout(scale: float) -> str:
    return f"{scale:.3g}"


def _scale_control_spec(scale: float) -> ControlSpec:
    return ControlSpec(
        name="scale",
        kind="slider",
        label="scale",
        default=scale,
        min=_EMPIRICAL_ROC_SCALE_MIN,
        max=max(_EMPIRICAL_ROC_SCALE_MAX, scale),
        step=0.05,
        continuous=True,
        slider_scale="log",
        readout_formatter=_format_scale_readout,
    )


def _n_samples_control_spec(n_samples: int) -> ControlSpec:
    return ControlSpec(
        name="n_samples",
        kind="slider",
        label="Samples / class",
        default=n_samples,
        min=10,
        max=max(5000, n_samples),
        step=1,
        continuous=False,
    )


def _distribution_control_spec(distribution: str, values: tuple[str, ...]) -> ControlSpec:
    return ControlSpec(
        name="distribution",
        kind="select",
        label="Distribution",
        default=distribution,
        values=values,
    )


def _theory_roc_control_specs(
    scale: float,
    distribution: str,
) -> tuple[ControlSpec, ControlSpec]:
    # Only analytic distributions can draw a theoretical curve, so the theory
    # explorer's selector excludes the sampling-only lognormal difference.
    return (
        _distribution_control_spec(distribution, _ANALYTIC_ROC_DISTRIBUTIONS),
        _scale_control_spec(scale),
    )


def _empirical_roc_control_specs(
    n_samples: int,
    scale: float,
    distribution: str,
) -> tuple[ControlSpec, ControlSpec, ControlSpec]:
    return (
        _distribution_control_spec(distribution, _EMPIRICAL_ROC_DISTRIBUTIONS),
        _scale_control_spec(scale),
        _n_samples_control_spec(n_samples),
    )


def _compute_epsilon_control_spec(*, default: bool = False) -> ControlSpec:
    return ControlSpec(
        name="compute_epsilon",
        kind="toggle_button",
        label="Compute ε",
        default=default,
        description="Compute the tightest (ε, δ)-DP guarantee for the current ROC",
    )


def _theory_roc_interactive_controls(
    scale: float,
    distribution: str,
    *,
    delta: float,
    compute_epsilon: bool,
    show_compute_epsilon_toggle: bool,
    selectable_distribution: bool,
) -> tuple[ControlSpec, ...]:
    controls: list[ControlSpec] = []
    if selectable_distribution:
        controls.extend(_theory_roc_control_specs(scale, distribution))
    else:
        controls.append(_scale_control_spec(scale))
    if show_compute_epsilon_toggle:
        controls.append(_compute_epsilon_control_spec(default=compute_epsilon))
    controls.append(_delta_control_spec(delta))
    return tuple(controls)


def _empirical_roc_interactive_controls(
    n_samples: int,
    scale: float,
    distribution: str,
    *,
    delta: float,
    compute_epsilon: bool,
    show_compute_epsilon_toggle: bool,
    selectable_distribution: bool,
) -> tuple[ControlSpec, ...]:
    controls: list[ControlSpec] = []
    if selectable_distribution:
        controls.extend(_empirical_roc_control_specs(n_samples, scale, distribution))
    else:
        controls.extend((_scale_control_spec(scale), _n_samples_control_spec(n_samples)))
    if show_compute_epsilon_toggle:
        controls.append(_compute_epsilon_control_spec(default=compute_epsilon))
    controls.append(_delta_control_spec(delta))
    return tuple(controls)


def _roc_visualizer_layout(
    *,
    distribution_name: str,
    selectable_distribution: bool,
    show_compute_epsilon_toggle: bool,
    below_left_footer: tuple[str, ...] = (),
    below_right_footer_actions: tuple[str, ...] = (),
    empirical: bool = False,
    sample_seed_getter: Callable[[], int | None] | None = None,
):
    from .interactive_widgets import roc_subplot_control_layout

    def _selected_distribution(controls: Mapping[str, Any]) -> str:
        if "distribution" in controls:
            return str(controls["distribution"].value)
        return distribution_name

    def scale_readout(controls: Mapping[str, Any]) -> str:
        return f"Scale = {_format_scale_readout(float(controls['scale'].value))}"

    def privacy_readout(controls: Mapping[str, Any]) -> str:
        distribution = _selected_distribution(controls)
        scale = float(controls["scale"].value)
        delta = float(controls["delta"].value)
        if empirical:
            sample_seed = sample_seed_getter() if sample_seed_getter is not None else None
            n_samples = int(round(float(controls["n_samples"].value)))
            return _empirical_privacy_dp_label(
                distribution,
                scale,
                delta,
                n_samples,
                sample_seed,
            )
        return _theory_privacy_dp_label(distribution, scale, delta)

    return roc_subplot_control_layout(
        scale_readout=scale_readout,
        privacy_readout=privacy_readout,
        compute_epsilon_control="compute_epsilon" if show_compute_epsilon_toggle else None,
        below_left=("scale",),
        below_right=("delta",),
        below_left_footer=below_left_footer,
        below_right_footer_actions=below_right_footer_actions,
        toolbar=("distribution",) if selectable_distribution else (),
        centered_title=distribution_name,
        centered_title_control="distribution" if selectable_distribution else None,
        figure_width=_ROC_FIGURE_WIDTH,
        slider_grid_template=roc_panel_slider_grid_template(),
    )


def _make_theory_roc_interactive_figure(
    distribution: str,
    scale: float,
    *,
    compute_epsilon: bool = False,
    delta: float = _DELTA_DEFAULT_THEORY,
    resolution: int = 1000,
) -> go.Figure:
    if compute_epsilon:
        return make_epsilon_from_delta_figure(
            distribution,
            scale,
            delta,
            resolution=resolution,
            figure_title="",
        )
    return make_theory_roc_figure(
        distribution,
        scale,
        resolution=resolution,
        figure_title="",
    )


def _make_empirical_roc_interactive_figure(
    distribution: str,
    scale: float,
    n_samples: int,
    sample_seed: int | None = None,
    *,
    compute_epsilon: bool = False,
    delta: float = _DELTA_DEFAULT_EMPIRICAL,
    resolution: int = 1000,
) -> go.Figure:
    return make_empirical_roc_figure(
        distribution,
        scale,
        n_samples,
        sample_seed,
        resolution=resolution,
        compute_epsilon=compute_epsilon,
        delta=delta,
    )


def _noop_action(_state, _controls):
    """Placeholder action handler for kernel-free (WASM) specs.

    The live ipywidgets renderer supplies a real handler; the WASM renderer ignores
    ``handler`` entirely and uses ``ActionSpec.state_updates`` instead.
    """

    return None


def _roc_artifact_name(prefix: str, **configuration: Any) -> str:
    """Stable, param-derived app id so distinct configs export distinct WASM apps."""

    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    distribution = str(configuration.get("distribution", ""))
    slug = re.sub(r"[^a-z0-9]+", "-", distribution.lower()).strip("-")
    return f"{prefix}-{slug}-{digest}" if slug else f"{prefix}-{digest}"


def theory_roc_spec(
    distribution: str = "Laplace",
    scale: float = 1.0,
    delta: float = _DELTA_DEFAULT_THEORY,
    *,
    compute_epsilon: bool = False,
    show_compute_epsilon_toggle: bool = True,
    selectable_distribution: bool = True,
) -> InteractiveSpec:
    """Backend-neutral spec for the theoretical ROC explorer (widget-free)."""

    distribution = _empirical_distribution(distribution)[0]
    controls = _theory_roc_interactive_controls(
        scale,
        distribution,
        delta=delta,
        compute_epsilon=compute_epsilon,
        show_compute_epsilon_toggle=show_compute_epsilon_toggle,
        selectable_distribution=selectable_distribution,
    )
    fixed_kwargs: dict[str, object] = {}
    if not selectable_distribution:
        fixed_kwargs["distribution"] = distribution
    if not show_compute_epsilon_toggle:
        fixed_kwargs["compute_epsilon"] = True
    return InteractiveSpec(
        name="theory_roc_visualizer",
        artifact_name=_roc_artifact_name(
            "theory-roc",
            distribution=distribution,
            scale=scale,
            delta=delta,
            compute_toggle=show_compute_epsilon_toggle,
            selectable=selectable_distribution,
        ),
        controls=controls,
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets", "wasm-marimo"),
        fixed_kwargs=fixed_kwargs,
        make_figure=partial(_make_theory_roc_interactive_figure, resolution=1000),
        figure_factory=("libdpy.visualization.roc_plots:_make_theory_roc_interactive_figure"),
        description="Theoretical ROC explorer for common distributions.",
    )


def empirical_roc_spec(
    n_samples: int,
    distribution: str = "Laplace",
    scale: float = 1.0,
    delta: float = _DELTA_DEFAULT_EMPIRICAL,
    *,
    compute_epsilon: bool = False,
    show_compute_epsilon_toggle: bool = True,
    selectable_distribution: bool = True,
    sample_seed: int = 0,
    action_handler: Callable[..., Any] | None = None,
) -> InteractiveSpec:
    """Backend-neutral spec for the empirical ROC explorer (widget-free).

    ``action_handler`` is the live-kernel resample callback; WASM ignores it and uses
    the declarative ``state_updates`` (advance ``sample_seed`` by one per click).
    """

    distribution = _empirical_distribution(distribution)[0]
    controls = _empirical_roc_interactive_controls(
        n_samples,
        scale,
        distribution,
        delta=delta,
        compute_epsilon=compute_epsilon,
        show_compute_epsilon_toggle=show_compute_epsilon_toggle,
        selectable_distribution=selectable_distribution,
    )
    fixed_kwargs: dict[str, object] = {}
    if not selectable_distribution:
        fixed_kwargs["distribution"] = distribution
    if not show_compute_epsilon_toggle:
        fixed_kwargs["compute_epsilon"] = True
    return InteractiveSpec(
        name="empirical_roc_visualizer",
        artifact_name=_roc_artifact_name(
            "empirical-roc",
            distribution=distribution,
            scale=scale,
            delta=delta,
            n_samples=n_samples,
            compute_toggle=show_compute_epsilon_toggle,
            selectable=selectable_distribution,
        ),
        controls=controls,
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets", "wasm-marimo"),
        fixed_kwargs=fixed_kwargs,
        make_figure=partial(_make_empirical_roc_interactive_figure, resolution=1000),
        figure_factory=("libdpy.visualization.roc_plots:_make_empirical_roc_interactive_figure"),
        description="Theoretical and empirical ROC explorer for common distributions.",
        actions=(
            ActionSpec(
                name="generate_samples",
                label="Generate Samples",
                handler=action_handler if action_handler is not None else _noop_action,
                button_style="info",
                state_updates={"sample_seed": 1},
            ),
        ),
        initial_state={"sample_seed": sample_seed},
    )


class EmpROCVisualizer(AbstractInteractivePlot):
    """ROC explorer backed by the generic interactive engine."""

    def __init__(
        self,
        n_samples: int,
        distribution: str = "Laplace",
        scale: float = 1.0,
        delta: float = _DELTA_DEFAULT_EMPIRICAL,
        *,
        compute_epsilon: bool = False,
        show_compute_epsilon_toggle: bool = True,
        selectable_distribution: bool = True,
        random_seed: int | None = None,
        auto_display: bool = True,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not _DELTA_MIN <= delta <= 1:
            raise ValueError(f"delta must be between {_DELTA_MIN} and 1")

        self.n_samples = int(n_samples)
        self.distribution = _empirical_distribution(distribution)[0]
        self.scale = float(scale)
        self.delta = float(delta)
        self.compute_epsilon = bool(compute_epsilon or not show_compute_epsilon_toggle)
        self.show_compute_epsilon_toggle = bool(show_compute_epsilon_toggle)
        self.selectable_distribution = bool(selectable_distribution)
        self._seed_generator = np.random.default_rng(random_seed)
        self._initial_sample_seed = int(self._seed_generator.integers(0, np.iinfo(np.int32).max))
        self._privacy_state = {"sample_seed": self._initial_sample_seed}
        self._rendered = self.widget()

        # Preserve the useful public attributes from the original implementation.
        # When the distribution is fixed there is no selector control to expose.
        self.distribution_control = self._rendered.controls.get("distribution")
        self.scale_slider = self._rendered.controls["scale"]
        self.sample_count_slider = self._rendered.controls["n_samples"]
        self.compute_epsilon_toggle = self._rendered.controls.get("compute_epsilon")
        self.delta_slider = self._rendered.controls["delta"]
        self.sample_button = self._rendered.actions["generate_samples"]
        self.interactive_plot = self._rendered.root

        if auto_display:
            display(self._rendered.root)

    def _generate_samples_action(self, state, _controls):
        state["sample_seed"] = int(self._seed_generator.integers(0, np.iinfo(np.int32).max))
        self._privacy_state["sample_seed"] = state["sample_seed"]
        return None

    def widget(self, **renderer_options):
        from .interactive_widgets import render_ipywidgets

        return render_ipywidgets(
            self.build_spec(),
            layout=_roc_visualizer_layout(
                distribution_name=self.distribution,
                selectable_distribution=self.selectable_distribution,
                show_compute_epsilon_toggle=self.show_compute_epsilon_toggle,
                below_left_footer=("n_samples",),
                below_right_footer_actions=("generate_samples",),
                empirical=True,
                sample_seed_getter=lambda: self._privacy_state.get("sample_seed"),
            ),
            **renderer_options,
        )

    def spec(self) -> InteractiveSpec:
        return empirical_roc_spec(
            self.n_samples,
            self.distribution,
            self.scale,
            self.delta,
            compute_epsilon=self.compute_epsilon,
            show_compute_epsilon_toggle=self.show_compute_epsilon_toggle,
            selectable_distribution=self.selectable_distribution,
            sample_seed=self._initial_sample_seed,
            action_handler=self._generate_samples_action,
        )

    @property
    def samples(self) -> tuple[np.ndarray, np.ndarray] | None:
        sample_seed = self._rendered.state["sample_seed"]
        if sample_seed is None:
            return None
        distribution = (
            self.distribution_control.value
            if self.distribution_control is not None
            else self.distribution
        )
        return _generate_empirical_roc_samples(
            distribution,
            self.scale_slider.value,
            int(round(self.sample_count_slider.value)),
            sample_seed,
        )

    def generate_samples(
        self,
        scale: float,
        distribution: str | None = None,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate samples directly; the old ``generate_samples(scale)`` still works."""

        resolved_seed = (
            int(self._seed_generator.integers(0, np.iinfo(np.int32).max)) if seed is None else seed
        )
        return _generate_empirical_roc_samples(
            distribution or self.distribution,
            scale,
            n_samples or self.n_samples,
            resolved_seed,
        )

    def on_button_clicked(self, _button=None) -> None:
        """Compatibility callback that triggers the generic resampling action."""

        self._generate_samples_action(
            self._rendered.state,
            {name: control.value for name, control in self._rendered.controls.items()},
        )
        self._rendered.update()

    def plot_curves(
        self,
        scale: float,
        distribution: str | None = None,
        n_samples: int | None = None,
        *,
        compute_epsilon: bool | None = None,
        delta: float | None = None,
    ) -> go.Figure:
        """Return the current state as a Plotly figure."""

        resolved_compute_epsilon = (
            self.compute_epsilon
            if compute_epsilon is None
            else compute_epsilon
        )
        resolved_delta = self.delta if delta is None else delta
        return _make_empirical_roc_interactive_figure(
            distribution or self.distribution,
            scale,
            n_samples or self.n_samples,
            self._rendered.state["sample_seed"],
            compute_epsilon=resolved_compute_epsilon,
            delta=resolved_delta,
        )

    def show(self):
        """Display the already-created widget without rebuilding its state."""

        display(self._rendered.root)
        return self._rendered.root


_ROC_RATE_EPS = 1e-12


def _optimal_roc_curve(
    dist_negative: Distribution,
    dist_positive: Distribution,
    *,
    resolution: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the optimal (Neyman-Pearson) ROC for an arbitrary distribution pair.

    The most powerful test at every level rejects where the likelihood ratio
    ``f_positive / f_negative`` is largest. Sweeping a threshold ``t`` down through that
    ratio and accumulating the mass of every cell with ratio ``>= t`` is the same as
    ordering cells by descending log-likelihood ratio and taking cumulative sums -- so
    that is what we do. A one-sided value threshold is optimal only when the ratio is
    monotone (Laplace, Gaussian, Logistic); for non-monotone-LR pairs (Student-t,
    Cauchy) the optimal rejection region is an interval, so a value threshold
    understates the achievable TPR -- and hence the privacy loss.

    The curve is concave by construction: the slope of the segment a cell contributes
    is exactly that cell's likelihood ratio, so descending-ratio order yields
    non-increasing slopes.
    """

    # Quantile-spaced edges of both distributions: dense in the tails (so the ROC
    # reaches very small FPR, needed for tiny delta) and around the medians.
    tail = np.logspace(-15, math.log10(0.5), max(2, resolution))
    probs = np.unique(np.concatenate([tail, 1.0 - tail]))
    edges = np.unique(
        np.concatenate([dist_negative.ppf(probs), dist_positive.ppf(probs)])
    )
    edges = edges[np.isfinite(edges)]
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    log_neg = dist_negative.logpdf(midpoints)
    log_pos = dist_positive.logpdf(midpoints)
    mass_negative = np.exp(log_neg) * widths
    mass_positive = np.exp(log_pos) * widths
    mass_negative /= mass_negative.sum()
    mass_positive /= mass_positive.sum()

    # Prepend the origin only; the normalized cumulative sums already end at (1, 1)
    # (appending an exact 1.0 could fall below the last cumulative value and break
    # the monotonicity that ``sklearn.metrics.auc`` requires).
    order = np.argsort(-(log_pos - log_neg))
    fpr = np.concatenate([[0.0], np.cumsum(mass_negative[order])])
    tpr = np.concatenate([[0.0], np.cumsum(mass_positive[order])])
    return fpr, tpr


_DISTRIBUTION_TYPE_TO_KEY: dict[type[Distribution], str] = {
    laplace: "laplace",
    norm: "norm",
    cauchy: "cauchy",
    logistic: "logistic",
}


def _distribution_cache_key(distribution_type: type[Distribution]) -> str:
    if distribution_type is t:
        return f"student_t:{_EMPIRICAL_ROC_STUDENT_T_DF}"
    try:
        return _DISTRIBUTION_TYPE_TO_KEY[distribution_type]
    except KeyError as error:
        raise ValueError(f"unsupported distribution type: {distribution_type!r}") from error


def _distribution_type_from_cache_key(distribution_key: str) -> type[Distribution]:
    if distribution_key.startswith("student_t:"):
        return t
    mapping: dict[str, type[Distribution]] = {
        "laplace": laplace,
        "norm": norm,
        "cauchy": cauchy,
        "logistic": logistic,
    }
    try:
        return mapping[distribution_key]
    except KeyError as error:
        raise ValueError(f"unsupported distribution cache key: {distribution_key!r}") from error


@lru_cache(maxsize=64)
def _analytic_roc_primitives_cached(
    distribution_key: str,
    scale: float,
    resolution: int,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...], bool]:
    """Return ``(lower, upper, fpr, tpr, lr_unbounded)`` for an analytic distribution pair."""

    distribution_type = _distribution_type_from_cache_key(distribution_key)
    dist_negative = _make_empirical_roc_dist(distribution_type, loc=0, scale=scale)
    dist_positive = _make_empirical_roc_dist(distribution_type, loc=1, scale=scale)
    lower, upper = _empirical_roc_pdf_x_range(dist_negative, dist_positive)
    fpr, tpr = _optimal_roc_curve(dist_negative, dist_positive, resolution=resolution)
    lr_unbounded = _likelihood_ratio_unbounded(dist_negative, dist_positive)
    return (
        lower,
        upper,
        tuple(float(value) for value in fpr),
        tuple(float(value) for value in tpr),
        lr_unbounded,
    )


def _get_analytic_roc_primitives(
    distribution_type: type[Distribution],
    scale: float,
    resolution: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return cached PDF grid bounds, optimal ROC arrays, and LR-unbounded flag."""

    lower, upper, fpr_tuple, tpr_tuple, lr_unbounded = _analytic_roc_primitives_cached(
        _distribution_cache_key(distribution_type),
        scale,
        resolution,
    )
    grid = np.linspace(lower, upper, resolution)
    return (
        lower,
        upper,
        grid,
        np.asarray(fpr_tuple, dtype=float),
        np.asarray(tpr_tuple, dtype=float),
        lr_unbounded,
    )


def _likelihood_ratio_unbounded(
    dist_negative: Distribution,
    dist_positive: Distribution,
) -> bool:
    """Whether the likelihood ratio is unbounded (so epsilon at ``delta = 0`` is infinite).

    Probe two successively deeper tail quantiles: if the log-likelihood ratio is still
    growing at the deeper point, the ratio diverges in that tail (e.g. Gaussian, whose
    ratio is ``exp`` of a line); if it plateaus (Laplace, Logistic) or decays (Cauchy,
    Student-t) the ratio is bounded and the tightest epsilon is finite.
    """

    depths = np.array([1e-12, 1e-15])
    right_x = dist_negative.isf(depths)
    left_x = dist_negative.ppf(depths)
    right_llr = dist_positive.logpdf(right_x) - dist_negative.logpdf(right_x)
    left_llr = dist_negative.logpdf(left_x) - dist_positive.logpdf(left_x)
    return bool(right_llr[1] > right_llr[0] + 1e-6 or left_llr[1] > left_llr[0] + 1e-6)


# Public aliases for lecture figures and external notebooks.
optimal_roc_curve = _optimal_roc_curve
likelihood_ratio_unbounded = _likelihood_ratio_unbounded


def _roc_slope_supremum_unbounded(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> bool:
    """Return whether the required slope keeps increasing as FPR approaches zero."""

    positive = fpr > _ROC_RATE_EPS
    if np.count_nonzero(positive) < 2:
        return False

    order = np.argsort(fpr[positive])
    smallest_fpr = fpr[positive][order]
    tpr_slopes = (tpr[positive][order] - delta) / smallest_fpr
    if tpr_slopes[0] <= _ROC_RATE_EPS or tpr_slopes[1] <= _ROC_RATE_EPS:
        return False
    return bool(tpr_slopes[0] > tpr_slopes[1] * (1.0 + 1e-4))


def _roc_inverse_slope_supremum_unbounded(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> bool:
    """Return whether ``(FPR - delta) / TPR`` keeps increasing as TPR approaches zero."""

    positive = tpr > _ROC_RATE_EPS
    if np.count_nonzero(positive) < 2:
        return False

    order = np.argsort(tpr[positive])
    smallest_tpr = tpr[positive][order]
    fpr_slopes = (fpr[positive][order] - delta) / smallest_tpr
    if fpr_slopes[0] <= _ROC_RATE_EPS or fpr_slopes[1] <= _ROC_RATE_EPS:
        return False
    return bool(fpr_slopes[0] > fpr_slopes[1] * (1.0 + 1e-4))


def _roc_origin_constraints_violated(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> bool:
    """Return whether either axis intercept violates the ``(epsilon, delta)`` strip."""

    zero_fpr = fpr <= _ROC_RATE_EPS
    if np.any(zero_fpr) and np.max(tpr[zero_fpr]) > delta + _ROC_RATE_EPS:
        return True
    zero_tpr = tpr <= _ROC_RATE_EPS
    return bool(np.any(zero_tpr) and np.max(fpr[zero_tpr]) > delta + _ROC_RATE_EPS)


def _roc_step_binding(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> tuple[float, float, float]:
    """Return ``(min_slope, governing_fpr, governing_tpr)`` for a step ROC at ``delta``."""

    min_slope = 0.0
    governing_fpr = 0.0
    governing_tpr = float(delta)

    def _consider(slope: float, point_fpr: float, point_tpr: float) -> None:
        nonlocal min_slope, governing_fpr, governing_tpr
        if slope > min_slope + _ROC_RATE_EPS:
            min_slope = slope
            governing_fpr = point_fpr
            governing_tpr = point_tpr

    positive_fpr = fpr > _ROC_RATE_EPS
    positive_tpr = tpr > _ROC_RATE_EPS

    if np.any(positive_fpr):
        slopes = (tpr[positive_fpr] - delta) / fpr[positive_fpr]
        best = int(np.argmax(slopes))
        fpr_values = fpr[positive_fpr]
        tpr_values = tpr[positive_fpr]
        _consider(float(slopes[best]), float(fpr_values[best]), float(tpr_values[best]))
    if np.any(positive_tpr):
        slopes = (fpr[positive_tpr] - delta) / tpr[positive_tpr]
        best = int(np.argmax(slopes))
        fpr_values = fpr[positive_tpr]
        tpr_values = tpr[positive_tpr]
        _consider(float(slopes[best]), float(fpr_values[best]), float(tpr_values[best]))

    for index in range(1, len(fpr)):
        f0, t0 = float(fpr[index - 1]), float(tpr[index - 1])
        f1, t1 = float(fpr[index]), float(tpr[index])
        if f1 > f0 + _ROC_RATE_EPS and t0 > _ROC_RATE_EPS:
            _consider((f1 - delta) / t0, f1, t0)
        if t1 > t0 + _ROC_RATE_EPS and f1 > _ROC_RATE_EPS:
            _consider((t1 - delta) / f1, f1, t1)

    return min_slope, governing_fpr, governing_tpr


def _reflect_roc_through_antidiagonal(
    fpr: np.ndarray,
    tpr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect a ROC through the anti-diagonal: ``(FPR, TPR) -> (1 - TPR, 1 - FPR)``.

    The DP inequality that binds near ``(1, 1)`` (the curve's "top half") is exactly the
    origin-side inequality of this reflection, so the top half can be evaluated by the same
    origin-side machinery applied to the reflected curve. A symmetric ROC reflects onto
    itself, so its two halves already agree; an asymmetric (sampled) ROC does not, which is
    why the top half must be checked separately. Points are reversed so ``FPR`` ascends.
    """

    reflected_fpr = np.ascontiguousarray((1.0 - tpr)[::-1])
    reflected_tpr = np.ascontiguousarray((1.0 - fpr)[::-1])
    return reflected_fpr, reflected_tpr


def _roc_step_binding_point(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> tuple[float, tuple[float, float] | None]:
    """Return the exact step binding as ``(min_slope, governing_point | None)``."""

    min_slope, governing_fpr, governing_tpr = _roc_step_binding(fpr, tpr, delta)
    if min_slope <= _ROC_RATE_EPS:
        return min_slope, None
    return min_slope, (governing_fpr, governing_tpr)


def _bottom_half_binding(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> tuple[float, tuple[float, float] | None]:
    """Return ``(min_slope, governing_point)`` for the DP inequality binding near ``(0, 0)``.

    ``min_slope`` is ``inf`` when no finite slope bounds the steep rise out of the origin
    (e.g. a Gaussian ROC at ``delta = 0``); the slope-supremum detectors recognise that
    divergence on a smoothly sampled analytic curve.
    """

    if (
        _roc_origin_constraints_violated(fpr, tpr, delta)
        or _roc_slope_supremum_unbounded(fpr, tpr, delta)
        or _roc_inverse_slope_supremum_unbounded(fpr, tpr, delta)
    ):
        return float("inf"), None
    return _roc_step_binding_point(fpr, tpr, delta)


def _top_half_binding(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> tuple[float, tuple[float, float] | None]:
    """Return ``(min_slope, governing_point)`` for the DP inequality binding near ``(1, 1)``.

    Evaluated as the origin-side binding of the anti-diagonal reflection. Only the exact
    axis-touching infinity check is applied: a finite step ROC reaches an infinite slope
    only by touching an axis inside the ``(eps, delta)`` strip, whereas the smooth-curve
    slope-supremum heuristics misfire on a finite step reflection (they read a merely
    decreasing-toward-the-corner slope as a divergence). The governing point is mapped back
    to the original ROC coordinates.
    """

    reflected_fpr, reflected_tpr = _reflect_roc_through_antidiagonal(fpr, tpr)
    if _roc_origin_constraints_violated(reflected_fpr, reflected_tpr, delta):
        return float("inf"), None
    min_slope, reflected_point = _roc_step_binding_point(reflected_fpr, reflected_tpr, delta)
    if reflected_point is None:
        return min_slope, None
    return min_slope, (1.0 - reflected_point[1], 1.0 - reflected_point[0])


def _roc_dp_step_binding(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> tuple[float, tuple[float, float] | None]:
    """Return ``(min_slope, governing_point)`` enforcing both halves of the ROC.

    The bottom-half binding alone ignores the symmetric DP inequality that binds near
    ``(1, 1)`` -- harmless for a symmetric ROC but not for an asymmetric (sampled) one. We
    keep whichever half demands the larger slope.
    """

    bottom_slope, bottom_point = _bottom_half_binding(fpr, tpr, delta)
    top_slope, top_point = _top_half_binding(fpr, tpr, delta)
    if top_slope > bottom_slope:
        return top_slope, top_point
    return bottom_slope, bottom_point


def find_roc_governing_point(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> tuple[float, float] | None:
    """Return the ROC point that sets the tightest ``(epsilon, delta)`` slope, if any."""

    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")
    if len(fpr) == 0:
        raise ValueError("fpr and tpr must not be empty")

    min_slope, governing_point = _roc_dp_step_binding(fpr, tpr, delta)
    if not np.isfinite(min_slope) or min_slope <= _ROC_RATE_EPS:
        return None
    return governing_point


def compute_min_log_slope_for_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> float:
    """Return ``ln(m)`` for the smallest slope ``m`` that upper bounds the ROC at ``delta``."""

    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")
    if len(fpr) == 0:
        raise ValueError("fpr and tpr must not be empty")

    min_slope, _ = _roc_dp_step_binding(fpr, tpr, delta)
    if not np.isfinite(min_slope):
        return float("inf")
    if min_slope <= _ROC_RATE_EPS:
        return 0.0
    return float(np.log(min_slope))


def compute_epsilon_for_delta(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
) -> float:
    """Return the minimum ``epsilon`` such that the ROC is ``(epsilon, delta)``-DP.

    This is exact for the step ROCs produced by sampling (sklearn ``roc_curve``);
    for the smooth, concave optimal ROC use :func:`optimal_roc_epsilon_for_delta`.
    """

    log_slope = compute_min_log_slope_for_roc(fpr, tpr, delta)
    if not np.isfinite(log_slope):
        return float("inf")
    return max(0.0, log_slope)


def optimal_roc_epsilon_for_delta(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
    *,
    lr_unbounded: bool,
) -> float:
    """Return the minimum ``epsilon`` for which a *concave* ROC is ``(epsilon, delta)``-DP.

    The optimal ROC (:func:`_optimal_roc_curve`) is concave, so the tightest slope is
    attained at one of its points and both DP inequalities reduce to chord slopes from
    the ``delta`` offsets -- no step-corner terms (which would fabricate operating
    points below a smooth curve).

    Epsilon is infinite only at ``delta = 0`` with an unbounded likelihood ratio: then
    the curve rises vertically out of the origin and no finite slope bounds it. For any
    ``delta > 0`` the slack clears that vertical start, so the chord slope (read off the
    deep-tail quantile grid) is finite and correct. ``lr_unbounded`` is supplied by the
    caller via :func:`_likelihood_ratio_unbounded` -- it cannot be read off a finite
    grid, on which the largest slope is always finite.
    """

    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")
    if len(fpr) == 0:
        raise ValueError("fpr and tpr must not be empty")

    if delta <= 0 and lr_unbounded:
        return float("inf")

    slope = 0.0
    positive_fpr = fpr > _ROC_RATE_EPS
    positive_tpr = tpr > _ROC_RATE_EPS
    if np.any(positive_fpr):
        slope = max(slope, float(np.max((tpr[positive_fpr] - delta) / fpr[positive_fpr])))
    if np.any(positive_tpr):
        slope = max(slope, float(np.max((fpr[positive_tpr] - delta) / tpr[positive_tpr])))
    if slope <= _ROC_RATE_EPS:
        return 0.0
    return max(0.0, float(np.log(slope)))


def _format_epsilon_for_title(epsilon: float) -> str:
    if not np.isfinite(epsilon):
        return "∞"
    if epsilon <= _ROC_RATE_EPS:
        return "0"
    return f"{epsilon:.3g}"


def _format_privacy_dp_label(epsilon: float, delta: float) -> str:
    """Return a compact ``(epsilon, delta)-DP`` label for widget readouts."""

    return f"({_format_epsilon_for_title(epsilon)}, {_format_delta_scientific(delta)})-DP"


def _theory_privacy_dp_label(
    distribution: str,
    scale: float,
    delta: float,
    *,
    resolution: int = 1000,
) -> str:
    """Compute the ``(epsilon, delta)-DP`` label for an analytic distribution pair."""

    _, distribution_type = _empirical_distribution(distribution)
    if distribution_type is None:
        raise ValueError(f"{distribution} has no closed-form density")

    _, _, _, fpr, tpr, lr_unbounded = _get_analytic_roc_primitives(
        distribution_type,
        scale,
        resolution,
    )
    if delta <= 0 and lr_unbounded:
        epsilon = float("inf")
    else:
        from libdpy.assignment_specific.privacy_auditing.utils import (
            one_sided_epsilon_from_roc_points,
        )

        epsilon, _ = one_sided_epsilon_from_roc_points(fpr, tpr, delta)
    return _format_privacy_dp_label(epsilon, delta)


def _empirical_privacy_dp_label(
    distribution: str,
    scale: float,
    delta: float,
    n_samples: int,
    sample_seed: int | None,
) -> str:
    """Compute the ``(epsilon, delta)-DP`` label from the current empirical ROC sample."""

    if sample_seed is None or n_samples <= 0:
        return ""
    display_name, distribution_type = _empirical_distribution(distribution)
    samples, labels = _generate_empirical_roc_samples(
        display_name,
        scale,
        n_samples,
        sample_seed,
    )
    scores = _sample_lr_scores(samples, distribution_type, scale)
    fpr, tpr = _empirical_roc_points(labels, scores)
    samples_neg = samples[labels == 0]
    samples_pos = samples[labels == 1]
    from libdpy.assignment_specific.privacy_auditing.utils import (
        selected_threshold_from_empirical_roc,
    )

    _, _, epsilon = selected_threshold_from_empirical_roc(samples_neg, samples_pos, delta)
    return _format_privacy_dp_label(epsilon, delta)


def _format_delta_scientific(delta: float) -> str:
    """Format ``delta`` as a power of ten, e.g. ``10^-6``."""

    if delta <= 0:
        return "0"
    if abs(delta - 1.0) < _ROC_RATE_EPS:
        return "1"

    log10 = math.log10(delta)
    if abs(log10 - round(log10)) < max(1e-9, abs(log10) * 1e-9):
        exponent = int(round(log10))
        if exponent == 0:
            return "1"
        return f"10^{exponent}"
    return f"10^{log10:.2g}"


def _delta_control_spec(delta: float = _DELTA_DEFAULT_THEORY) -> ControlSpec:
    return ControlSpec(
        name="delta",
        kind="slider",
        label="δ",
        default=delta,
        min=_DELTA_MIN,
        max=1.0,
        step=0.05,
        continuous=True,
        slider_scale="log",
        readout_formatter=_format_delta_scientific,
    )


def _add_privacy_bound_layer(
    figure: go.Figure,
    log_slope: float,
    delta: float,
    *,
    resolution: int,
    line_color: str = "blue",
    one_sided: bool = False,
) -> None:
    if one_sided:
        bound_x, bound_y = one_sided_privacy_bound(log_slope, delta, resolution)
    else:
        bound_x, bound_y = calc_privacy_bound(log_slope, delta, resolution)
    figure.add_trace(
        go.Scatter(
            x=bound_x,
            y=bound_y,
            mode="lines",
            name="(ε, δ) bound",
            line={"color": line_color, "width": 3},
            legend="legend2",
        ),
        row=1,
        col=2,
    )


def _format_tau_star_legend(tau_star: float) -> str:
    if not math.isfinite(tau_star):
        return "ε-governing point"
    return f"ε-governing point (τ★={tau_star:.3g})"


def _add_roc_governing_point_layer(
    figure: go.Figure,
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float,
    *,
    samples_neg: np.ndarray | None = None,
    samples_pos: np.ndarray | None = None,
    tau_star: float | None = None,
    point: tuple[float, float] | None = None,
) -> None:
    """Mark the ROC point used for the lecture's one-sided compute-ε step."""

    if point is not None:
        point_fpr, point_tpr = point
        trace_name = _format_tau_star_legend(float("nan") if tau_star is None else tau_star)
    elif samples_neg is not None and samples_pos is not None:
        from libdpy.assignment_specific.privacy_auditing.utils import (
            selected_threshold_from_empirical_roc,
        )

        tau_star, (point_fpr, point_tpr), _ = selected_threshold_from_empirical_roc(
            samples_neg,
            samples_pos,
            delta,
        )
        trace_name = _format_tau_star_legend(tau_star)
    else:
        governing_point = find_roc_governing_point(fpr, tpr, delta)
        if governing_point is None:
            return
        point_fpr, point_tpr = governing_point
        trace_name = "ε-governing point"

    figure.add_trace(
        go.Scatter(
            x=[point_fpr],
            y=[point_tpr],
            mode="markers",
            name=trace_name,
            marker={
                "symbol": "circle-open",
                "size": 14,
                "color": _EPSILON_FIGURE_BOUND_COLOR,
                "line": {"width": 3, "color": _EPSILON_FIGURE_BOUND_COLOR},
            },
            legend="legend2",
        ),
        row=1,
        col=2,
    )


def make_epsilon_from_delta_figure(
    distribution: str,
    scale: float,
    delta: float,
    *,
    resolution: int = 1000,
    figure_title: str | None = None,
) -> go.Figure:
    """Build the ROC explorer that computes the tightest ``epsilon`` for a fixed ``delta``."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")

    display_name, distribution_type = _empirical_distribution(distribution)
    if distribution_type is None:
        raise ValueError(
            f"{display_name} has no closed-form density; use the empirical ROC explorer instead"
        )

    dist_negative = _make_empirical_roc_dist(distribution_type, loc=0, scale=scale)
    dist_positive = _make_empirical_roc_dist(distribution_type, loc=1, scale=scale)
    lower, upper, grid, fpr, tpr, lr_unbounded = _get_analytic_roc_primitives(
        distribution_type,
        scale,
        resolution,
    )

    if delta <= 0 and lr_unbounded:
        epsilon = float("inf")
    else:
        from libdpy.assignment_specific.privacy_auditing.utils import (
            one_sided_epsilon_from_roc_points,
        )

        epsilon, _ = one_sided_epsilon_from_roc_points(fpr, tpr, delta)
    log_slope = epsilon if np.isfinite(epsilon) else float("inf")

    figure = _new_roc_canvas()
    _add_theory_pdf_layer(figure, dist_negative, dist_positive, grid)
    # Draw the same optimal (Neyman-Pearson) ROC that the epsilon/bound are computed
    # from, so the bound stays tangent to the curve. For non-monotone-LR pairs
    # (Cauchy, Student-t) this lies strictly above a one-sided value-threshold ROC.
    figure.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Theoretical ROC — AUC {auc(fpr, tpr):.3f}",
            line={"color": _EPSILON_FIGURE_ROC_COLOR},
            legend="legend2",
        ),
        row=1,
        col=2,
    )
    if np.isfinite(log_slope):
        _add_privacy_bound_layer(
            figure,
            log_slope,
            delta,
            resolution=resolution,
            line_color=_EPSILON_FIGURE_BOUND_COLOR,
            one_sided=True,
        )
    _add_random_classifier_layer(figure)

    resolved_title = (
        figure_title
        if figure_title is not None
        else f"{display_name} distributions: scale={scale:.2g}"
    )
    return _finalize_roc_canvas(figure, resolved_title, lower, upper)


def make_empirical_epsilon_from_delta_figure(
    distribution: str,
    scale: float,
    delta: float,
    n_samples: int,
    sample_seed: int | None = None,
    *,
    resolution: int = 1000,
    histogram_bins: int | None = None,
) -> go.Figure:
    """Build the sampled ROC explorer that computes ``epsilon`` from an empirical ROC."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if not _DELTA_MIN <= delta <= 1:
        raise ValueError(f"delta must be between {_DELTA_MIN} and 1")
    if histogram_bins is not None and histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")
    if sample_seed is None:
        raise ValueError("sample_seed is required to draw an empirical ROC")

    resolved_histogram_bins = (
        max(5, int(math.ceil(math.sqrt(n_samples)))) if histogram_bins is None else histogram_bins
    )

    display_name, distribution_type = _empirical_distribution(distribution)
    samples, labels = _generate_empirical_roc_samples(
        display_name,
        scale,
        n_samples,
        sample_seed,
    )

    if distribution_type is not None:
        dist_negative = _make_empirical_roc_dist(distribution_type, loc=0, scale=scale)
        dist_positive = _make_empirical_roc_dist(distribution_type, loc=1, scale=scale)
        lower, upper = _empirical_roc_pdf_x_range(dist_negative, dist_positive)
    else:
        lower, upper = _samples_pdf_x_range(samples, labels)

    figure = _new_roc_canvas()
    _add_sample_pdf_layer(figure, samples, labels, lower, upper, resolved_histogram_bins)
    scores = _sample_lr_scores(samples, distribution_type, scale)
    fpr, tpr = _empirical_roc_points(labels, scores)
    figure.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Empirical ROC — AUC {auc(fpr, tpr):.3f}",
            line={"color": _EPSILON_FIGURE_ROC_COLOR, "dash": "solid"},
            legend="legend2",
        ),
        row=1,
        col=2,
    )
    samples_neg = samples[labels == 0]
    samples_pos = samples[labels == 1]
    from libdpy.assignment_specific.privacy_auditing.utils import (
        selected_threshold_from_empirical_roc,
    )

    tau_star, governing_point, epsilon = selected_threshold_from_empirical_roc(
        samples_neg,
        samples_pos,
        delta,
    )
    log_slope = epsilon if np.isfinite(epsilon) else float("inf")
    if np.isfinite(log_slope):
        _add_privacy_bound_layer(
            figure,
            log_slope,
            delta,
            resolution=resolution,
            line_color=_EPSILON_FIGURE_BOUND_COLOR,
            one_sided=True,
        )
        _add_roc_governing_point_layer(
            figure,
            fpr,
            tpr,
            delta,
            tau_star=tau_star,
            point=governing_point,
        )
    _add_random_classifier_layer(figure)

    return _finalize_roc_canvas(figure, "", lower, upper)


class TheoryROCVisualizer(AbstractInteractivePlot):
    """Theoretical ROC explorer with optional epsilon-from-delta overlay."""

    def __init__(
        self,
        distribution: str = "Laplace",
        scale: float = 1.0,
        delta: float = _DELTA_DEFAULT_THEORY,
        *,
        compute_epsilon: bool = False,
        show_compute_epsilon_toggle: bool = True,
        selectable_distribution: bool = True,
        auto_display: bool = True,
    ):
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not _DELTA_MIN <= delta <= 1:
            raise ValueError(f"delta must be between {_DELTA_MIN} and 1")

        self.distribution = _empirical_distribution(distribution)[0]
        self.scale = float(scale)
        self.delta = float(delta)
        self.compute_epsilon = bool(compute_epsilon or not show_compute_epsilon_toggle)
        self.show_compute_epsilon_toggle = bool(show_compute_epsilon_toggle)
        self.selectable_distribution = bool(selectable_distribution)
        self._rendered = self.widget()

        # When the distribution is fixed there is no selector control to expose.
        self.distribution_control = self._rendered.controls.get("distribution")
        self.scale_slider = self._rendered.controls["scale"]
        self.compute_epsilon_toggle = self._rendered.controls.get("compute_epsilon")
        self.delta_slider = self._rendered.controls["delta"]
        self.interactive_plot = self._rendered.root

        if auto_display:
            display(self._rendered.root)

    def widget(self, **renderer_options):
        from .interactive_widgets import render_ipywidgets

        return render_ipywidgets(
            self.build_spec(),
            layout=_roc_visualizer_layout(
                distribution_name=self.distribution,
                selectable_distribution=self.selectable_distribution,
                show_compute_epsilon_toggle=self.show_compute_epsilon_toggle,
            ),
            **renderer_options,
        )

    def spec(self) -> InteractiveSpec:
        return theory_roc_spec(
            self.distribution,
            self.scale,
            self.delta,
            compute_epsilon=self.compute_epsilon,
            show_compute_epsilon_toggle=self.show_compute_epsilon_toggle,
            selectable_distribution=self.selectable_distribution,
        )

    def plot_curves(
        self,
        scale: float,
        distribution: str | None = None,
        *,
        compute_epsilon: bool | None = None,
        delta: float | None = None,
    ) -> go.Figure:
        """Return the current state as a Plotly figure."""

        resolved_compute_epsilon = (
            self.compute_epsilon
            if compute_epsilon is None
            else compute_epsilon
        )
        resolved_delta = self.delta if delta is None else delta
        return _make_theory_roc_interactive_figure(
            distribution or self.distribution,
            scale,
            compute_epsilon=resolved_compute_epsilon,
            delta=resolved_delta,
        )

    def show(self):
        """Display the already-created widget without rebuilding its state."""

        display(self._rendered.root)
        return self._rendered.root


class EpsilonFromDeltaVisualizer(TheoryROCVisualizer):
    """Backward-compatible alias for ``TheoryROCVisualizer`` with epsilon mode enabled."""

    def __init__(
        self,
        distribution: str = "Laplace",
        scale: float = 1.0,
        delta: float = _DELTA_DEFAULT_THEORY,
        *,
        selectable_distribution: bool = True,
        auto_display: bool = True,
    ):
        super().__init__(
            distribution=distribution,
            scale=scale,
            delta=delta,
            compute_epsilon=True,
            show_compute_epsilon_toggle=False,
            selectable_distribution=selectable_distribution,
            auto_display=auto_display,
        )


class EmpiricalEpsilonFromDeltaVisualizer(EmpROCVisualizer):
    """Empirical ROC explorer with the compute-ε toggle exposed (off by default)."""

    def __init__(
        self,
        n_samples: int,
        distribution: str = "Laplace",
        scale: float = 1.0,
        delta: float = _DELTA_DEFAULT_EMPIRICAL,
        *,
        selectable_distribution: bool = True,
        random_seed: int | None = None,
        auto_display: bool = True,
    ):
        super().__init__(
            n_samples,
            distribution=distribution,
            scale=scale,
            delta=delta,
            compute_epsilon=False,
            show_compute_epsilon_toggle=True,
            selectable_distribution=selectable_distribution,
            random_seed=random_seed,
            auto_display=auto_display,
        )

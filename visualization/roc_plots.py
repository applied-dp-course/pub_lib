import math
from enum import Enum
from functools import partial
from typing import Callable

import numpy as np
import plotly.graph_objects as go
from IPython.core.display_functions import display
from plotly.subplots import make_subplots
from scipy.stats import laplace, norm
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


def _empirical_distribution(name: str):
    normalized = name.strip().lower()
    if normalized in {"gaussian", "normal", "norm"}:
        return "Gaussian", norm
    if normalized == "laplace":
        return "Laplace", laplace
    raise ValueError("distribution must be 'Gaussian' or 'Laplace'")


def _generate_empirical_roc_samples(
    distribution: str,
    scale: float,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, distribution_type = _empirical_distribution(distribution)
    rng = np.random.default_rng(seed)
    negative_samples = distribution_type.rvs(
        loc=0,
        scale=scale,
        size=n_samples,
        random_state=rng,
    )
    positive_samples = distribution_type.rvs(
        loc=1,
        scale=scale,
        size=n_samples,
        random_state=rng,
    )
    labels = np.concatenate([np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=int)])
    return np.concatenate([negative_samples, positive_samples]), labels


def make_empirical_roc_figure(
    distribution: str,
    scale: float,
    n_samples: int,
    sample_seed: int | None = None,
    *,
    resolution: int = 1000,
    histogram_bins: int | None = None,
) -> go.Figure:
    """Build the Gaussian/Laplace theoretical and empirical ROC explorer."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if histogram_bins is not None and histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")
    resolved_histogram_bins = (
        max(5, int(math.ceil(math.sqrt(n_samples)))) if histogram_bins is None else histogram_bins
    )

    display_name, distribution_type = _empirical_distribution(distribution)
    dist_negative = distribution_type(loc=0, scale=scale)
    dist_positive = distribution_type(loc=1, scale=scale)
    lower = min(dist_negative.ppf(0.001), dist_positive.ppf(0.001))
    upper = max(dist_negative.ppf(0.999), dist_positive.ppf(0.999))
    samples = None
    labels = None
    if sample_seed is not None:
        samples, labels = _generate_empirical_roc_samples(
            display_name,
            scale,
            n_samples,
            sample_seed,
        )
        lower = min(lower, float(np.min(samples)))
        upper = max(upper, float(np.max(samples)))

    x = np.linspace(lower, upper, resolution)
    thresholds = np.linspace(lower, upper, resolution)
    fpr_theoretical = dist_negative.sf(thresholds)
    tpr_theoretical = dist_positive.sf(thresholds)
    theoretical_auc = auc(fpr_theoretical, tpr_theoretical)

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Probability Density Functions", "ROC Curves"),
        horizontal_spacing=0.08,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=dist_negative.pdf(x),
            mode="lines",
            name="Negative theory",
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
            name="Positive theory",
            line={"color": "blue"},
            legend="legend",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=fpr_theoretical,
            y=tpr_theoretical,
            mode="lines",
            name=f"Theoretical ROC — AUC {theoretical_auc:.3f}",
            line={"color": "purple"},
            legend="legend2",
        ),
        row=1,
        col=2,
    )

    if samples is not None and labels is not None:
        bin_size = (upper - lower) / resolved_histogram_bins
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
        fpr_empirical, tpr_empirical, _ = roc_curve(labels, samples)
        empirical_auc = auc(fpr_empirical, tpr_empirical)
        figure.add_trace(
            go.Scatter(
                x=fpr_empirical,
                y=tpr_empirical,
                mode="lines",
                name=f"Empirical ROC — AUC {empirical_auc:.3f}",
                line={"color": "green", "dash": "dash"},
                legend="legend2",
            ),
            row=1,
            col=2,
        )

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
    figure.update_layout(
        title=(
            f"{display_name} distributions: scale={scale:.2f}, " f"samples per class={n_samples}"
        ),
        width=1000,
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
        margin={"t": 90, "b": 70, "l": 65, "r": 35},
    )
    figure.update_xaxes(title_text="x", range=[lower, upper], row=1, col=1)
    figure.update_yaxes(title_text="Probability Density", row=1, col=1)
    figure.update_xaxes(
        title_text="False Positive Rate",
        range=[0, 1],
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text="True Positive Rate",
        range=[0, 1],
        row=1,
        col=2,
    )
    return figure


class EmpROCVisualizer(AbstractInteractivePlot):
    """Gaussian/Laplace ROC explorer backed by the generic interactive engine."""

    def __init__(
        self,
        n_samples: int,
        distribution: str = "Laplace",
        scale: float = 1.0,
        *,
        random_seed: int | None = None,
        auto_display: bool = True,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")

        self.n_samples = int(n_samples)
        self.distribution = _empirical_distribution(distribution)[0]
        self.scale = float(scale)
        self._seed_generator = np.random.default_rng(random_seed)
        self._rendered = self.widget()

        # Preserve the useful public attributes from the original implementation.
        self.distribution_control = self._rendered.controls["distribution"]
        self.scale_slider = self._rendered.controls["scale"]
        self.sample_count_slider = self._rendered.controls["n_samples"]
        self.sample_button = self._rendered.actions["generate_samples"]
        self.interactive_plot = self._rendered.root

        if auto_display:
            display(self._rendered.root)

    def _generate_samples_action(self, state, _controls):
        state["sample_seed"] = int(self._seed_generator.integers(0, np.iinfo(np.int32).max))
        return None

    def spec(self) -> InteractiveSpec:
        return InteractiveSpec(
            name="empirical_roc_visualizer",
            artifact_name="empirical-roc-visualizer",
            controls=(
                ControlSpec(
                    name="distribution",
                    kind="select",
                    label="Distribution",
                    default=self.distribution,
                    values=("Laplace", "Gaussian"),
                ),
                ControlSpec(
                    name="scale",
                    kind="slider",
                    label="Scale",
                    default=self.scale,
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    continuous=True,
                ),
                ControlSpec(
                    name="n_samples",
                    kind="slider",
                    label="Samples / class",
                    default=self.n_samples,
                    min=10,
                    max=max(5000, self.n_samples),
                    step=0.05,
                    continuous=True,
                    slider_scale="log",
                ),
            ),
            preferred_backend="ipywidgets",
            allowed_backends=("ipywidgets",),
            make_figure=partial(
                make_empirical_roc_figure,
                resolution=1000,
            ),
            figure_factory=("libdpy.visualization.roc_plots:make_empirical_roc_figure"),
            description=("Gaussian/Laplace theoretical and empirical ROC explorer."),
            actions=(
                ActionSpec(
                    name="generate_samples",
                    label="Generate Samples",
                    handler=self._generate_samples_action,
                    button_style="info",
                ),
            ),
            initial_state={"sample_seed": None},
        )

    @property
    def samples(self) -> tuple[np.ndarray, np.ndarray] | None:
        sample_seed = self._rendered.state["sample_seed"]
        if sample_seed is None:
            return None
        return _generate_empirical_roc_samples(
            self.distribution_control.value,
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
    ) -> go.Figure:
        """Return the current state as a Plotly figure."""

        return make_empirical_roc_figure(
            distribution or self.distribution,
            scale,
            n_samples or self.n_samples,
            self._rendered.state["sample_seed"],
        )

    def show(self):
        """Display the already-created widget without rebuilding its state."""

        display(self._rendered.root)
        return self._rendered.root

import math
from enum import Enum

from IPython.core.display_functions import display
from ipywidgets import FloatSlider, Button, interactive, VBox, HBox
from matplotlib import pyplot as plt

from scipy.stats import rv_continuous as Distribution, norm, laplace, uniform
import numpy as np
from typing import Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve


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


def ROC_and_distributions_visualization(
    desicion_rule_params: Callable,
    dist0: Distribution,
    dist1: Distribution,
    by_threshold: bool,
    res: int = 100,
):
    # set the location of the two subplot titles to the center
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distributions PDF", "ROC Curve"),
        column_widths=[0.45, 0.45],
        horizontal_spacing=0.1,
    )

    # Draw the two distributions in the first subplot
    draw_two_distributions(dist0, dist1, fig, res)
    # Draw the ROC curve and a diagonal line in the second subplot
    draw_ROCs_and_diag_from_desicion_rule(
        desicion_rule_params, [[dist0, dist1]], fig, by_threshold, 2, res
    )
    # Set params for the decision rule
    if by_threshold:
        param = (dist0.mean() + dist1.mean()) / 2
        param_type = ComparisonType.GENERAL
        param_title = "Threshold"
        param_range = range_from_distribution(dist0, dist1, res)
    else:
        param = 0.5
        param_type = check_comparison_type(dist0, dist1)
        param_title = "FPR"
        param_range = np.linspace(0.0001, 0.9999, res)
    # Add initial ROC marker and threshold line
    fig.update_layout(
        shapes=create_ROC_point_and_thresholds(
            desicion_rule_params, dist0, dist1, param, param_type, res
        )
    )
    # color the area to the left of the threshold in light blue and the area to the right in light red

    # Add slider steps for interactivity
    steps = [
        dict(
            method="relayout",
            args=[
                dict(
                    shapes=create_ROC_point_and_thresholds(
                        desicion_rule_params, dist0, dist1, param, param_type, res
                    )
                )
            ],
            label=f"{param:.2f}",
            execute=True,
        )
        for param in param_range
    ]
    sliders = [
        dict(
            active=param,
            currentvalue=dict(prefix=param_title, font={'size': 14}),
            pad={"t": 40},
            steps=steps,
        )
    ]

    # Update layout
    fig.update_layout(
        height=500,
        autosize=True,
        # Reserve top margin so the title sits above the subplots instead of
        # overlapping them when the figure is scaled to fit a slide.
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
        yaxis=dict(
            title=dict(text="Probability Density"),
        ),
        xaxis2=dict(title=dict(text="False Positive Rate (FPR)"), domain=[0.55, 1.0]),
        yaxis2=dict(title=dict(text="True Positive Rate (TPR)"), domain=[0, 1]),
        sliders=sliders,
    )
    return fig


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


class EmpROCVisualizer:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.scale_slider = FloatSlider(
            min=0.1, max=2.0, step=0.1, value=1.0, description='σ:', continuous_update=False
        )
        self.sample_button = Button(description='Generate Samples', button_style='info')
        self.sample_button.on_click(self.on_button_clicked)
        self.samples = None

        # Create the interactive plot
        self.interactive_plot = interactive(self.plot_curves, scale=self.scale_slider)

        # Display widgets
        display(
            VBox(
                [HBox([self.scale_slider, self.sample_button]), self.interactive_plot.children[-1]]
            )
        )

    def generate_samples(self, scale):
        negative_samples = np.random.laplace(0, scale, self.n_samples)
        positive_samples = np.random.laplace(1, scale, self.n_samples)
        labels = np.concatenate([np.zeros(self.n_samples), np.ones(self.n_samples)])
        samples = np.concatenate([negative_samples, positive_samples])
        return samples, labels

    def on_button_clicked(self, b):
        self.samples = self.generate_samples(self.scale_slider.value)
        self.interactive_plot.update()

    def plot_curves(self, scale):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Generate x values for theoretical distributions
        x = np.linspace(-4, 5, 1000)
        pdf_negative = laplace.pdf(x, loc=0, scale=scale)
        pdf_positive = laplace.pdf(x, loc=1, scale=scale)

        # Plot theoretical distributions
        ax1.plot(x, pdf_negative, 'b-', label='Theoretical Negative (μ=0)', alpha=0.7)
        ax1.plot(x, pdf_positive, 'r-', label='Theoretical Positive (μ=1)', alpha=0.7)

        # Plot empirical distributions if samples exist
        if self.samples is not None:
            samples, labels = self.samples

            # Create histograms
            ax1.hist(
                samples[labels == 0],
                bins=30,
                density=True,
                alpha=0.5,
                color='blue',
                label='Empirical Negative',
            )
            ax1.hist(
                samples[labels == 1],
                bins=30,
                density=True,
                alpha=0.5,
                color='red',
                label='Empirical Positive',
            )

        ax1.set_title('Probability Density Functions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.legend()
        ax1.grid(True)

        # Calculate theoretical ROC curve
        thresholds = np.linspace(-4, 5, 1000)
        tpr_theo = []
        fpr_theo = []

        for threshold in thresholds:
            tp = 1 - laplace.cdf(threshold, loc=1, scale=scale)
            fp = 1 - laplace.cdf(threshold, loc=0, scale=scale)
            tpr_theo.append(tp)
            fpr_theo.append(fp)

        # Calculate theoretical AUC
        roc_auc_theo = auc(fpr_theo, tpr_theo)

        # Plot theoretical ROC curve
        ax2.plot(
            fpr_theo, tpr_theo, 'b-', label=f'Theoretical ROC (AUC = {roc_auc_theo:.3f})', alpha=0.7
        )

        # Plot empirical ROC curve if samples exist
        if self.samples is not None:
            samples, labels = self.samples
            fpr_emp, tpr_emp, _ = roc_curve(labels, samples)
            roc_auc_emp = auc(fpr_emp, tpr_emp)

            ax2.plot(fpr_emp, tpr_emp, 'g--', label=f'Empirical ROC (AUC = {roc_auc_emp:.3f})')

        # Add random classifier line
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

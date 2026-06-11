"""Privacy-specific visualization utilities."""

from typing import Tuple
import numpy as np

import plotly.graph_objects as go
from ipywidgets import FloatSlider, VBox, HBox

import matplotlib.pyplot as plt


def calc_privacy_bound(log_slope: float, shift: float, res) -> Tuple[np.ndarray, np.ndarray]:
    slope = np.exp(log_slope)
    x_1 = np.linspace(0, (1 - shift) / (1 + slope), np.floor(res / 2).astype(int))
    y_1 = slope * x_1 + shift
    x_2 = np.linspace((1 - shift) / (1 + slope), 1 - shift, np.ceil(res / 2).astype(int))
    y_2 = 1 / slope * x_2 + 1 - (1 - shift) / slope
    x = np.concatenate([x_1, x_2])
    y = np.concatenate([y_1, y_2])
    return x, y


def create_distribution(dist_type, mean, std):
    return dist_type(loc=mean, scale=std)


def draw_ROCs_and_diag_from_distributions(
    distributions, fig, show_diagonal=True, diagonal_offset=0, res=100
):
    colors = ['red', 'green', 'orange', 'purple', 'brown']

    for i, (dist0, dist1) in enumerate(distributions):
        # Calculate ROC curve points
        thresholds = np.linspace(dist0.ppf(0.001), dist0.ppf(0.999), res)
        fpr = 1 - dist0.cdf(thresholds)
        tpr = 1 - dist1.cdf(thresholds)

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f'ROC {i+1}', line=dict(color=color, width=2)
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
                mode='lines',
                name='Diagonal',
                line=dict(color='gray', width=1, dash='dash'),
            )
        )


# Used by lecture_3
class PrivacyPlot:
    def __init__(self, distribution_types: list, sensitivity: float, std: float, res: int = 100):

        # Initialize parameters
        self.log_slope = np.log(1.5)
        self.shift = 0.0
        self.res = res
        self.fig = go.FigureWidget()

        # Add privacy bound
        x, y = calc_privacy_bound(self.log_slope, self.shift, res)
        self.fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='lines', name='Privacy bound', line=dict(color='blue', width=4)
            )
        )

        # Add ROC curves of the distributions
        distributions = []
        for dist_type in distribution_types:
            dist0 = create_distribution(dist_type, 0, std)
            dist1 = create_distribution(dist_type, sensitivity, std)
            distributions.append([dist0, dist1])
        draw_ROCs_and_diag_from_distributions(distributions, self.fig, False, 0, res)

        # Create sliders for log_slope and shift at the bottom of the plot
        self.log_slope_slider = FloatSlider(
            value=self.log_slope, min=0, max=np.log(30), step=0.05, description='log(Slope)'
        )
        self.shift_slider = FloatSlider(
            value=self.shift, min=0, max=1, step=0.01, description='Shift'
        )

        # Set up callbacks
        self.log_slope_slider.observe(self.update_plot, names='value')
        self.shift_slider.observe(self.update_plot, names='value')

        # Create figure
        self.fig.update_layout(
            title={
                'text': f"ROC for distributions with std / sensitivity = {std/sensitivity :.2f}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            },
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600,
        )

        # Create the widget display
        self.widget = VBox([self.fig, HBox([self.log_slope_slider, self.shift_slider])])

    def update_plot(self, change):
        self.log_slope = self.log_slope_slider.value
        self.shift = self.shift_slider.value

        # Update the privacy bound trace
        x, y = calc_privacy_bound(self.log_slope, self.shift, self.res)
        with self.fig.batch_update():
            self.fig.data[0].x = x
            self.fig.data[0].y = y

    def show(self):
        return self.widget


def plot_epsilon_delta_tradeoff(
    epsilons, deltas, mechanism_names=None, title="Privacy Budget Trade-off"
):
    plt.figure(figsize=(10, 6))

    if mechanism_names is None:
        mechanism_names = [f'Mechanism {i+1}' for i in range(len(epsilons))]

    for i, (eps_list, delta_list, name) in enumerate(zip(epsilons, deltas, mechanism_names)):
        plt.plot(eps_list, delta_list, 'o-', label=name, linewidth=2, markersize=6)

    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Failure Probability (δ)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


def plot_privacy_loss_distribution(privacy_losses, epsilon, title="Privacy Loss Distribution"):
    plt.figure(figsize=(10, 6))
    plt.hist(privacy_losses, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')

    # Add vertical line at epsilon
    plt.axvline(epsilon, color='red', linestyle='--', linewidth=2, label=f'ε = {epsilon}')
    plt.axvline(-epsilon, color='red', linestyle='--', linewidth=2, label=f'-ε = {-epsilon}')

    plt.xlabel('Privacy Loss')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_roc_curves(fpr_list, tpr_list, labels=None, title="ROC Curves Comparison"):
    plt.figure(figsize=(8, 8))

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(fpr_list))]

    for fpr, tpr, label in zip(fpr_list, tpr_list, labels):
        auc = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {auc:.3f})')

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def plot_privacy_accounting(epsilons, steps, title="Privacy Budget Consumption"):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, epsilons, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Training Steps')
    plt.ylabel('Cumulative Privacy Budget (ε)')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add horizontal lines for common privacy budgets
    privacy_levels = [0.1, 1.0, 10.0]
    colors = ['green', 'orange', 'red']
    labels = ['Strong Privacy', 'Moderate Privacy', 'Weak Privacy']

    for level, color, label in zip(privacy_levels, colors, labels):
        if level <= max(epsilons):
            plt.axhline(level, color=color, linestyle='--', alpha=0.7, label=f'{label} (ε={level})')

    plt.legend()
    plt.show()

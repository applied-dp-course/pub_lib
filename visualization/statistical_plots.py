"""Statistical visualization utilities."""

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, norm

from .interactive import AbstractInteractivePlot, ControlSpec, InteractiveSpec


def make_histogram_figure(data, title="", x_label="", show_kde=False, bins=50):
    """Build a histogram figure without displaying it."""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        data,
        bins=bins,
        density=True,
        color='b',
        edgecolor='black',
        label=f'Mean: {np.mean(data)}\nStandard deviation: {np.std(data)}',
    )
    if show_kde:
        x = np.linspace(min(data), max(data), 1000)
        ax.plot(x, norm.pdf(x, 0, 1), 'k-', label='Theoretical Standard N(0,1)')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def generate_normalized_samples(
    num_samples,
    lower_bound,
    upper_bound,
    normalization_func,
    *,
    sample_size=10000,
    seed=None,
):
    rng = np.random.default_rng(seed)
    samples = rng.uniform(lower_bound, upper_bound, size=(num_samples, sample_size))
    mean = (upper_bound + lower_bound) / 2
    std_dev = np.sqrt(((upper_bound - lower_bound) ** 2) / 12)
    return normalization_func(samples, mean, std_dev)


def make_clt_figure(
    sample_size,
    *,
    normalization_func,
    lower_bound=-3,
    upper_bound=8,
    experiments=10_000,
    seed=0,
):
    """Build one reproducible central-limit-theorem state."""

    sample_size = int(sample_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if experiments < 2:
        raise ValueError("experiments must be at least 2")
    normalized_samples = generate_normalized_samples(
        experiments,
        lower_bound,
        upper_bound,
        normalization_func,
        sample_size=sample_size,
        seed=seed,
    )
    flattened = np.asarray(normalized_samples, dtype=float).reshape(-1)
    if not np.all(np.isfinite(flattened)):
        raise ValueError("normalization_func returned non-finite values")

    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=flattened,
            histnorm='probability density',
            name=f'Mean: {np.mean(flattened):.2f}, Std: {np.std(flattened):.2f}',
            opacity=0.75,
        )
    )
    kde = gaussian_kde(flattened)
    x_kde = np.linspace(
        float(np.min(flattened)) - 0.001,
        float(np.max(flattened)) + 0.001,
        500,
    )
    y_kde = kde(x_kde)
    figure.add_trace(
        go.Scatter(x=x_kde, y=y_kde, mode='lines', line=dict(color='red'), name='KDE Gaussian')
    )
    figure.update_layout(
        title=f"Distribution of Normalized Sums (sample size={sample_size})",
        xaxis_title="Sample Mean",
        yaxis_title="Density",
        showlegend=True,
        bargap=0.05,
    )
    return figure


def clt_plot_spec(
    normalization_func,
    *,
    lower_bound=-3,
    upper_bound=8,
    experiments=10_000,
    seed=0,
):
    return InteractiveSpec(
        name="central_limit_theorem",
        artifact_name="central-limit-theorem",
        controls=(
            ControlSpec(
                name="sample_size",
                kind="slider",
                label="Sample size",
                default=100,
                min=10,
                max=1000,
                step=10,
                continuous=False,
            ),
        ),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets",),
        make_figure=make_clt_figure,
        figure_factory="libdpy.visualization.statistical_plots:make_clt_figure",
        fixed_kwargs={
            "normalization_func": normalization_func,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "experiments": experiments,
            "seed": seed,
        },
    )


def make_normal_distribution_comparison_figure(
    data, theoretical_mean=0, theoretical_std=1, title="Distribution Comparison"
):
    """Build a normal-distribution comparison figure without displaying it."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        data,
        bins=50,
        density=True,
        alpha=0.7,
        color='lightblue',
        edgecolor='black',
        label=f'Empirical Data\nMean: {np.mean(data):.3f}\nStd: {np.std(data):.3f}',
    )
    x = np.linspace(
        min(data.min(), theoretical_mean - 4 * theoretical_std),
        max(data.max(), theoretical_mean + 4 * theoretical_std),
        1000,
    )
    theoretical_pdf = norm.pdf(x, theoretical_mean, theoretical_std)
    ax.plot(
        x,
        theoretical_pdf,
        'r-',
        linewidth=2,
        label=f'Theoretical N({theoretical_mean},{theoretical_std}²)',
    )
    kde = gaussian_kde(data)
    ax.plot(x, kde(x), 'g--', linewidth=2, label='Empirical KDE')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def make_convergence_to_normal_figure(
    sample_sizes, lower_bound=-3, upper_bound=8, normalization_func=None
):
    """Build a CLT convergence grid figure without displaying it."""

    if normalization_func is None:

        def default_normalization(sums, mean, std_dev, n):
            return (sums - n * mean) / (std_dev * np.sqrt(n))

        normalization_func = default_normalization

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, n in enumerate(sample_sizes[:4]):
        normalized_samples = generate_normalized_samples(
            n, lower_bound, upper_bound, normalization_func
        )
        axes[i].hist(
            normalized_samples,
            bins=30,
            density=True,
            alpha=0.7,
            color='lightblue',
            edgecolor='black',
        )
        x = np.linspace(-4, 4, 100)
        axes[i].plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
        axes[i].set_title(f'Sample Size: {n}')
        axes[i].set_xlabel('Normalized Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.suptitle('Central Limit Theorem: Convergence to Normal Distribution', y=1.02)
    return fig


def make_sample_means_distribution_figure(
    num_experiments=1000, sample_size=30, population_dist='uniform', seed=None
):
    """Build a sample-means distribution figure without displaying it."""

    rng = np.random.default_rng(seed)
    sample_means = []

    for _ in range(num_experiments):
        if population_dist == 'uniform':
            sample = rng.uniform(-3, 8, sample_size)
        elif population_dist == 'exponential':
            sample = rng.exponential(2, sample_size)
        elif population_dist == 'binomial':
            sample = rng.binomial(10, 0.3, sample_size)
        else:
            sample = rng.normal(0, 1, sample_size)

        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        sample_means,
        bins=50,
        density=True,
        alpha=0.7,
        color='lightgreen',
        edgecolor='black',
        label=f'Sample Means\nMean: {np.mean(sample_means):.3f}\nStd: {np.std(sample_means):.3f}',
    )

    if population_dist == 'uniform':
        pop_mean = 2.5
        pop_std = np.sqrt(((8 - (-3)) ** 2) / 12)
    elif population_dist == 'exponential':
        pop_mean = 2
        pop_std = 2
    elif population_dist == 'binomial':
        pop_mean = 10 * 0.3
        pop_std = np.sqrt(10 * 0.3 * 0.7)
    else:
        pop_mean = 0
        pop_std = 1

    theoretical_mean = pop_mean
    theoretical_std = pop_std / np.sqrt(sample_size)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    theoretical_pdf = norm.pdf(x, theoretical_mean, theoretical_std)
    ax.plot(
        x,
        theoretical_pdf,
        'r-',
        linewidth=2,
        label=f'Theoretical N({theoretical_mean:.2f},{theoretical_std:.3f}²)',
    )

    ax.set_title(
        f'Distribution of Sample Means\n{population_dist.capitalize()} Population, Sample Size: {sample_size}'
    )
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def make_laplace_distributions_figure(b, loc1, loc2):
    """Build a Matplotlib Laplace comparison figure without displaying it."""

    x = np.linspace(50, 60, 1000)

    def laplace_pdf(x_vals, loc, scale):
        return 1 / (2 * scale) * np.exp(-np.abs(x_vals - loc) / scale)

    y1 = laplace_pdf(x, loc1, b)
    y2 = laplace_pdf(x, loc2, b)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y1, label='Dist 1', color='blue')
    ax.plot(x, y2, label='Dist 2', color='red')
    ax.set_title('Laplace Distributions')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_ylim(0, 2)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


def make_laplace_comparison_figure(
    scale,
    *,
    loc1=55,
    loc2=56,
    x_range=(50, 60),
):
    """Build two Laplace densities for a concrete scale."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    x = np.linspace(x_range[0], x_range[1], 1000)
    figure = go.Figure(
        [
            go.Scatter(
                x=x,
                y=1 / (2 * scale) * np.exp(-np.abs(x - loc1) / scale),
                mode="lines",
                name="Dist 1",
                line={"color": "blue"},
            ),
            go.Scatter(
                x=x,
                y=1 / (2 * scale) * np.exp(-np.abs(x - loc2) / scale),
                mode="lines",
                name="Dist 2",
                line={"color": "red"},
            ),
        ]
    )
    figure.update_layout(
        title="Laplace Distributions",
        xaxis_title="x",
        yaxis_title="Density",
        yaxis_range=[0, 2],
        width=600,
        height=400,
    )
    return figure


def laplace_comparison_spec(loc1=55, loc2=56):
    return InteractiveSpec(
        name="laplace_scale",
        artifact_name="laplace-scale",
        controls=(
            ControlSpec(
                name="scale",
                kind="slider",
                label="Scale (b)",
                default=1.0,
                min=0.01,
                max=4.0,
                step=0.01,
            ),
        ),
        preferred_backend="wasm-marimo",
        allowed_backends=("ipywidgets", "wasm-marimo"),
        make_figure=make_laplace_comparison_figure,
        figure_factory=("libdpy.visualization.statistical_plots:make_laplace_comparison_figure"),
        fixed_kwargs={"loc1": loc1, "loc2": loc2},
    )


class LaplaceComparison(AbstractInteractivePlot):
    """Interactive Laplace scale comparison for notebooks and the static site."""

    def __init__(self, loc1: float = 55, loc2: float = 56):
        self.loc1 = float(loc1)
        self.loc2 = float(loc2)

    def spec(self) -> InteractiveSpec:
        return laplace_comparison_spec(self.loc1, self.loc2)

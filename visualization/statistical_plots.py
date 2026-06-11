"""Statistical visualization utilities."""

import numpy as np
from IPython.core.display_functions import display
from ipywidgets import interactive, FloatSlider
from matplotlib import pyplot as plt

import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm
import ipywidgets as widgets


# Used by notebook0
def plot_histogram(data, title="", x_label="", show_kde=False, bins=50):
    plt.figure(figsize=(8, 6))
    plt.hist(
        data,
        bins=bins,
        density=True,
        color='b',
        edgecolor='black',
        label=f'Mean: {np.mean(data)}\nStandard deviation: {np.std(data)}',
    )
    if show_kde:
        x = np.linspace(min(data), max(data), 1000)
        plt.plot(x, norm.pdf(x, 0, 1), 'k-', label='Theoretical Standard N(0,1)')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def generate_normalized_samples(num_samples, lower_bound, upper_bound, normalization_func):
    samples = np.random.uniform(lower_bound, upper_bound, size=(num_samples, 10000))
    mean = (upper_bound + lower_bound) / 2
    std_dev = np.sqrt(((upper_bound - lower_bound) ** 2) / 12)
    return normalization_func(samples, mean, std_dev)


# Create the CLT plot with slider functionality
# Used by notebook 0
def plot_CLT_with_sample_slider(normalization_func):
    lower_bound, upper_bound = -3, 8
    initial_sample_size = 100
    normalized_samples = generate_normalized_samples(
        initial_sample_size, lower_bound, upper_bound, normalization_func
    )

    # Initialize the figure
    fig = go.Figure()

    # Add initial histogram
    fig.add_trace(
        go.Histogram(
            x=normalized_samples.flatten(),
            histnorm='probability density',
            name=f'Mean: {np.mean(normalized_samples):.2f}, Std: {np.std(normalized_samples):.2f}',
            opacity=0.75,
        )
    )

    # Add initial KDE curve
    kde = gaussian_kde(normalized_samples.flatten())
    x_kde = np.linspace(
        min(normalized_samples.flatten()) - 0.001, max(normalized_samples.flatten()) + 0.001, 1000
    )
    y_kde = kde(x_kde)

    fig.add_trace(
        go.Scatter(x=x_kde, y=y_kde, mode='lines', line=dict(color='red'), name='KDE Gaussian')
    )

    # Create steps for the slider
    steps = []
    for sample_size in range(10, 1001, 50):
        updated_samples = generate_normalized_samples(
            sample_size, lower_bound, upper_bound, normalization_func
        )
        kde = gaussian_kde(updated_samples.flatten())
        x_kde = np.linspace(min(updated_samples.flatten()), max(updated_samples.flatten()), 1000)
        y_kde = kde(x_kde)

        step = dict(
            method="update",
            args=[
                {
                    'x': [updated_samples.flatten(), x_kde],
                    'y': [None, y_kde],
                    'name': [
                        f'Mean: {updated_samples.mean():.3f}, Std: {updated_samples.std():.3f}',
                        'KDE Gaussian',
                    ],
                }
            ],
            label=f'{sample_size}',
        )
        steps.append(step)

    # Add slider to the plot
    sliders = [
        dict(active=0, currentvalue={"prefix": "Number of Samples: "}, pad={"t": 50}, steps=steps)
    ]

    # Configure layout
    fig.update_layout(
        sliders=sliders,
        title="Distribution of Sample Means with KDE",
        xaxis_title="Sample Mean",
        yaxis_title="Density",
        showlegend=True,
        bargap=0.05,
    )

    fig.show()


def plot_normal_distribution_comparison(
    data, theoretical_mean=0, theoretical_std=1, title="Distribution Comparison"
):
    plt.figure(figsize=(10, 6))

    # Plot histogram of data
    plt.hist(
        data,
        bins=50,
        density=True,
        alpha=0.7,
        color='lightblue',
        edgecolor='black',
        label=f'Empirical Data\nMean: {np.mean(data):.3f}\nStd: {np.std(data):.3f}',
    )

    # Plot theoretical normal distribution
    x = np.linspace(
        min(data.min(), theoretical_mean - 4 * theoretical_std),
        max(data.max(), theoretical_mean + 4 * theoretical_std),
        1000,
    )
    theoretical_pdf = norm.pdf(x, theoretical_mean, theoretical_std)
    plt.plot(
        x,
        theoretical_pdf,
        'r-',
        linewidth=2,
        label=f'Theoretical N({theoretical_mean},{theoretical_std}²)',
    )

    # Plot KDE of empirical data
    kde = gaussian_kde(data)
    plt.plot(x, kde(x), 'g--', linewidth=2, label='Empirical KDE')

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_convergence_to_normal(
    sample_sizes, lower_bound=-3, upper_bound=8, normalization_func=None
):
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

        # Overlay theoretical normal
        x = np.linspace(-4, 4, 100)
        axes[i].plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')

        axes[i].set_title(f'Sample Size: {n}')
        axes[i].set_xlabel('Normalized Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Central Limit Theorem: Convergence to Normal Distribution', y=1.02)
    plt.show()


def plot_sample_means_distribution(num_experiments=1000, sample_size=30, population_dist='uniform'):
    sample_means = []

    for _ in range(num_experiments):
        if population_dist == 'uniform':
            sample = np.random.uniform(-3, 8, sample_size)
        elif population_dist == 'exponential':
            sample = np.random.exponential(2, sample_size)
        elif population_dist == 'binomial':
            sample = np.random.binomial(10, 0.3, sample_size)
        else:
            sample = np.random.normal(0, 1, sample_size)

        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    plt.figure(figsize=(10, 6))
    plt.hist(
        sample_means,
        bins=50,
        density=True,
        alpha=0.7,
        color='lightgreen',
        edgecolor='black',
        label=f'Sample Means\nMean: {np.mean(sample_means):.3f}\nStd: {np.std(sample_means):.3f}',
    )

    # Theoretical distribution of sample means
    if population_dist == 'uniform':
        pop_mean = 2.5  # (-3 + 8) / 2
        pop_std = np.sqrt(((8 - (-3)) ** 2) / 12)  # Uniform distribution std
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
        plt.plot(
            x,
            theoretical_pdf,
            'r-',
            linewidth=2,
            label=f'Theoretical N({theoretical_mean:.2f},{theoretical_std:.3f}²)',
        )

    plt.title(
        f'Distribution of Sample Means\n{population_dist.capitalize()} Population, Sample Size: {sample_size}'
    )
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_laplace_distributions(b, loc1, loc2):
    # Generate x values
    x = np.linspace(50, 60, 1000)

    # Calculate Laplace distributions
    def laplace_pdf(x, loc, b):
        return 1 / (2 * b) * np.exp(-np.abs(x - loc) / b)

    y1 = laplace_pdf(x, loc1, b)
    y2 = laplace_pdf(x, loc2, b)

    # Create plot with fixed y-axis
    plt.figure(figsize=(6, 4))
    plt.plot(x, y1, label=f'Dist 1', color='blue')
    plt.plot(x, y2, label=f'Dist 2', color='red')

    plt.title('Laplace Distributions')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.ylim(0, 2)  # Fix y-axis from 0 to 1
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def create_laplace_interactive(loc1=55, loc2=56):
    interactive_plot = interactive(
        plot_laplace_distributions,
        b=FloatSlider(min=0.01, max=4, step=0.01, value=1, description='Scale (b)'),
        loc1=loc1,
        loc2=loc2,
    )
    display(interactive_plot)

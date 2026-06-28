from typing import Any, List

import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from libdpy.ml.models import Model


def make_accuracy_epsilon_noise_comparison_figure(
    accuracy_array, epsilon_array, noise_scale_array, title
):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    axs[0].plot(noise_scale_array, accuracy_array)
    axs[0].set_title('Accuracy vs Noise')
    axs[0].set_xlabel('Noise scale')
    axs[0].set_ylabel('Accuracy')
    axs[1].plot(noise_scale_array, epsilon_array)
    axs[1].set_title('Epsilon vs Noise')
    axs[1].set_xlabel('Noise scale')
    axs[1].set_ylabel('Epsilon')
    axs[2].plot(epsilon_array, accuracy_array)
    axs[2].set_title('Accuracy vs Epsilon')
    axs[2].set_xlabel('Epsilon')
    axs[2].set_ylabel('Accuracy')
    return fig


def plot_accuracy_epsilon_noise_comparison(accuracy_array, epsilon_array, noise_scale_array, title):
    fig = make_accuracy_epsilon_noise_comparison_figure(
        accuracy_array, epsilon_array, noise_scale_array, title
    )
    display(fig)
    return fig


def make_weights_visualization_figure(
    models: List[Model],
    noise_scale_array: np.ndarray[Any, np.dtype[np.float64]],
    epsilon_array: np.ndarray[Any, np.dtype[np.float64]],
    accuracy_array: np.ndarray[Any, np.dtype[np.float64]],
    title: str,
):
    num_models = len(models)
    num_rows = num_models // 3
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    fig.suptitle(title)
    for i, model in enumerate(models[: num_rows * 3]):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        ax.set_title(
            f"noise: {noise_scale_array[i]:.2f}, epsilon: {epsilon_array[i]:.3f}, accuracy: {accuracy_array[i]:.2f}"
        )
        predictor = model.get_final_weights()
        new_size = int(np.sqrt(predictor.shape[0]))
        predictor = predictor[: new_size**2].reshape(new_size, new_size)
        ax.imshow(predictor, cmap='gray')
        ax.axis('off')
    return fig


def plot_weights_visualization(
    models: List[Model],
    noise_scale_array: np.ndarray[Any, np.dtype[np.float64]],
    epsilon_array: np.ndarray[Any, np.dtype[np.float64]],
    accuracy_array: np.ndarray[Any, np.dtype[np.float64]],
    title: str,
):
    fig = make_weights_visualization_figure(
        models, noise_scale_array, epsilon_array, accuracy_array, title
    )
    display(fig)
    return fig


def make_average_category_weights_figure(category_means, noise_scale_array, title):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(noise_scale_array, category_means)
    ax[0].set_title(title)
    ax[0].set_xlabel('Noise scale')
    ax[0].set_ylabel('Average weight')
    ax[1].plot(noise_scale_array, np.abs(category_means[:, 2] - category_means[:, 0]))
    ax[1].set_title('Difference between category 1 and -1')
    ax[1].set_xlabel('Noise scale')
    ax[1].set_ylabel('Average weight difference')
    return fig


def plot_average_category_weights(category_means, noise_scale_array, title):
    fig = make_average_category_weights_figure(category_means, noise_scale_array, title)
    display(fig)
    return fig


def make_roc_multiple_noise_factors_figure(
    positive_samples_list, negative_samples_list, noise_factor_vec, title
):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(noise_factor_vec)))

    for pos_samples, neg_samples, noise_factor, color in zip(
        positive_samples_list, negative_samples_list, noise_factor_vec, colors
    ):
        labels = np.concatenate([np.zeros(len(neg_samples)), np.ones(len(pos_samples))])
        samples = np.concatenate([neg_samples, pos_samples])
        fpr_emp, tpr_emp, _ = roc_curve(labels, samples)
        ax.plot(fpr_emp, tpr_emp, color=color, linestyle='--', label=f'ROC (σ={noise_factor:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_ROC_multiple_noise_factors(
    positive_samples_list, negative_samples_list, noise_factor_vec, title
):
    fig = make_roc_multiple_noise_factors_figure(
        positive_samples_list, negative_samples_list, noise_factor_vec, title
    )
    display(fig)
    return fig

from typing import Dict

import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve


def make_weight_figure(predictor, title="", cmap='gray'):
    new_size = int(np.sqrt(predictor.shape[0]))
    predictor = predictor[: new_size**2].reshape(new_size, new_size)
    fig, ax = plt.subplots()
    ax.imshow(predictor, cmap=cmap)
    fig.colorbar(ax.images[0], ax=ax)
    ax.set_title(f"Weight Visualization {title}")
    return fig


def plot_weight(predictor, title="", cmap='gray'):
    fig = make_weight_figure(predictor, title=title, cmap=cmap)
    display(fig)
    return fig


def plot_confusion_matrix(confusion_matrix, ax, accuracy, classes, prefix):
    im = ax.imshow(confusion_matrix, cmap='cool')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{confusion_matrix[i, j]:.3f}', ha='center', va='center', color='black')
    ax.set_title(f'{prefix}: accuracy = {accuracy:.3f}')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')


def make_confusion_matrixes_figure(confusion_matrix_train, confusion_matrix_test, classes):
    train_accuracy = np.trace(confusion_matrix_train) / len(classes)
    test_accuracy = np.trace(confusion_matrix_test) / len(classes)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    fig.suptitle('Confusion Matrixes')
    plot_confusion_matrix(confusion_matrix_train, ax[0], train_accuracy, classes, 'Train')
    plot_confusion_matrix(confusion_matrix_test, ax[1], test_accuracy, classes, 'Test')
    return fig


def plot_confusion_matrixes(confusion_matrix_train, confusion_matrix_test, classes):
    fig = make_confusion_matrixes_figure(confusion_matrix_train, confusion_matrix_test, classes)
    display(fig)
    return fig


def make_binned_classification_grid_figure(
    data_by_class_and_bin, num_samples, probability_bins=[0, 0.5, 0.75, 0.9, 1], seed=None
):
    rng = np.random.default_rng(seed)
    num_classes = len(data_by_class_and_bin)
    num_bins = len(probability_bins) - 1
    data_dim = int(np.sqrt(data_by_class_and_bin[0][0].shape[1]))

    fig = plt.figure(figsize=(3 * num_bins, 3 * num_classes * num_samples))
    fig.suptitle("Representative samples by prediction confidence", fontsize=20, y=0.98)
    gs = plt.GridSpec(num_classes * num_samples, num_bins)
    gs.update(wspace=0.05, hspace=0.05)
    axs = np.empty((num_classes * num_samples, num_bins), dtype=object)

    for class_idx in range(num_classes):
        for bin_idx in range(num_bins):
            bin_data = data_by_class_and_bin[class_idx][bin_idx]
            for sample_idx in range(num_samples):
                row_idx = class_idx * num_samples + sample_idx
                ax = fig.add_subplot(gs[row_idx, bin_idx])
                axs[row_idx, bin_idx] = ax

                if len(bin_data) > 0:
                    indices = list(range(len(bin_data)))
                    rng.shuffle(indices)
                    selected_indices = indices[:num_samples]
                    selected_samples = bin_data[selected_indices, : data_dim**2]

                    if len(bin_data) > sample_idx:
                        ax.imshow(
                            selected_samples[sample_idx].reshape(data_dim, data_dim), cmap='gray'
                        )
                ax.axis('off')

    for bin_idx in range(num_bins):
        axs[0, bin_idx].set_title(
            f'[{probability_bins[bin_idx]:.2f},\n{probability_bins[bin_idx+1]:.2f})', pad=10
        )

    for class_idx in range(num_classes):
        row_idx = class_idx * num_samples
        fig.text(
            0.02,
            1 - (row_idx + num_samples / 2) / (num_classes * num_samples),
            f'Class {class_idx}',
            rotation=90,
            verticalalignment='center',
            fontsize=16,
        )

    for class_idx in range(1, num_classes):
        y_coord = 0.975 - class_idx * num_samples / (num_classes * num_samples)
        fig.add_artist(
            plt.Line2D(
                [0.05, 0.95],
                [y_coord, y_coord],
                color='blue',
                linestyle='--',
                linewidth=4,
                transform=fig.transFigure,
            )
        )

    for bin_idx in range(1, num_bins):
        x_coord = 0.045 + bin_idx * 0.91 / num_bins
        fig.add_artist(
            plt.Line2D(
                [x_coord, x_coord],
                [0.05, 0.95],
                color='blue',
                linestyle='--',
                linewidth=4,
                transform=fig.transFigure,
            )
        )

    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    return fig


def plot_binned_classification_grid(
    data_by_class_and_bin, num_samples, probability_bins=[0, 0.5, 0.75, 0.9, 1], seed=None
):
    fig = make_binned_classification_grid_figure(
        data_by_class_and_bin, num_samples, probability_bins=probability_bins, seed=seed
    )
    display(fig)
    return fig


def create_dual_histogram(v1, v2, num_bins):
    v_combined = np.concatenate([v1, v2])
    p05 = np.percentile(v_combined, 5)
    p95 = np.percentile(v_combined, 95)
    bin_edges = np.linspace(p05, p95, num_bins + 1)
    return bin_edges


def make_distributions_and_roc_figure(
    positive_samples, negative_samples, title, n_bins=None
):
    if n_bins is None:
        n_bins = int(np.sqrt(len(positive_samples) + len(negative_samples)))

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    bin_edges = create_dual_histogram(positive_samples, negative_samples, n_bins)
    ax1.hist(
        positive_samples,
        bins=bin_edges,
        density=True,
        alpha=0.5,
        color='blue',
        label='Empirical Positive',
    )
    ax1.hist(
        negative_samples,
        bins=bin_edges,
        density=True,
        alpha=0.5,
        color='red',
        label='Empirical Negative',
    )
    ax1.set_title('Probability Density Functions')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True)

    labels = np.concatenate([np.zeros(len(negative_samples)), np.ones(len(positive_samples))])
    samples = np.concatenate([negative_samples, positive_samples])
    fpr_emp, tpr_emp, _ = roc_curve(labels, samples)
    ax2.plot(fpr_emp, tpr_emp, 'g--', label='Empirical ROC')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_distributions_and_ROC(positive_samples, negative_samples, title, n_bins=None):
    fig = make_distributions_and_roc_figure(
        positive_samples, negative_samples, title, n_bins=n_bins
    )
    display(fig)
    return fig

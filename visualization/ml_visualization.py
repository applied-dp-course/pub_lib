from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve


def plot_weight(predictor, title="", cmap='gray'):
    new_size = int(np.sqrt(predictor.shape[0]))
    predictor = predictor[: new_size**2].reshape(new_size, new_size)
    plt.imshow(predictor, cmap=cmap)
    plt.colorbar()
    plt.title(f"Weight Visualization " + title)
    plt.show()


def plot_confusion_matrix(confusion_matrix, ax, accuracy, classes, prefix):
    im = ax.imshow(confusion_matrix, cmap='cool')
    # print the confusion matrix values in white
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


def plot_confusion_matrixes(confusion_matrix_train, confusion_matrix_test, classes):
    train_accuracy = np.trace(confusion_matrix_train) / len(classes)
    test_accuracy = np.trace(confusion_matrix_test) / len(classes)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    plt.title('Confusion Matrixes')
    plot_confusion_matrix(confusion_matrix_train, ax[0], train_accuracy, classes, 'Train')
    plot_confusion_matrix(confusion_matrix_test, ax[1], test_accuracy, classes, 'Test')
    plt.show()


def plot_binned_classification_grid(
    data_by_class_and_bin, num_samples, probability_bins=[0, 0.5, 0.75, 0.9, 1], seed=None
):
    rng = np.random.default_rng(seed)
    num_classes = len(data_by_class_and_bin)
    num_bins = len(probability_bins) - 1
    data_dim = int(np.sqrt(data_by_class_and_bin[0][0].shape[1]))

    # Create figure and adjust subplot parameters
    fig = plt.figure(figsize=(3 * num_bins, 3 * num_classes * num_samples))

    # Add title with padding
    fig.suptitle("Representative samples by prediction confidence", fontsize=20, y=0.98)

    # Adjust the layout to make room for title
    gs = plt.GridSpec(num_classes * num_samples, num_bins)
    gs.update(wspace=0.05, hspace=0.05)  # Reduce spacing between subplots

    # Create axes array
    axs = np.empty((num_classes * num_samples, num_bins), dtype=object)

    # Plot samples
    for class_idx in range(num_classes):
        for bin_idx in range(num_bins):
            # Get data for this class and bin
            bin_data = data_by_class_and_bin[class_idx][bin_idx]

            # Plot each sample
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

    # Add bin ranges on top row
    for bin_idx in range(num_bins):
        axs[0, bin_idx].set_title(
            f'[{probability_bins[bin_idx]:.2f},\n{probability_bins[bin_idx+1]:.2f})', pad=10
        )

    # Add class labels on the left with adjusted position
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

    # Add horizontal lines between classes
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

    # Add vertical lines between bins
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

    # Adjust subplot parameters to make room for title
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    plt.show()


def create_dual_histogram(v1, v2, num_bins):
    v_combined = np.concatenate([v1, v2])
    p05 = np.percentile(v_combined, 5)
    p95 = np.percentile(v_combined, 95)
    bin_edges = np.linspace(p05, p95, num_bins + 1)
    return bin_edges


def plot_distributions_and_ROC(positive_samples, negative_samples, title, n_bins=None):
    if n_bins is None:
        n_bins = int(np.sqrt(len(positive_samples) + len(negative_samples)))

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # plot_joint_histograms(ax1, positive_samples, negative_samples, n_bins)
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

    # Plot ROC curve
    labels = np.concatenate([np.zeros(len(negative_samples)), np.ones(len(positive_samples))])
    samples = np.concatenate([negative_samples, positive_samples])
    fpr_emp, tpr_emp, _ = roc_curve(labels, samples)

    ax2.plot(fpr_emp, tpr_emp, 'g--', label=f'Empirical ROC')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

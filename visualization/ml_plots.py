"""Machine learning visualization utilities."""

import numpy as np
from matplotlib import pyplot as plt


def make_weight_figure(predictor, title="", cmap='gray'):
    fig, ax = plt.subplots()
    ax.imshow(predictor, cmap=cmap)
    fig.colorbar(ax.images[0], ax=ax)
    ax.set_title(f"Weight Visualization {title}")
    return fig


def make_predictions_figure(
    X_test, y_test, predictions, indices=None, num_images=10, figsize=(10, 5), cmap='gray'
):
    if indices is None:
        indices = range(min(num_images, len(X_test)))
    else:
        indices = indices[:num_images]

    fig = plt.figure(figsize=figsize)
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(X_test[idx, :-1].reshape(28, 28), cmap=cmap)
        ax.set_title(f'Pred: {predictions[idx]}, True: {y_test[idx]}')
        ax.axis('off')
    fig.tight_layout()
    return fig


def make_grayscale_histogram_figure(predictor):
    fig, ax = plt.subplots()
    counts, bin_edges, _ = ax.hist(predictor.flatten(), bins=30, edgecolor='black')
    ax.set_title("Weight Image Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return fig, counts, bin_edges


def describe_grayscale_histogram(counts, bin_edges) -> None:
    for i in range(len(counts)):
        print(
            f"Bin {i + 1}: Range [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}] has {counts[i]:.0f} values."
        )


def categorize_weight(predictor, threshold=0.01, title="Weight Visualization"):
    new_image = predictor.copy()

    maybe_white = predictor < -threshold
    maybe_gray = abs(predictor) <= threshold
    maybe_black = predictor > threshold

    new_image[maybe_black] = predictor[maybe_black].mean()
    new_image[maybe_gray] = predictor[maybe_gray].mean()
    new_image[maybe_white] = predictor[maybe_white].mean()

    fig = make_weight_figure(
        new_image,
        title
        + f"\nMeans: black = {np.round(new_image[maybe_black].mean(), 3)}, "
        f"gray = {np.round(new_image[maybe_gray].mean(), 3)}, "
        f"white = {np.round(new_image[maybe_white].mean(), 3)}",
    )
    return new_image, fig


def naive_classifier(X, y, label_a=0, label_b=1):
    X_a = np.array(X[y == label_a]) / 255.0
    X_b = np.array(X[y == label_b]) / 255.0
    as_mean = X_b.mean(axis=0)
    bs_mean = X_a.mean(axis=0)

    subtraction = as_mean - bs_mean
    weight_image = subtraction.reshape(28, 28)
    return weight_image


def make_logistic_classification_figure(
    data, labels, final_predictor, title="", xlabel=""
):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(labels)):
        if labels[i] == 1:
            ax.scatter(
                data[i, 0], labels[i], color='blue', label='Class 1' if i == 0 else "", marker='o'
            )
        else:
            ax.scatter(
                data[i, 0], labels[i], color='red', label='Class 0' if i == 0 else "", marker='x'
            )
    x_values = np.linspace(-5, 5, 100)
    logits = final_predictor[0] * x_values + final_predictor[1] * 0.5
    probabilities = 1 / (1 + np.exp(-logits))
    ax.plot(x_values, probabilities, label='Logistic Curve (Class 1 Probability)', color='green')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Labels (0 or 1)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    return fig


def make_losses_per_epoch_figure(
    training_losses: list[np.float64], validation_losses: list[np.float64]
):
    fig, ax = plt.subplots()
    ax.plot(training_losses, label='Training loss')
    ax.plot(validation_losses, label='Validation loss')
    ax.set_title('Training and Validation Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim(0, 0.8)
    ax.legend()
    return fig

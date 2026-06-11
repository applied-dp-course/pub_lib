"""Machine learning visualization utilities."""

import numpy as np
from matplotlib import pyplot as plt


def plot_weight(predictor, title="", cmap='gray'):
    plt.imshow(predictor, cmap=cmap)
    plt.colorbar()
    plt.title(f"Weight Visualization " + title)
    return plt


# Used by lecture_8
def plot_predictions(
    X_test, y_test, predictions, indices=None, num_images=10, figsize=(10, 5), cmap='gray'
):
    if indices is None:
        indices = range(min(num_images, len(X_test)))
    else:
        indices = indices[:num_images]

    plt.figure(figsize=figsize)
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx, :-1].reshape(28, 28), cmap=cmap)
        plt.title(f'Pred: {predictions[idx]}, True: {y_test[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Used by lecture_8
def plot_grayscale_histogram(predictor):
    counts, bin_edges, _ = plt.hist(predictor.flatten(), bins=30, edgecolor='black')
    plt.title("Weight Image Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    for i in range(len(counts)):
        print(
            f"Bin {i + 1}: Range [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}] has {counts[i]:.0f} values."
        )


# Used by lecture_8
def categorize_weight(predictor, threshold=0.01, title="Weight Visualization"):
    new_image = predictor.copy()

    # Define categories based on the threshold
    maybe_white = predictor < -threshold
    maybe_gray = abs(predictor) <= threshold
    maybe_black = predictor > threshold

    # Assign mean values to the new image
    new_image[maybe_black] = predictor[maybe_black].mean()
    new_image[maybe_gray] = predictor[maybe_gray].mean()
    new_image[maybe_white] = predictor[maybe_white].mean()

    plot_weight(
        new_image,
        title + f"\nMeans: black = {np.round(new_image[maybe_black].mean(), 3)}, "
        f"gray = {np.round(new_image[maybe_gray].mean(), 3)}, "
        f"white = {np.round(new_image[maybe_white].mean(), 3)}",
    ).show()

    return new_image


# Used by lecture_8
def naive_classifier(X, y, label_a=0, label_b=1):
    X_a = np.array(X[y == label_a]) / 255.0
    X_b = np.array(X[y == label_b]) / 255.0
    as_mean = X_b.mean(axis=0)
    bs_mean = X_a.mean(axis=0)

    subtraction = as_mean - bs_mean
    weight_image = subtraction.reshape(28, 28)
    return weight_image


# Used by lecture_8
def plot_logistic_classification(data, labels, final_predictor, title="", xlabel=""):
    plt.figure(figsize=(8, 6))
    for i in range(len(labels)):
        if labels[i] == 1:
            plt.scatter(
                data[i, 0], labels[i], color='blue', label='Class 1' if i == 0 else "", marker='o'
            )
        else:
            plt.scatter(
                data[i, 0], labels[i], color='red', label='Class 0' if i == 0 else "", marker='x'
            )
    x_values = np.linspace(-5, 5, 100)
    logits = (
        final_predictor[0] * x_values + final_predictor[1] * 0.5
    )  # Keep feature 2 constant for visualization
    probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid function

    plt.plot(x_values, probabilities, label='Logistic Curve (Class 1 Probability)', color='green')

    plt.xlabel(xlabel)
    plt.ylabel('Labels (0 or 1)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_losses_per_epoch_from_lists(
    training_losses: list[np.float64], validation_losses: list[np.float64]
):
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.8)
    plt.legend()
    plt.show()

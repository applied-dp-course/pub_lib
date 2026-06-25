"""
MNIST-specific utilities.

This module centralizes helpers that assume MNIST conventions such as:
- OpenML dataset id/name defaults like `mnist_784`
- pixel scaling by 255.0
- 28x28 image shapes
- 10-class one-hot encoding (TensorFlow/Keras helpers)

Some helpers (TensorFlow/Keras) use optional dependencies; their imports are
performed inside the functions to keep this module importable without TF.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from libdpy.ml.data_types import LabeledData


def filter_dataset(mask, X, y):
    """Filter an OpenML-style dataset using a label-only mask: mask(y) -> bool/idx."""
    X_filtered = np.array(X[mask(y)])
    y_filtered = np.array(y[mask(y)])
    return X_filtered, y_filtered


def apply_label_flipping(
    data_set: Dict[str, np.ndarray], flip_probability: float, seed=None
) -> Dict[str, np.ndarray]:
    """Flip binary labels (0/1) with probability `flip_probability`."""
    if flip_probability <= 0:
        return data_set

    rng = np.random.default_rng(seed)
    flipped_labels = np.array(
        [
            label if rng.random() > flip_probability else 1 - label
            for label in data_set["labels"]
        ]
    )
    return {"data": data_set["data"], "labels": flipped_labels}


def add_bias_and_normalize(data: np.ndarray) -> np.ndarray:
    """Scale MNIST pixels to [0,1] and append a bias column of 1s."""
    data = data / 255.0
    data_with_bias = np.hstack([data, np.ones((data.shape[0], 1))])
    return data_with_bias


def load_and_preprocess(
    data_params,
    mask=lambda d: (d == 0) | (d == 1),
    data_source: str = "mnist_784",
    seed=None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load MNIST from OpenML, optionally filter labels, normalize pixels, add bias,
    and split into train/test.

    `data_params` is expected to have `test_ratio` and `flip_probability` fields.
    """
    mnist = fetch_openml(data_source, version=1)
    X = mnist["data"]
    y = mnist["target"].astype(np.int8)

    X_filtered, y_filtered = filter_dataset(mask, X, y)
    X_with_bias = add_bias_and_normalize(X_filtered)

    X_train, X_test, y_train, y_test = train_test_split(
        X_with_bias, y_filtered, test_size=data_params.test_ratio, random_state=42
    )

    train_data_set = {"data": X_train, "labels": y_train}
    test_data_set = {"data": X_test, "labels": y_test}

    if getattr(data_params, "flip_probability", 0.0) > 0:
        train_data_set = apply_label_flipping(
            train_data_set, data_params.flip_probability, seed=seed
        )
    return train_data_set, test_data_set


def load_data(
    test_ratio: float = 0.1,
    mask: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda X, y: np.ones_like(y, dtype=bool),
    label_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    data_source: str = "mnist_784",
) -> Tuple[LabeledData, Optional[LabeledData]]:
    """Load MNIST from OpenML as `LabeledData`, with optional filtering/mapping/split."""
    labeled_data = fetch_openml(data_source, version=1)
    X = np.array(labeled_data["data"]) / 255.0
    y = labeled_data["target"].astype(np.int8)
    X_filtered = np.array(X[mask(X, y)])
    y_filtered = np.array(y[mask(X, y)])
    if label_map is not None:
        y_filtered = label_map(y_filtered)
    if test_ratio == 0:
        labeled_train_data = {"data": X_filtered, "labels": y_filtered}
        labeled_test_data = None
        return labeled_train_data, labeled_test_data
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=test_ratio
        )
        labeled_train_data = {"data": X_train, "labels": y_train}
        labeled_test_data = {"data": X_test, "labels": y_test}
        return labeled_train_data, labeled_test_data


# --- TensorFlow/Keras MNIST helpers (optional dependencies) ---


def create_cnn_model():
    """Create a simple CNN model for MNIST (28x28x1 -> 10 classes)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def create_dense_model():
    """Create a simple dense model for MNIST (784 -> 10 classes)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(28 * 28,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def load_and_preprocess_data():
    """Load MNIST via Keras datasets API and one-hot encode labels (10 classes)."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def prepare_data(x_train, x_test, use_cnn: bool):
    """Shape MNIST inputs either as (N,28,28,1) for CNN or (N,784) for MLP."""
    import tensorflow as tf

    if use_cnn:
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    else:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)
    return x_train, x_test

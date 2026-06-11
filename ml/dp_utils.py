from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from libdpy.ml import mnist_utils as _mnist_utils


# Configuration Classes
@dataclass
class DataParameters:
    data_scale: float  # not used
    test_ratio: float
    init_sigma: float  # not used
    bias_factor: float  # not used
    flip_probability: float = 0.0


@dataclass
class LearningParameters:
    learning_rate: float
    num_iterations: int
    noise_factor: float = 0.0
    clipping_radius: float | None = None
    is_noised: bool = False


# Utility Functions
def clip_gradient(gradient: np.ndarray, clipping_radius: float) -> np.ndarray:
    norm = np.linalg.norm(gradient, axis=1, keepdims=True)
    factor = np.clip(clipping_radius / (norm + 1e-8), None, 1)
    return gradient * factor


def filter_dataset(mask, X, y):
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.filter_dataset(mask, X, y)


def calculate_loss(test_probabilities, test_set):
    return -np.mean(
        test_set['labels'] * np.log(test_probabilities + 1e-15)
        + (1 - test_set['labels']) * np.log(1 - test_probabilities + 1e-15)
    )


def apply_label_flipping(
    data_set: Dict[str, np.ndarray], flip_probability: float
) -> Dict[str, np.ndarray]:
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.apply_label_flipping(data_set, flip_probability)


def add_bias_and_normalize(data):
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.add_bias_and_normalize(data)


def load_and_preprocess(
    data_params, mask=lambda d: (d == 0) | (d == 1), data_source: str = 'mnist_784'
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.load_and_preprocess(data_params, mask=mask, data_source=data_source)

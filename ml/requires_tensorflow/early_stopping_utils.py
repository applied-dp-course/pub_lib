import math
from typing import List, Tuple

import numpy as np
from dp_accounting import pld

from libdpy.ml.models import Model
from libdpy.ml.training_utils import (
    calc_accuracy,
    calc_cross_entropy,
    shuffle_data,
    split_to_batches,
    load_data,
    PrivateLearningWithEarlyStoppingParameters,
)
from libdpy.ml.requires_tensorflow.dp_sgd import calc_DP_SGD_delta
from libdpy.ml.data_types import LabeledData


def get_dataset_size(data: LabeledData) -> int:
    return len(data['labels'])


def split_train_validation(
    labeled_data: LabeledData, train_size: float, rng: np.random.Generator | None = None
) -> tuple[LabeledData, LabeledData]:
    if rng is None:
        rng = np.random.default_rng()
    train_size = int(train_size * len(labeled_data['labels']))
    indexes_permutation = rng.permutation(labeled_data['data'].shape[0])
    train_indexes = indexes_permutation[:train_size]
    validation_indexes = indexes_permutation[train_size:]
    train_data = {
        'data': labeled_data['data'][train_indexes],
        'labels': labeled_data['labels'][train_indexes],
    }
    validation_data = {
        'data': labeled_data['data'][validation_indexes],
        'labels': labeled_data['labels'][validation_indexes],
    }
    return train_data, validation_data


def get_data(seed=None) -> Tuple[LabeledData, LabeledData, LabeledData | None]:
    mask = lambda x, y: (y == 0) | (y == 1)
    label_map = lambda y: y

    train_and_validation_data, test_data = load_data(mask=mask, label_map=label_map)
    train_data, validation_data = split_train_validation(
        train_and_validation_data, train_size=0.8, rng=np.random.default_rng(seed)
    )
    return train_data, validation_data, test_data


def calc_dp_sgd_sigma(
    epsilon: float,
    delta: float,
    clipping_radius: float,
    batch_size: int,
    sample_size: int,
    num_epochs: int,
) -> float:
    if epsilon == 0:
        return np.inf
    if epsilon == math.inf:
        return 0
    sigma_upper_bound: float = 100
    search_params = pld.common.BinarySearchParameters(0, sigma_upper_bound)

    return pld.common.inverse_monotone_function(
        lambda sigma: calc_DP_SGD_delta(
            epsilon, sigma, clipping_radius, batch_size, sample_size, num_epochs
        ),
        delta,
        search_params,
    )


def get_cross_entropy_loss(model: Model, data: LabeledData) -> np.float64:
    return np.mean(calc_cross_entropy(model, data))


def get_training_and_validation_loss(
    model: Model, train_data: LabeledData, validation_data: LabeledData
) -> tuple[np.float64, np.float64]:
    training_loss = get_cross_entropy_loss(model, train_data)
    validation_loss = get_cross_entropy_loss(model, validation_data)
    return training_loss, validation_loss


def get_epoch_batches(
    train_data: LabeledData, batch_size: int, rng: np.random.Generator | None = None
) -> List[LabeledData]:
    train_data = shuffle_data(train_data, rng=rng)
    batches = split_to_batches(train_data, batch_size)
    return batches


def dp_sgd_epoch(
    train_data: LabeledData,
    batch_size: int,
    model_class,
    curr_weights,
    params: PrivateLearningWithEarlyStoppingParameters,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    epoch_batches = get_epoch_batches(train_data, batch_size, rng=rng)
    for batch in epoch_batches:
        full_gradient = model_class.calc_full_gradient(curr_weights, batch)
        norm = np.linalg.norm(full_gradient, axis=1, keepdims=True)
        full_gradient *= np.clip(params.clipping_radius / (norm + 1e-8), None, 1)
        gradient = np.mean(full_gradient, axis=0)
        gradient += rng.normal(0, params.noise_factor, gradient.shape)
        curr_weights += params.learning_rate * gradient
    return curr_weights


def get_accuracy(model: Model, data: LabeledData) -> np.float64:
    return np.mean(calc_accuracy(model, data))


def get_loss_change(
    validation_data: LabeledData, current_model: Model, previous_epoch_model: Model
) -> float:
    current_loss_per_element = calc_cross_entropy(current_model, validation_data)
    previous_loss_per_element = calc_cross_entropy(previous_epoch_model, validation_data)
    loss_change_per_element = current_loss_per_element - previous_loss_per_element
    loss_change = float(np.mean(loss_change_per_element))
    return loss_change


def should_stop_early_non_dp(
    validation_data: LabeledData,
    current_model: Model,
    previous_epoch_model: Model,
    loss_change_threshold: float,
) -> bool:
    loss_change = get_loss_change(validation_data, current_model, previous_epoch_model)
    should_stop_early = loss_change > loss_change_threshold
    return should_stop_early

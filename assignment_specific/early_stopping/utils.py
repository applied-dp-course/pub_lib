from typing import List

import numpy as np
from tqdm import tqdm

from libdpy.ml.data_types import LabeledData
from libdpy.ml.models import LogisticModel
from libdpy.ml.requires_tensorflow.early_stopping_utils import (
    calc_dp_sgd_sigma,
    get_dataset_size,
    get_accuracy,
)
from libdpy.ml.training_utils import (
    PrivateBatchLearningParameters,
    PrivateLearningWithEarlyStoppingParameters,
    DP_SGD,
)

DELTA = 1e-6
CLIPPING_RADIUS = 1.0
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOSS_CHANGE_RANGE_MIN = -1
LOSS_CHANGE_RANGE_MAX = 1
LOSS_CHANGE_THRESHOLD = 0


def get_private_learning_params(
    train_data_size: int, epsilon: float, num_epochs: int
) -> PrivateBatchLearningParameters:
    noise_factor = calc_dp_sgd_sigma(
        epsilon, DELTA, CLIPPING_RADIUS, BATCH_SIZE, train_data_size, num_epochs
    )
    training_params = PrivateBatchLearningParameters(
        learning_rate=LEARNING_RATE,
        num_epochs=num_epochs,
        batch_size=BATCH_SIZE,
        noise_factor=noise_factor,
        clipping_radius=CLIPPING_RADIUS,
    )
    return training_params


def get_private_learning_with_early_stopping_params(
    train_data_size: int, epsilon: float, num_epochs: int, is_early_stopping_dp: bool
) -> PrivateLearningWithEarlyStoppingParameters:
    noise_factor = calc_dp_sgd_sigma(
        epsilon, DELTA, CLIPPING_RADIUS, BATCH_SIZE, train_data_size, num_epochs
    )
    training_params = PrivateLearningWithEarlyStoppingParameters(
        learning_rate=LEARNING_RATE,
        num_epochs=num_epochs,
        batch_size=BATCH_SIZE,
        noise_factor=noise_factor,
        clipping_radius=CLIPPING_RADIUS,
        epsilon=epsilon,
        delta=DELTA,
        loss_change_range_min=LOSS_CHANGE_RANGE_MIN,
        loss_change_range_max=LOSS_CHANGE_RANGE_MAX,
        loss_change_threshold=LOSS_CHANGE_THRESHOLD,
        is_early_stopping_dp=is_early_stopping_dp,
    )
    return training_params


def secretly_test_sensitivity(sensitivity):
    is_correct_sensitivity = hash(sensitivity) == 1733065020077936
    if is_correct_sensitivity:
        print("Yay! The sensitivity calculation is correct!")
    else:
        print("The sensitivity calculation is incorrect. Think, fix and try again :)")


def train_models(
    training_data: LabeledData, num_epochs: int, epsilon: float, num_models: int = 1
) -> List[LogisticModel]:
    training_params = get_private_learning_params(
        get_dataset_size(training_data), epsilon, num_epochs
    )
    trained_models = [
        DP_SGD(training_data, LogisticModel, training_params) for _ in tqdm(range(num_models))
    ]
    return trained_models


def get_mean_accuracy(
    training_data: LabeledData,
    test_data: LabeledData,
    num_epochs: int,
    epsilon: float,
    number_of_experiments: int = 5,
) -> np.float64:
    trained_models = train_models(
        training_data, num_epochs, epsilon, num_models=number_of_experiments
    )
    accuracies = [get_accuracy(model, test_data) for model in trained_models]
    mean_accuracy = np.mean(accuracies)
    return mean_accuracy

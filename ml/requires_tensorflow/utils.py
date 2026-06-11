import platform
from typing import Any, Tuple, List

import numpy as np
import tensorflow as tf
from dp_accounting import pld

from libdpy.ml.data_types import LabeledData
from libdpy.ml.models import GradientBasedModel, Model, NaiveModel

from libdpy.ml.training_utils import (
    BatchLearningParameters,
    calc_output_perturbation_delta,
    PrivateBatchLearningParameters,
    DP_naive_training,
    calc_accuracy,
)


def get_tf_optimizer(params: BatchLearningParameters):
    if platform.system() == "Darwin" and "arm" in platform.processor():
        return tf.keras.optimizers.legacy.SGD(learning_rate=params.learning_rate)
    else:
        return tf.keras.optimizers.SGD(learning_rate=params.learning_rate)


def tensorflow_SGD(
    labeled_data: LabeledData, model_class, params: BatchLearningParameters
) -> GradientBasedModel:
    batch_size = params.batch_size if params.batch_size is not None else len(labeled_data['labels'])
    num_batches = len(labeled_data['labels']) // batch_size
    data = tf.data.Dataset.from_tensor_slices(
        (
            labeled_data['data'][: batch_size * num_batches],
            labeled_data['labels'][: batch_size * num_batches],
        )
    ).batch(batch_size, drop_remainder=True)

    num_classes = len(np.unique(labeled_data['labels']))
    model = model_class.get_model(num_classes)

    optimizer = get_tf_optimizer(params)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    weights_arr = [model.get_weights()[0]]
    for _ in range(params.num_epochs):
        model.fit(data, epochs=1, verbose=0)
        weights_arr.append(model.get_weights())
    return model_class(weights_arr)


def calc_output_perturbation_epsilon(sigma: float, delta: float, sensitivity: float) -> float:
    epsilon_upper_bound: float = 100
    search_params = pld.common.BinarySearchParameters(0, epsilon_upper_bound)
    return pld.common.inverse_monotone_function(
        lambda epsilon: calc_output_perturbation_delta((epsilon), sigma, sensitivity),
        delta,
        search_params,
    )


def calc_output_perturbation_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    sigma_upper_bound: float = 100
    search_params = pld.common.BinarySearchParameters(0, sigma_upper_bound)
    return pld.common.inverse_monotone_function(
        lambda sigma: calc_output_perturbation_delta(epsilon, (sigma), sensitivity),
        delta,
        search_params,
    )


def run_DP_naive_experiment(
    train_labeled_data: LabeledData,
    test_labeled_data: LabeledData,
    noise_scale: np.ndarray[Any, np.dtype[np.float64]],
    num_experiments: int,
    delta: float,
) -> Tuple[
    List[Model], np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    training_sample_size = train_labeled_data['data'].shape[0]
    params_size = len(noise_scale)
    models = []
    accuracy_array = np.zeros(params_size)
    epsilon_array = np.zeros(params_size)
    for i in range(params_size):
        params = PrivateBatchLearningParameters(noise_factor=noise_scale[i])
        model = DP_naive_training(train_labeled_data, NaiveModel, params)
        models.append(model)
        accuracy_array[i] = np.mean(
            [
                np.mean(
                    calc_accuracy(
                        DP_naive_training(train_labeled_data, NaiveModel, params), test_labeled_data
                    )
                )
                for _ in range(num_experiments)
            ]
        )
        epsilon_array[i] = calc_output_perturbation_epsilon(
            noise_scale[i], delta, 1.0 / training_sample_size
        )
    return models, accuracy_array, epsilon_array

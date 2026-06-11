from typing import Callable, Tuple, Optional, Any
import numpy as np
from libdpy.ml.data_types import LabeledData
from libdpy.ml.models import LogisticModel, GradientBasedModel
from libdpy.ml.training_utils import (
    load_data,
    BatchLearningParameters,
    get_batch_size,
    init_weights,
    shuffle_and_split,
    evaluate_model,
)

from typing import TypeAlias

matrix: TypeAlias = np.ndarray


def default_noise(gradient: matrix, noise_factor: float) -> matrix:
    return gradient


def process_data(mask, label_map) -> Tuple[LabeledData, Optional[LabeledData]]:
    labeled_train_data, labeled_test_data = load_data(mask=mask, label_map=label_map)
    assert labeled_test_data is not None
    labeled_train_data = {
        'data': labeled_train_data['data'][:2000],
        'labels': labeled_train_data['labels'][:2000],
    }
    labeled_train_data['data'] = LogisticModel.prepare_MNIST_data(labeled_train_data['data'])
    labeled_test_data['data'] = LogisticModel.prepare_MNIST_data(labeled_test_data['data'])
    return labeled_train_data, labeled_test_data


def DP_SGD_optimizer(
    labeled_data: LabeledData,
    model_class,
    params: BatchLearningParameters,
    clipped_gradiant_method: Callable,
    noise_addition_method: Callable,
) -> GradientBasedModel:
    batch_size = get_batch_size(labeled_data, params)
    curr_weights, weights_arr = init_weights(labeled_data, model_class)

    for _ in range(params.num_epochs):
        batches = shuffle_and_split(batch_size, labeled_data)
        for batch_labeled_data in batches:
            full_gradient = model_class.calc_full_gradient(
                curr_weights, batch_labeled_data
            )  # this is a matrix of [gradients] * [number of samples] in the batch

            gradient = clipped_gradiant_method(
                full_gradient, params.clipping_radius
            )  # return mean vector of clipped gradient
            gradient = noise_addition_method(gradient, params.noise_factor)  # gets mean

            curr_weights += params.learning_rate * gradient
        weights_arr.append(curr_weights.copy())
    return model_class(weights_arr)


def run_model(
    params_array: list[BatchLearningParameters],
    clipped_gradiant_method: Callable,
    noise_addition_method: Callable,
    train_data,
    test_data,
):
    model = DP_SGD_optimizer(
        train_data, LogisticModel, params_array[0], clipped_gradiant_method, noise_addition_method
    )
    accuracy, square_loss, log_loss = evaluate_model(model, test_data)
    print(f"Accuracy: {accuracy:.3f}, Square Loss: {square_loss:.3f}, Log Loss: {log_loss:.3f}")

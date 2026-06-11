import math
from dataclasses import dataclass
import typing
from typing import List, Tuple, Any, Callable, Optional, TypeVar

import numpy as np
from scipy import stats

from libdpy.ml.data_types import LabeledData
from libdpy.ml.models import Model, GradientBasedModel
from libdpy.ml import mnist_utils as _mnist_utils

T = TypeVar("T", bound=GradientBasedModel)


@dataclass
class BatchLearningParameters:
    learning_rate: float = 1
    num_epochs: int = 1
    batch_size: int | None = None
    clipping_radius: float | None = None
    noise_factor: float = 0.0


def shuffle_data(labeled_data: LabeledData) -> LabeledData:
    idx = np.random.permutation(labeled_data['data'].shape[0])
    return {'data': labeled_data['data'][idx], 'labels': labeled_data['labels'][idx]}


def split_to_batches(labeled_data: LabeledData, batch_size: int) -> List[LabeledData]:
    sample_size = len(labeled_data['labels'])
    batches = []
    for i in range(0, sample_size, batch_size):
        batch_labeled_data = {
            'data': labeled_data['data'][i : i + batch_size],
            'labels': labeled_data['labels'][i : i + batch_size],
        }
        batches.append(batch_labeled_data)
    return batches


def naive_training(
    labeled_data: LabeledData, model_class=Model, params: BatchLearningParameters | None = None
) -> Model:
    data, labels = labeled_data['data'], labeled_data['labels']
    mean_label_0_data = np.mean(data[labels == 0], axis=0)
    mean_label_1_data = np.mean(data[labels == 1], axis=0)
    weights = mean_label_1_data - mean_label_0_data
    weights /= np.linalg.norm(weights)
    return model_class([weights])


def SGD(
    labeled_data: LabeledData, model_class, params: BatchLearningParameters
) -> GradientBasedModel:
    batch_size = params.batch_size if params.batch_size is not None else len(labeled_data['labels'])
    curr_weights = model_class.get_initial_weights((labeled_data['data'])[0].shape)
    weights_arr = [curr_weights.copy()]

    for _ in range(params.num_epochs):
        labeled_data = shuffle_data(labeled_data)
        batches = split_to_batches(labeled_data, batch_size)

        for batch_labeled_data in batches:
            full_gradient = model_class.calc_full_gradient(curr_weights, batch_labeled_data)
            gradient = np.mean(full_gradient, axis=0)
            curr_weights += params.learning_rate * gradient

        weights_arr.append(curr_weights.copy())
    model = model_class(weights_arr)
    model.learning_rate = params.learning_rate
    return model


def calc_accuracy(model: Model, labeled_data: LabeledData) -> np.ndarray[Any, np.dtypes.BoolDType]:
    predictions = model.predict(labeled_data['data']).argmax(axis=1)
    return typing.cast(np.ndarray[Any, np.dtypes.BoolDType], predictions == labeled_data['labels'])


def calc_square_loss(
    model: Model, labeled_data: LabeledData
) -> np.ndarray[Any, np.dtype[np.float64]]:
    data, labels = labeled_data['data'], labeled_data['labels']
    predictions = model.predict(data)
    if predictions.ndim == 1:
        return (1 - predictions[labels]) ** 2
    else:
        return (1 - predictions[range(len(labels)), labels]) ** 2


def calc_cross_entropy(
    model: Model, labeled_data: LabeledData
) -> np.ndarray[Any, np.dtype[np.float64]]:
    data, labels = labeled_data['data'], labeled_data['labels']
    predictions = model.predict(data)
    if predictions.ndim == 1:
        return -np.log(np.clip(predictions[labels], 1e-4, 1 - 1e-4))
    else:
        return -np.log(np.clip(predictions[range(len(labels)), labels], 1e-4, 1 - 1e-4))


def evaluate_model(model: Model, labeled_test_data: LabeledData) -> Tuple[float, float, float]:
    results = calc_accuracy(model, labeled_test_data)
    accuracy = np.mean(results)
    square_loss = np.mean(calc_square_loss(model, labeled_test_data))
    log_loss = np.mean(calc_cross_entropy(model, labeled_test_data))
    return accuracy, square_loss, log_loss


def calc_confusion_matrix(model: Model, labeled_data: LabeledData) -> np.ndarray:
    data = labeled_data['data']
    labels = labeled_data['labels']
    predictions = model.predict(data)
    classes = np.unique(labels).astype(int)
    confusion_matrix = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
            confusion_matrix[i][j] = np.sum(
                (np.argmax(predictions, axis=1) == j) & (labels == i)
            ) / np.sum(labels == i)

    return confusion_matrix


def split_samples_by_label_and_confidence(
    model: Model, labeled_data: LabeledData, num_bins: int
) -> Tuple[list, np.ndarray]:
    data = labeled_data['data']
    labels = labeled_data['labels']

    predictions = model.predict(data)
    class_probabilities = np.max(predictions, axis=1)
    probability_bins = np.histogram_bin_edges(class_probabilities, bins=num_bins)
    probability_bins[0] = 0.0
    probability_bins[-1] = 1.0

    classes = np.unique(labels).astype(int)
    data_by_class_and_bin = []
    for i in classes:
        data_by_bin = []
        for j in range(num_bins):
            data_by_bin.append(
                data[
                    (probability_bins[j] <= class_probabilities)
                    & (class_probabilities < probability_bins[j + 1])
                    & (labels == i)
                ]
            )
        data_by_class_and_bin.append(data_by_bin)
    return data_by_class_and_bin, probability_bins


def shuffle_and_split(batch_size, labeled_data):
    labeled_data = shuffle_data(labeled_data)
    batches = split_to_batches(labeled_data, batch_size)
    return batches


def init_weights(labeled_data, model_class):
    curr_weights = model_class.get_initial_weights((labeled_data['data'])[0].shape)
    weights_arr = [curr_weights.copy()]
    return curr_weights, weights_arr


def get_batch_size(labeled_data, params):
    batch_size = params.batch_size if params.batch_size is not None else len(labeled_data['labels'])
    return batch_size


def categorize_weights(predictor: np.ndarray, threshold: float = 1) -> np.ndarray:
    pred_mean = np.mean(predictor)
    pred_std = np.std(predictor)
    categories = np.zeros_like(predictor)
    categories[predictor < pred_mean - pred_std * threshold] = -1
    categories[predictor > pred_mean + pred_std * threshold] = 1
    return categories


def categorize_predictor(predictor: np.ndarray, categorized_weights: np.ndarray) -> np.ndarray:
    categories = np.unique(categorized_weights)
    predictor_category_mean = np.zeros_like(categories)
    for i, category in enumerate(categories):
        predictor_category_mean[i] = np.mean(predictor[categorized_weights == category])
    return predictor_category_mean


def load_data(
    test_ratio: float = 0.1,
    mask: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda X, y: np.ones_like(y, dtype=bool),
    label_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    data_source: str = 'mnist_784',
) -> Tuple[LabeledData, Optional[LabeledData]]:
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.load_data(
        test_ratio=test_ratio, mask=mask, label_map=label_map, data_source=data_source
    )


@dataclass
class PrivateBatchLearningParameters(BatchLearningParameters):
    noise_factor: float = 0
    clipping_radius: float = np.inf


def DP_naive_training(
    labeled_data: LabeledData,
    model_class=Model,
    params: PrivateBatchLearningParameters | None = None,
) -> Model:
    assert params is not None
    data, labels = labeled_data['data'], labeled_data['labels']
    mean_label_0_data = np.mean(data[labels == 0], axis=0)
    mean_label_1_data = np.mean(data[labels == 1], axis=0)
    weights = mean_label_1_data - mean_label_0_data
    weights += np.random.normal(0, params.noise_factor, weights.shape)
    weights /= np.linalg.norm(weights)
    return model_class([weights])


def DP_SGD(
    labeled_data: LabeledData, model_class: type[T], params: PrivateBatchLearningParameters
) -> T:
    batch_size = params.batch_size if params.batch_size is not None else len(labeled_data['labels'])
    curr_weights = model_class.get_initial_weights((labeled_data['data'])[0].shape)
    weights_arr = [curr_weights.copy()]

    for _ in range(params.num_epochs):
        labeled_data = shuffle_data(labeled_data)
        batches = split_to_batches(labeled_data, batch_size)

        for batch_labeled_data in batches:
            full_gradient = model_class.calc_full_gradient(curr_weights, batch_labeled_data)
            norm = np.linalg.norm(full_gradient, axis=1, keepdims=True)
            full_gradient *= np.clip(params.clipping_radius / (norm + 1e-8), None, 1)
            gradient = np.mean(full_gradient, axis=0)
            gradient += np.random.normal(0, params.noise_factor, gradient.shape)
            curr_weights += params.learning_rate * gradient
        weights_arr.append(curr_weights.copy())
    model = model_class(weights_arr)
    model.learning_rate = params.learning_rate
    return model  # type: ignore


def calc_output_perturbation_delta(epsilon: float, sigma: float, sensitivity: float) -> float:
    if sigma == 0:
        return np.inf
    norm_sigma = sigma / sensitivity
    upper_cdfs = stats.norm.cdf(0.5 / norm_sigma - norm_sigma * epsilon)
    lower_log_cdfs = stats.norm.logcdf(-0.5 / norm_sigma - norm_sigma * epsilon)
    return upper_cdfs - np.exp(epsilon + lower_log_cdfs)


def calc_average_category_weights(models: List[Model], categories: np.ndarray):
    num_categories = len(np.unique(categories))
    category_means = np.zeros((len(models), num_categories))

    for i, model in enumerate(models):
        category_means[i, :] = categorize_predictor(model.get_final_weights(), categories)
    return category_means


@dataclass
class PrivateLearningWithEarlyStoppingParameters(PrivateBatchLearningParameters):
    epsilon: float = math.inf
    delta: float = 0
    is_early_stopping_dp: bool = True
    loss_change_range_min: float = -1
    loss_change_range_max: float = 1
    loss_change_threshold: float = 0

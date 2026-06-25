from typing import Any, Callable, Tuple

import numpy as np

from libdpy.ml.models import Model
from libdpy.ml.training_utils import LabeledData, shuffle_data, calc_cross_entropy


def choose_extreme_element(
    labeled_data: LabeledData, score_function: Callable[[Any, LabeledData], float], model_generator
):
    model = model_generator(labeled_data)
    score = score_function(model, labeled_data)
    min_index = np.argmin(score)
    return {
        'data': labeled_data['data'][min_index],
        'labels': np.array([labeled_data['labels'][min_index]]),
    }


def calc_elements_score_by_comparison(
    labeled_data: LabeledData, score_function, model_generator, num_subsets: int
):
    size = labeled_data['data'].shape[0]
    subset_size = size // num_subsets

    models = []
    subsets = []
    for i in range(num_subsets):
        subset = {
            'data': labeled_data['data'][subset_size * i : subset_size * (i + 1)],
            'labels': labeled_data['labels'][subset_size * i : subset_size * (i + 1)],
        }
        subsets.append(subset)
        models.append(model_generator(subset))

    score_differences = np.zeros(size)
    for i in range(num_subsets):
        pos_score = score_function(models[i], subsets[i])
        neg_score = np.mean(
            [score_function(models[j], subsets[i]) for j in range(num_subsets) if j != i], axis=0
        )
        score_differences[i * subset_size : (i + 1) * subset_size] = pos_score - neg_score
    return score_differences


def choose_extreme_element_by_comparison(
    labeled_data: LabeledData,
    score_function,
    model_generator,
    num_subsets: int = 10,
    num_experiments: int = 5,
    seed: int | None = None,
) -> LabeledData:
    rng = np.random.default_rng(seed)
    size = labeled_data['data'].shape[0]
    labeled_data = shuffle_data(labeled_data, rng=rng)
    subset_size = size // num_subsets
    size = subset_size * num_subsets
    labeled_data = {'data': labeled_data['data'][:size], 'labels': labeled_data['labels'][:size]}

    score_differences = np.zeros(size)
    for _ in range(num_experiments):
        indexes = rng.permutation(size)
        inverse_indexes = np.argsort(indexes)
        shuffled_data = {
            'data': labeled_data['data'][indexes],
            'labels': labeled_data['labels'][indexes],
        }
        subset_score_differences = calc_elements_score_by_comparison(
            shuffled_data, score_function, model_generator, num_subsets
        )
        score_differences += subset_score_differences[inverse_indexes]

    score_differences /= num_experiments
    max_index = np.argmax(score_differences)
    return {
        'data': labeled_data['data'][max_index],
        'labels': np.array([labeled_data['labels'][max_index]]),
    }


def create_random_datasets(
    labeled_data: LabeledData,
    element: LabeledData,
    subset_size: int | None = None,
    seed: int | None = None,
) -> Tuple[LabeledData, LabeledData]:
    rng = np.random.default_rng(seed)
    if subset_size is None:
        subset_size = labeled_data['data'].shape[0] // 2
    labeled_data = shuffle_data(labeled_data, rng=rng)
    labeled_data = {
        'data': labeled_data['data'][:subset_size],
        'labels': labeled_data['labels'][:subset_size],
    }

    matching_indices = np.where(np.all(labeled_data['data'] == element['data'], axis=1))[0]
    if len(matching_indices) > 0:
        dataset_pos = labeled_data.copy()
        index = matching_indices[0]
        dataset_neg = {
            'data': np.delete(labeled_data['data'], index, axis=0),
            'labels': np.delete(labeled_data['labels'], index, axis=0),
        }
    else:
        dataset_neg = labeled_data.copy()
        dataset_pos = {
            'data': np.append(labeled_data['data'], element['data'].reshape(1, -1), axis=0),
            'labels': np.append(labeled_data['labels'], element['labels']),
        }
    return dataset_pos, dataset_neg


########### Tomer added this function to run the NN example
def create_random_datasets_new(
    labeled_data: LabeledData,
    element: LabeledData,
    subset_size: int | None = None,
    seed: int | None = None,
) -> Tuple[LabeledData, LabeledData]:
    rng = np.random.default_rng(seed)
    if subset_size is None:
        subset_size = labeled_data['data'].shape[0] // 2
    labeled_data = shuffle_data(labeled_data, rng=rng)
    labeled_data = {
        'data': labeled_data['data'][:subset_size],
        'labels': labeled_data['labels'][:subset_size],
    }

    matching_indices = np.where(np.all(labeled_data['data'] == element['data'], axis=1))[0]
    if len(matching_indices) > 0:
        dataset_pos = labeled_data.copy()
        index = matching_indices[0]
        dataset_neg = {
            'data': np.delete(labeled_data['data'], index, axis=0),
            'labels': np.delete(labeled_data['labels'], index, axis=0),
        }
    else:
        dataset_neg = labeled_data.copy()
        dataset_pos = {
            'data': np.append(
                labeled_data['data'], element['data'], axis=0
            ),  # I edited only this part
            'labels': np.append(labeled_data['labels'], element['labels']),
        }
    return dataset_pos, dataset_neg


#############################################################################################


def score_by_log_loss(
    model: Model, labeled_data: LabeledData
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return -calc_cross_entropy(model, labeled_data)


def score_by_gradient(model, labeled_data: LabeledData) -> np.float64:
    gradients = model.calc_gradient(labeled_data)
    if np.ndim(gradients) == 2:
        gradients = np.mean(gradients, axis=0)
    return -np.linalg.norm(gradients)


def score_by_angle(model, labeled_data: LabeledData) -> float:
    data = labeled_data['data']
    weights = model.get_final_weights()
    data_norm = np.linalg.norm(data, axis=1) if np.ndim(data) > 1 else np.linalg.norm(data)
    return np.arccos(np.abs(np.dot(data, weights)) / (data_norm * np.linalg.norm(weights)))

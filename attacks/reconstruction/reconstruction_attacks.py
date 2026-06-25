from typing import Any, Callable
import typing
import numpy as np

from libdpy.ml.data_types import LabeledData


def API_reconstruction_attack(data: np.ndarray, answers: np.ndarray) -> np.ndarray:
    return np.round(answers)


def query_function_API(data, model):
    return model.predict(data)[:, 1]


def query_function_WB(data, model, learning_rate: float | None = None) -> np.ndarray:
    sample_size = data.shape[0]
    weights_list = model.get_intermediate_weights()
    if learning_rate is None:
        learning_rate = getattr(model, "learning_rate", 1.0)
    prob = np.clip(1 / (1 + np.exp(-np.dot(data, weights_list[-2]))), 1e-6, 1 - 1e-6)
    answers = (
        np.dot(data.T, prob) + ((weights_list[-1] - weights_list[-2]) / learning_rate) * sample_size
    )
    return answers


def query_function_BB(data: np.ndarray, model) -> np.ndarray:
    # Black-box reconstruction query: dot product of data and predicted probabilities.
    prob = model.predict(data)[:, 1]
    answers = np.dot(data.T, prob)
    return answers


def perform_reconstruction_attack(
    model,
    data: np.ndarray,
    query_func: Callable[[np.ndarray, Any], np.ndarray],
    reconstruction_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    answers = query_func(data, model)
    reconstructed_target = reconstruction_func(data.T, answers)
    return reconstructed_target


def get_mean_reconstruction_error(reconstructed_target: np.ndarray, x: np.ndarray) -> float:
    return typing.cast(float, np.mean(abs(reconstructed_target - x.T)))


def compute_average_error_reconstruction_attack(
    mechanism,
    labeled_data: LabeledData,
    query_func: Callable[[np.ndarray, Any], np.ndarray],
    reconstruction_func,
    num_experiments: int,
    subset_size: int,
    resample: bool = False,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    cumulative_error = 0.0
    data_size = labeled_data['data'].shape[0]
    for _ in range(num_experiments):
        subset_indices = rng.choice(data_size, subset_size, replace=False)
        subset_data = labeled_data['data'][subset_indices]
        subset_labels = labeled_data['labels'][subset_indices]

        # merge the data and labels into a single
        model = mechanism({'data': subset_data, 'labels': subset_labels})
        if resample:
            subset_indices = rng.choice(data_size, subset_size, replace=False)
            subset_data = labeled_data['data'][subset_indices]
            subset_labels = labeled_data['labels'][subset_indices]
        reconstructed_vector = perform_reconstruction_attack(
            model, subset_data, query_func, reconstruction_func
        )
        cumulative_error += get_mean_reconstruction_error(reconstructed_vector, subset_labels)
    return np.round(cumulative_error / num_experiments, 4)


def choose_random_subset(df, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    masks_by_size = {len(df): rng.choice([0, 1], size=len(df))}

    def query(data):
        data_size = len(data)
        if data_size not in masks_by_size:
            masks_by_size[data_size] = rng.choice([0, 1], size=data_size)
        return masks_by_size[data_size]

    return query


def get_reconstruction_error(reconstructed_target: np.ndarray, x: np.ndarray) -> float:
    return typing.cast(float, np.sum(abs(reconstructed_target - x.T)))

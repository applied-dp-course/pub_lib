from abc import abstractmethod

import numpy as np
from typing import Any

from libdpy.ml.data_types import LabeledData


class Model:
    @staticmethod
    @abstractmethod
    def prepare_MNIST_data(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    @staticmethod
    @abstractmethod
    def data_as_image(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    def __init__(self, weights_arr: np.ndarray[Any, np.dtype[np.float64]], num_classes: int = 2):
        self.weights_arr = weights_arr
        self.final_weights = weights_arr[-1]

    @abstractmethod
    def predict(
        self, data: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def get_final_weights(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass


class GradientBasedModel(Model):
    @staticmethod
    @abstractmethod
    def calc_full_gradient(
        curr_weights: np.ndarray[Any, np.dtype[np.float64]], labeled_data: LabeledData
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    @staticmethod
    @abstractmethod
    def get_initial_weights(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def calc_gradient(self, labeled_data: LabeledData) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def get_intermediate_weights(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass


class LogisticModel(GradientBasedModel):
    @staticmethod
    def prepare_MNIST_data(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.hstack([data, np.ones((data.shape[0], 1))])

    def predict(
        self, data: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        prob = 1 / (1 + np.exp(-np.dot(data, self.final_weights)))
        return np.array([1 - prob, prob]).T

    def get_final_weights(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.final_weights

    @staticmethod
    def data_as_image(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        sqrt_size = int(np.sqrt(data.shape[0]))
        return data[: sqrt_size**2].reshape(sqrt_size, sqrt_size)

    @staticmethod
    def calc_full_gradient(
        curr_weights: np.ndarray[Any, np.dtype[np.float64]], labeled_data: LabeledData
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        data, labels = labeled_data['data'], labeled_data['labels']
        prob = np.clip(1 / (1 + np.exp(-np.dot(data, curr_weights))), 1e-4, 1 - 1e-4)
        return data * (labels - prob)[:, np.newaxis]

    @staticmethod
    def get_initial_weights(shape: tuple[int]) -> np.ndarray[Any, np.dtype[np.float64]]:  # type: ignore
        return np.zeros(shape)

    def calc_gradient(self, labeled_data: LabeledData) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.calc_full_gradient(self.final_weights, labeled_data)

    def get_intermediate_weights(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.weights_arr


class NaiveModel(Model):
    @staticmethod
    def prepare_MNIST_data(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return data / np.linalg.norm(data, axis=1)[:, np.newaxis]

    @staticmethod
    def data_as_image(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        sqrt_size = int(np.sqrt(data.shape[0]))
        return data.reshape(sqrt_size, sqrt_size)

    def predict(
        self, data: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        prob = (np.dot(data, self.final_weights) + 1) / 2
        return np.array([1 - prob, prob]).T

    def get_final_weights(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.final_weights

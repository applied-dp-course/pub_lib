from typing import Any

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten

from libdpy.ml.data_types import LabeledData
from libdpy.ml.models import GradientBasedModel


class DNNModel(GradientBasedModel):
    @staticmethod
    def _reshape_flat_images(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        if data.ndim != 2:
            return data

        side = int(np.sqrt(data.shape[1]))
        if side * side != data.shape[1]:
            raise ValueError("Flat image data must have a square number of features.")
        return data.reshape(-1, side, side)

    @staticmethod
    def prepare_MNIST_data(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return data.reshape(-1, 28, 28)

    @staticmethod
    def data_as_image(
        data: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return data

    def __init__(self, weights_arr: np.ndarray[Any, np.dtype[np.float64]], num_classes: int = 2):
        super().__init__(weights_arr)
        self.model = self.get_model(num_classes)
        # Don't set weights in __init__ - TensorFlow model has its own weight structure
        # self.final_weights is used for other purposes in the parent class

    def predict(
        self, data: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        data = self._reshape_flat_images(data)
        return self.model.predict(data, verbose=0)

    def get_final_weights(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        raise NotImplementedError("DNN weights are not supported by weight-visualization helpers.")

    def calc_gradient(self, labeled_data: LabeledData) -> np.ndarray[Any, np.dtype[np.float64]]:
        data = self._reshape_flat_images(labeled_data['data'])
        input_data = tf.convert_to_tensor(data)
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            predictions = self.model(input_data)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                labeled_data['labels'], predictions
            )
        return tape.gradient(loss, input_data).numpy()

    def get_intermediate_weights(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.weights_arr

    @staticmethod
    def get_model(num_classes: int = 2) -> Sequential:
        # Create a simple TensorFlow model
        model = Sequential(
            [Flatten(), Dense(128, activation='relu'), Dense(num_classes, activation='softmax')]
        )
        return model


class FullyConnectedModel(DNNModel):
    @staticmethod
    def get_model(num_classes: int = 2):
        return Sequential(
            [
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(num_classes, activation='softmax'),
            ]
        )


class FullyDeepConnectedModel(DNNModel):
    @staticmethod
    def get_model(num_classes: int = 2) -> Sequential:
        return tf.keras.Sequential(
            [
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(num_classes, activation='softmax'),
            ]
        )

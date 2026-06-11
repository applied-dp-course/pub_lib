from typing import Dict

import numpy as np

from libdpy.ml.dp_utils import LearningParameters, calculate_loss


def clipping_method(gradient: np.ndarray, c: float):  # accuracy 99.90
    norms = np.linalg.norm(gradient, axis=1, keepdims=True)  # Compute norms of each row
    scaling_factors = np.minimum(norms, c) / norms  # Compute scaling factors
    return gradient * scaling_factors


class LogisticRegressionModel:
    def __init__(self, input_dim: int, clipping_method):
        self.final_weights = np.zeros(input_dim)
        self.clipping_method = clipping_method

    def train(self, data_set: Dict[str, np.ndarray], params: LearningParameters):
        weights_arr = [self.final_weights.copy()]
        data, labels = data_set['data'], data_set['labels']

        for _ in range(params.num_iterations):
            prob = np.clip(1 / (1 + np.exp(-np.dot(data, self.final_weights))), 1e-4, 1 - 1e-4)
            full_gradient = data * (labels - prob)[:, np.newaxis]

            if params.clipping_radius:
                full_gradient = self.clipping_method(full_gradient, params.clipping_radius)
            gradient = np.mean(full_gradient, axis=0)

            if params.is_noised:
                noise = np.random.normal(0, params.noise_factor, gradient.shape)
                gradient += noise

            self.final_weights += params.learning_rate * gradient
            weights_arr.append(self.final_weights.copy())
        return weights_arr

    def predict(self, data: np.ndarray) -> np.ndarray:
        prod = np.dot(data, self.final_weights)
        return 1 / (1 + np.exp(-prod))


def train_and_evaluate(
    processed_data, learning_params: LearningParameters, clipping_methods, print_result=True
):
    train_set, test_set = processed_data
    model = LogisticRegressionModel(train_set['data'].shape[1], clipping_methods)
    weights_arr = model.train(train_set, learning_params)

    # Evaluate on test data
    test_probabilities = model.predict(test_set['data'])
    test_loss = calculate_loss(test_probabilities, test_set)

    predictions = (test_probabilities >= 0.5).astype(int)
    accuracy = np.mean(predictions == test_set['labels'])

    if print_result:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    return weights_arr, predictions, test_loss, accuracy

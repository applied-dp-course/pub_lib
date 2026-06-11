from typing import Callable

import numpy as np
import pandas as pd

from libdpy.attacks.reconstruction.reconstruction_attacks import get_reconstruction_error


class QueryMechanism:
    """
    A class to simulate querying a database, with noisy response.
    """

    def __init__(
        self, database, sensitive_attribute, public_features, noise_function, frame_size=100
    ):
        self.__full_database = database
        self.__is_dataframe = isinstance(database, pd.DataFrame)

        if self.__is_dataframe:
            self.dataset = self.__full_database.head(frame_size)
        else:
            # For numpy arrays, just take first frame_size rows
            self.dataset = self.__full_database[:frame_size]

        self.sensitive_attribute = sensitive_attribute
        self.public_features = public_features
        self.dataset_size = frame_size
        self.noise_function = noise_function

    def set_dataset(self, new_size=100):
        """Randomly samples a new dataset frame of specified size for querying."""
        self.dataset_size = new_size

        if self.__is_dataframe:
            # For DataFrames, ensure we don't sample more than available
            available_size = len(self.__full_database)
            actual_size = min(new_size, available_size)
            if actual_size == available_size:
                self.dataset = self.__full_database
            else:
                self.dataset = self.__full_database.sample(n=actual_size)
        else:
            # For numpy arrays, just take first new_size rows
            self.dataset = self.__full_database[:new_size]

    def get_public_data(self):
        """Retrieves the current frame with only the public features."""
        if self.__is_dataframe:
            return self.dataset[self.public_features]
        else:
            # For numpy arrays, return as is (assuming all columns are public)
            return self.dataset

    def get_true_response_for_compression_only(self):
        """Gets the values of the sensitive attribute for the current frame.
        SHOULD NOT be used to anything but comparison."""
        if self.__is_dataframe:
            return self.dataset[self.sensitive_attribute]
        else:
            # For numpy arrays, return the last column as sensitive attribute
            return self.dataset[:, -1] if self.dataset.ndim > 1 else self.dataset

    def exact_query_for_compression_only(self, queries):
        """Computes the exact response for a list of queries on the sensitive attribute.
        Used for examples and internal functions only."""
        true_response = self.get_true_response_for_compression_only()
        if self.__is_dataframe:
            true_response = true_response.values

        return true_response @ np.stack(
            [query(self.get_public_data()) for query in queries], axis=1
        )

    def noisy_query(self, queries, noise):
        """Computes a noisy response by adding noise to the exact query results."""
        Qx = self.exact_query_for_compression_only(queries)
        z = self.noise_function(noise, queries)
        r = Qx + z
        return r

    def query_public_data(self, queries):
        """Applies queries only to the public features in the current frame."""
        return np.stack([query(self.get_public_data()) for query in queries], axis=1)


def run_simulation(
    data_size: int,
    mechanism: QueryMechanism,
    num_experiments: int,
    recon_func: Callable,
    query_function: Callable,
    noise_range: list[int],
    num_queries: int,
):
    """Simulate the reconstruction error across different noise scales."""
    lin_reg_error_arr = np.zeros(len(noise_range))

    for i, noise in enumerate(noise_range):
        lin_reg_error_arr[i] = compute_average_error_for_noise_scale(
            data_size, mechanism, num_experiments, recon_func, query_function, noise, num_queries
        )
    return lin_reg_error_arr


def compute_average_error_for_noise_scale(
    data_size, mechanism, num_experiments, recon_func, query_function, noise, num_queries
):
    """Compute the average over num_experiments reconstruction error for a given noise scale."""
    cumulative_error = 0

    for _ in range(num_experiments):
        mechanism.set_dataset(data_size)
        original_vector = mechanism.get_true_response_for_compression_only()
        reconstructed_vector = perform_reconstruction_attack_mechanizm(
            mechanism, recon_func, query_function, noise, num_queries
        )
        cumulative_error += get_reconstruction_error(reconstructed_vector, original_vector)

    return cumulative_error / (num_experiments * data_size)


def perform_reconstruction_attack_mechanizm(
    mechanism, recon_func, query_function, noise_scale, num_queries
):
    """The whole mechanism - true response + adding a noise"""
    public_data = mechanism.get_public_data()
    Q = [query_function(public_data) for _ in range(num_queries)]
    subsets = mechanism.query_public_data(Q).T
    r = mechanism.noisy_query(Q, noise_scale)

    if recon_func == mysterious_reconstructor:
        reconstructed_target = recon_func(subsets, public_data, r)
    else:
        reconstructed_target = recon_func(subsets, r)
    return reconstructed_target


def mysterious_predictor(features):
    if features['latino'] <= 0.5:
        if features['asian'] <= 0.5:
            if features['englishability'] <= 0.5:
                if features['educ'] <= 9.5:
                    return 0
                else:
                    return 1
            else:
                if features['militaryservice'] <= 0.5:
                    return 1
                else:
                    return 1
        else:
            if features['englishability'] <= 0.5:
                if features['divorced'] <= 0.5:
                    return 0
                else:
                    return 0
            else:
                if features['educ'] <= 13.5:
                    return 1
                else:
                    return 0
    else:
        if features['englishability'] <= 0.5:
            if features['disability'] <= 0.5:
                if features['asian'] <= 0.5:
                    return 0
                else:
                    return 1
            else:
                if features['educ'] <= 4.5:
                    return 0
                else:
                    return 0
        else:
            if features['educ'] <= 9.5:
                if features['black'] <= 0.5:
                    return 0
                else:
                    return 1
            else:
                if features['divorced'] <= 0.5:
                    return 1
                else:
                    return 1


def mysterious_reconstructor(subsets, public_frame, noise_response):
    return public_frame.apply(lambda row: mysterious_predictor(row.to_dict()), axis=1)

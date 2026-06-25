from typing import Callable

import numpy as np
from libdpy.privacy_mechanisms.noise import laplace_noise


class AboveThresholdOnline:
    def __init__(
        self,
        database,
        epsilon: float,
        sensitivity: float,
        threshold: float,
        seed: int | None = None,
    ):
        self.database = database
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.rng = np.random.default_rng(seed)
        self.noised_threshold = threshold + laplace_noise(
            2 * self.sensitivity / self.epsilon, rng=self.rng
        )
        self.can_be_used = True

    def is_above_threshold(self, query: Callable):
        if not self.can_be_used:
            raise Exception(
                "AboveThresholdOnline has already answered positively and cannot be used again."
            )
        response = query(self.database)
        noised_response = response + laplace_noise(
            4 * self.sensitivity / self.epsilon, rng=self.rng
        )
        is_above_threshold = noised_response >= self.noised_threshold
        if is_above_threshold:
            self.can_be_used = False
        return is_above_threshold


if __name__ == '__main__':
    database = [1966, 1978, 1994, 2002, 2004, 2005, 2009, 2011, 2013, 2021]
    above_threshold_online = AboveThresholdOnline(
        database=database, threshold=5, epsilon=5, sensitivity=1
    )
    for i in range(2025, 1948, -1):
        count_later_than_year_query = lambda data, i=i: len([item for item in data if item >= i])
        is_above_threshold = above_threshold_online.is_above_threshold(count_later_than_year_query)
        if is_above_threshold:
            print(i)
            break

from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta

from libdpy.hypothesis_testing import (
    FPR_from_threshold,
    TPR_from_threshold,
    threshold_from_FPR,
)


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """
    This function returns a lower bound of the success probability using Clopper Pearson method.
    :param successes: number of successes
    :param trials: number of trials
    :param alpha: the probability not to get a lower bound
    :return: the lower bound of the success probability
    """
    return beta.ppf(alpha, successes, trials - successes + 1)


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """This function returns an upper bound of the success probability using Clopper Pearson method."""
    return beta.ppf(1 - alpha, successes + 1, trials - successes)


def legacy_algorithm(database: list) -> float:
    """secret legacy algorithm"""
    value = database[-1]
    trimmed_value = min(5, max(4, value))
    noised_value = trimmed_value + np.random.laplace(scale=1 / 3)
    return noised_value


def plot_outputs_histograms(algorithm: Callable, db0, db1, repetitions_number: int):
    results_0 = [algorithm(db0) for _ in range(repetitions_number)]
    results_1 = [algorithm(db1) for _ in range(repetitions_number)]
    plt.figure(figsize=(8, 6))
    plt.hist(
        results_0,
        bins=100,
        density=True,
        color='b',
        edgecolor='black',
        label=f'Mean: {np.mean(results_0)}\nStandard deviation: {np.std(results_0)}',
    )
    plt.hist(
        results_1,
        bins=100,
        density=True,
        color='y',
        edgecolor='yellow',
        label=f'Mean: {np.mean(results_1)}\nStandard deviation: {np.std(results_1)}',
    )

    plt.title("Histograms of results")
    plt.xlabel("result")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return

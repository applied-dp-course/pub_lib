from typing import Callable, List

import numpy as np
from libdpy.attacks.membership_inference.classical_auditing_utils import get_mean_estimation_error


def compute_exact_height_of_shortest(students_heights: list, number_of_players: int = 5) -> float:
    """Compute exact height of the shortest player in a team."""
    exact_height_of_shortest = sorted(students_heights, reverse=True)[number_of_players - 1]
    return exact_height_of_shortest


def get_students_height_mean_estimation_error(
    estimation_function: Callable, students_heights: List[float], experiments_number: int
) -> float:
    return get_mean_estimation_error(
        estimation_function=estimation_function,
        data_generator=lambda: students_heights,
        error_function=lambda data, estimations: abs(
            estimations - compute_exact_height_of_shortest(data)
        ),
        experiments_number=experiments_number,
    )


def count_pass_threshold(database: list, threshold: float) -> int:
    """
    This function counts the number of items in the database that are greater or equal to the threshold.
    """
    return len([item for item in database if item >= threshold])


def get_count_pass_threshold_query(threshold: float) -> Callable:
    """
    This function generates a query that counts the number of items in the database that are greater or equal to the threshold.
    """
    return lambda database: count_pass_threshold(database, threshold)


NUMBER_OF_PLAYERS_IN_BASKETBALL = 5
HEIGHTS_TO_CHECK = np.linspace(start=2.1, stop=1.5, num=13).tolist()


def parse_above_threshold_responses(above_threshold_responses: List[bool]) -> float:
    """This function parses the responses of the Above Threshold mechanism"""
    if True in above_threshold_responses:
        positive_answer_index = len(above_threshold_responses) - 1
        return HEIGHTS_TO_CHECK[positive_answer_index]
    else:
        return 1.5  # We can only guess


def noiseless_above_threshold(
    database: list, queries: List[Callable], threshold: float, epsilon: float = 0.0
) -> List[bool]:
    """
    Noiseless version of the Above Threshold mechanism.

    Args:
        database: A sensitive database
        queries: List of queries with sensitivity=1
        threshold: The threshold value for query results
        epsilon: Privacy parameter (not used in noiseless version)

    Returns:
        List of boolean responses - whether results pass the threshold.
        The list ends with the first positive result, if exists.
    """
    responses = []
    for query in queries:
        response = query(database)
        if response >= threshold:
            responses.append(True)
            return responses
        else:
            responses.append(False)
    return responses


# ReportNoisyMax for estimating the median, given a sample X, a range R, privacy parameter Eps, and number of queries T
def ReportNoisyMax_for_median(X, n, R, Eps, T, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    L, U = R
    responses = np.linspace(start=L, stop=U, num=T).tolist()
    ind_max = np.argmax(
        [
            -abs(np.sum(X <= response) - ((n - 1) / 2 + 1)) + rng.laplace(0, 1 / Eps)
            for response in responses
        ]
    )
    return responses[ind_max]


def DP_binary_search(X, n, R, Eps, T, ind, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    counter_right = 0
    L, U = R
    for t in range(T):
        mid = (L + U) / 2
        count = np.sum(X <= mid) + rng.laplace(0, 1 / (Eps / T))
        if count > n / 2:
            U = mid
        else:
            L = mid
            counter_right += 1
    if ind:
        return counter_right / T
    return mid

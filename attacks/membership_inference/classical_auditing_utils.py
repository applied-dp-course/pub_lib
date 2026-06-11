import math
from typing import Callable

import numpy as np
from scipy.stats import beta


def compute_threshold(responses_pos, responses_neg, error_prob, delta) -> float:
    num_experiments = len(responses_pos)

    # Handle empty arrays
    if num_experiments == 0:
        return 0.0

    threshold_list = np.unique(np.concatenate((responses_pos, responses_neg)))
    threshold_list.sort()
    threshold_list = threshold_list[:-1]

    # Handle case where threshold_list is empty
    if len(threshold_list) == 0:
        return 0.0

    eps_lower_bound_estimate = []
    for threshold in threshold_list:
        num_pos_guesses_neg = np.sum([response > threshold for response in responses_neg])
        num_pos_guesses_pos = np.sum([response > threshold for response in responses_pos])
        FPR = beta.ppf(
            1 - error_prob / 2, num_pos_guesses_neg + 1, num_experiments - num_pos_guesses_neg
        )
        TPR = beta.ppf(
            error_prob / 2, num_pos_guesses_pos, num_experiments - num_pos_guesses_pos + 1
        )
        if TPR > 0 and FPR == 0:
            eps_lower_bound_estimate.append(np.inf)
        else:
            if TPR - delta <= FPR:
                eps_lower_bound_estimate.append(0)
            else:
                eps_lower_bound_estimate.append(np.log((TPR - delta) / FPR))

    # Handle empty eps_lower_bound_estimate
    if len(eps_lower_bound_estimate) == 0:
        return 0.0

    return threshold_list[np.argmax(np.nan_to_num(eps_lower_bound_estimate))]


def threshold_guesser(response: float, threshold: float) -> bool:
    return response > threshold


def get_mean_estimation_error(
    estimation_function: Callable,
    data_generator: Callable,
    error_function: Callable,
    experiments_number: int,
    **kwargs,
) -> float:
    errors_sum = 0.0
    for _ in range(experiments_number):
        data = data_generator()
        estimation = estimation_function(data, **kwargs)
        errors_sum += error_function(data, estimation)
    mean_error = errors_sum / experiments_number
    return mean_error


def get_median_error(data: list, estimation: float) -> float:
    return abs(np.median(data) - estimation)


def epsilon_lower_bound(
    mechanism: Callable,
    dataset_neg: list,
    dataset_pos: list,
    guesser: Callable,
    num_experiments: int,
    error_prob: float,
    delta: float = 0,
) -> float:
    responses_neg = [mechanism(dataset_neg) for _ in range(num_experiments)]
    responses_pos = [mechanism(dataset_pos) for _ in range(num_experiments)]

    num_pos_guesses_neg = np.sum([guesser(response) for response in responses_neg])
    num_pos_guesses_pos = np.sum([guesser(response) for response in responses_pos])

    if error_prob < 0:
        FPR = num_pos_guesses_neg / num_experiments
        TPR = num_pos_guesses_pos / num_experiments
    else:
        FPR = beta.ppf(
            1 - error_prob / 2, num_pos_guesses_neg + 1, num_experiments - num_pos_guesses_neg
        )
        TPR = beta.ppf(
            error_prob / 2, num_pos_guesses_pos, num_experiments - num_pos_guesses_pos + 1
        )

    if TPR > 0 and FPR == 0:
        return math.inf
    if TPR - delta <= FPR:
        return 0
    return np.log((TPR - delta) / FPR)

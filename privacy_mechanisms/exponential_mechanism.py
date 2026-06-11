import random
from typing import Callable, List, TypeVar

import numpy as np

T = TypeVar('T')


def get_exact_median(database: list) -> float:
    return float(np.median(database))


def exponential_mechanism(
    database: list,
    possible_responses: List[T],
    utility_function: Callable,
    sensitivity: float,
    epsilon: float,
) -> T:
    scores = [utility_function(database, r) for r in possible_responses]
    probabilities = np.array([np.exp(epsilon * score / (2 * sensitivity)) for score in scores])
    probabilities = probabilities / np.sum(probabilities)
    response = random.choices(possible_responses, weights=probabilities, k=1)[0]
    return response

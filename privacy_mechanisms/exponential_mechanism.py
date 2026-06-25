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
    rng: np.random.Generator | None = None,
) -> T:
    if rng is None:
        rng = np.random.default_rng()
    scores = [utility_function(database, r) for r in possible_responses]
    probabilities = np.array([np.exp(epsilon * score / (2 * sensitivity)) for score in scores])
    probabilities = probabilities / np.sum(probabilities)
    response = possible_responses[rng.choice(len(possible_responses), p=probabilities)]
    return response

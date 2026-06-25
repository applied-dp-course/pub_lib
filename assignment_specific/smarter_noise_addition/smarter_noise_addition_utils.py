from typing import List, Callable, Tuple

import numpy as np

from libdpy.attacks.membership_inference.classical_auditing_utils import get_mean_estimation_error
from libdpy.privacy_mechanisms.noise import laplace_noise

PASSENGERS_NUMBER = 10
WEIGHTS_RANGES = [(i * 20, (i + 1) * 20) for i in range(10)]
Range = Tuple[float, float]  # Format: (range_min, range_max)


def get_error(weights: List[float], estimation: float) -> float:
    """Calculate error between true sum and estimation."""
    return abs(sum(weights) - estimation)


def get_passengers_weights(rng: np.random.Generator | None = None) -> List[float]:
    """Generate random passenger weights for evaluation."""
    if rng is None:
        rng = np.random.default_rng()
    weights_range = WEIGHTS_RANGES[rng.integers(len(WEIGHTS_RANGES))]
    weights_range_min, weights_range_max = weights_range
    passengers_weights = [
        rng.uniform(weights_range_min, weights_range_max) for _ in range(PASSENGERS_NUMBER)
    ]
    return passengers_weights


def get_passengers_weight_mean_estimation_error(
    estimation_function: Callable, epsilon: float, experiments_number: int, seed=None, **kwargs
) -> float:
    """Calculate mean estimation error over multiple experiments."""
    rng = np.random.default_rng(seed)
    return get_mean_estimation_error(
        estimation_function=estimation_function,
        data_generator=lambda: get_passengers_weights(rng=rng),
        epsilon=epsilon,
        experiments_number=experiments_number,
        error_function=get_error,
        **kwargs,
    )


def project(value: float, range_min: float, range_max: float) -> float:
    return max(range_min, min(value, range_max))


def estimate_sum(
    weights: list,
    weights_range: Range,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> float:
    """
    This function estimates the sum of the weights in a DP way.
    It projects the weights to a predefined range to ensure privacy.
    If the weights are not in this range, the estimation is inaccurate.
    :param weights: list of 10 weights
    :param weights_range: the expected range of the weights.
    :param epsilon: the privacy parameter
    :return: float - the estimated sum
    """
    range_min, range_max = weights_range
    sensitivity = range_max - range_min
    projected_weights = [project(weight, range_min, range_max) for weight in weights]
    estimated_sum = sum(projected_weights) + laplace_noise(scale=sensitivity / epsilon, rng=rng)
    return estimated_sum

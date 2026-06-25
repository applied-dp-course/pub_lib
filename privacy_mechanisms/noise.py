from collections.abc import Callable

import numpy as np
from scipy import stats


def laplace_noise(scale: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.laplace(loc=0, scale=scale))


def Laplace_sum(dataset: list, scale: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    return np.sum(dataset) + rng.laplace(0, scale)


def Laplace_median(dataset: list, scale: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    return np.median(dataset) + rng.laplace(0, scale)


def Gaussian_sum(dataset: list, scale: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    return np.sum(dataset) + rng.normal(0, scale)


def Gaussian_median(dataset: list, scale: float, rng: np.random.Generator | None = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    return np.median(dataset) + rng.normal(0, scale)


def logistic_noise(
    size: int | tuple[int, ...] = 1,
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw independent logistic noise samples."""

    if rng is None:
        rng = np.random.default_rng()
    return stats.logistic(loc=0, scale=scale).rvs(size=size, random_state=rng)


def two_logistic_noise(
    size: int | tuple[int, ...] = 1,
    scale1: float = 1.0,
    scale2: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw independent samples of the sum of two logistics."""

    if rng is None:
        rng = np.random.default_rng()
    z1 = stats.logistic(loc=0, scale=scale1).rvs(size=size, random_state=rng)
    z2 = stats.logistic(loc=0, scale=scale2).rvs(size=size, random_state=rng)
    return z1 + z2


def sample_outputs(
    noise_sampler: Callable[..., np.ndarray],
    mu: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return ``mu + noise_sampler(size=n, rng=rng)``."""

    return mu + noise_sampler(size=n, rng=rng)

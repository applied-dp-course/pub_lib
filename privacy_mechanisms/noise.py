import numpy as np


def laplace_noise(scale: float) -> float:
    return float(np.random.laplace(loc=0, scale=scale))


def Laplace_sum(dataset: list, scale: float) -> float:
    return np.sum(dataset) + np.random.laplace(0, scale)


def Laplace_median(dataset: list, scale: float) -> float:
    return np.median(dataset) + np.random.laplace(0, scale)


def Gaussian_sum(dataset: list, scale: float) -> float:
    return np.sum(dataset) + np.random.normal(0, scale)


def Gaussian_median(dataset: list, scale: float) -> float:
    return np.median(dataset) + np.random.normal(0, scale)

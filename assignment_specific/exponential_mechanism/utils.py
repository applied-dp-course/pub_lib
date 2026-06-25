import numpy as np

from matplotlib import pyplot as plt
from libdpy.attacks.membership_inference.classical_auditing_utils import (
    get_mean_estimation_error,
    get_median_error,
)


MIN_GRADE, MAX_GRADE = 0, 100
EPSILON = 1
POSSIBLE_MEDIANS = [
    MIN_GRADE + 0.5 * i for i in range(2 * (MAX_GRADE - MIN_GRADE) + 1)
]  # 0, 0.5, 1, ..., 100


def get_random_grades(seed=None) -> list:
    """Generate random grades for evaluation.

    ``seed`` may be an int (reproducible), ``None`` (fresh randomness), or an existing
    ``np.random.Generator`` (threaded from a caller), thanks to ``default_rng`` pass-through.
    """
    rng = np.random.default_rng(seed)

    grades_number = rng.integers(50, 150)
    expected_grade = rng.uniform(70, 90)

    grades = rng.normal(loc=expected_grade, scale=15, size=grades_number)
    grades = [max(MIN_GRADE, min(MAX_GRADE, grade)) for grade in grades]  # type: ignore
    grades = [round(grade) for grade in grades]  # type: ignore
    return grades  # type: ignore


def get_mean_median_estimation_error(
    estimation_function, epsilon: float, experiments_number: int, seed=None, **kwargs
) -> float:
    rng = np.random.default_rng(seed)
    return get_mean_estimation_error(
        data_generator=lambda: get_random_grades(seed=rng),
        estimation_function=estimation_function,
        experiments_number=experiments_number,
        epsilon=epsilon,
        error_function=get_median_error,
        **kwargs,
    )


def plot_estimations(
    estimation_function, epsilon: float, experiments_number: int, seed=0, **kwargs
):
    """Plot distribution of median estimations."""
    grades = get_random_grades(seed=seed)
    exact_median = np.median(grades)
    median_estimations = [
        estimation_function(grades, epsilon=epsilon, **kwargs) for _ in range(experiments_number)
    ]

    plt.axvline(exact_median, color='green', label='exact median')
    plt.hist(
        median_estimations, density=True, color='blue', label='median estimations', bins=range(101)
    )
    plt.xlim(MIN_GRADE, MAX_GRADE)
    plt.title("Median Estimations Distribution")
    plt.legend()
    plt.show()


def generate_pairings(n):
    pairings = set()

    def backtrack(pairing, remaining_indices):
        if len(pairing) == n // 2:
            pairings.add(tuple(sorted(tuple(pair) for pair in pairing)))
            return
        for i in range(len(remaining_indices)):
            for j in range(i + 1, len(remaining_indices)):
                new_pairing = pairing + [[remaining_indices[i], remaining_indices[j]]]
                new_remaining = (
                    remaining_indices[:i]
                    + remaining_indices[i + 1 : j]
                    + remaining_indices[j + 1 :]
                )
                backtrack(new_pairing, new_remaining)

    backtrack([], list(range(n)))
    return [list(pairing) for pairing in pairings]

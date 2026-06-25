"""Above Threshold mechanism implementation."""

from typing import Callable, List

import numpy as np
from matplotlib import pyplot as plt

from libdpy.privacy_mechanisms.noise import laplace_noise


def above_threshold(
    database: list,
    queries: List[Callable],
    threshold: float,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> List[bool]:
    """
    This function implements the Above Threshold mechanism.
    :param database: a sensitive database
    :param queries: a list of queries with sensitivity=1, which are functions that gets a database as input and returns a number
    :param threshold: the threshold value for the queries results
    :param epsilon: the privacy parameter
    :param rng: optional NumPy generator for deterministic noise; defaults to fresh randomness.
    :return: a list of boolean responses - an estimation for whether the results pass the threshold.
    The list ends with the first positive result, if exists.
    """
    if rng is None:
        rng = np.random.default_rng()
    responses = []
    noised_threshold = threshold + laplace_noise(2 / epsilon, rng=rng)
    for query in queries:
        noised_response = query(database) + laplace_noise(4 / epsilon, rng=rng)
        if noised_response >= noised_threshold:
            responses.append(True)
            return responses
        else:
            responses.append(False)
    return responses


def simulate_above_threshold(epsilon_list, p, T, num_trials, n, m, seed=None):
    rng = np.random.default_rng(seed)

    # Set up a grid of subplots for visualizing histograms (one for each epsilon).
    fig, axs = plt.subplots(1, len(epsilon_list), figsize=(20, 5))

    # Step 1: Generate the dataset
    # Create a binary matrix X where each element represents whether a student attends a class (1) or not (0).
    X = rng.binomial(1, p, (n, m))

    # Define the queries: Each query computes the sum of a specific column (class attendance count).
    queries = [(lambda dataset, j=i: np.sum(dataset[:, j])) for i in range(m)]

    # Compute the true (noiseless) sum of students for each class (column).
    sum_students = np.sum(X, axis=0)

    # Determine which classes exceed the threshold T based on the noiseless data.
    bits_Above_threshold = sum_students > T

    # Iterate through each epsilon value to perform experiments and record results.
    for idx, epsilon in enumerate(epsilon_list):
        # Initialize lists to collect results for the current epsilon.
        halt_iterations = []  # The iteration where the algorithm halts.
        true_halt_values = []  # The true (noiseless) value at the halt point.
        correct_halts = []  # Whether we halted when we were supposed to.
        missed_halts = []  # How many times we should have halted but didn’t.

        # Repeat the experiment for the given number of trials.
        for _ in range(num_trials):
            # Run the AboveThreshold algorithm on the dataset with the given threshold and privacy level.
            T_F_list = above_threshold(X, queries, T, epsilon, rng=rng)

            # Record the halt iteration (length of T_F_list).
            halt_iteration = len(T_F_list)
            halt_iterations.append(halt_iteration)

            # Record the true (noiseless) value at the halt iteration.
            true_halt_values.append(sum_students[halt_iteration - 1])

            # Check if the halt decision was correct (true value exceeds the threshold).
            correct_halts.append(true_halt_values[-1] > T)

            # Calculate missed halts: How many times the true value exceeded T before halting.
            missed_halts.append(np.sum(bits_Above_threshold[:halt_iteration]))

        # Step 3: Visualization - Plot a histogram for the differences between true values of the queries and the threshold.
        axs[idx].hist(np.array(true_halt_values) - T, bins=20)
        axs[idx].set_xlabel("True value - threshold")
        axs[idx].set_ylabel("Frequency")
        axs[idx].set_title(f"Epsilon = {epsilon}")

        # Step 4: Print statistics for the current epsilon.
        print(f"Epsilon = {epsilon}")
        print(f"Mean halt iteration: {np.mean(halt_iterations)}")
        print(f"Mean true halt value: {np.mean(true_halt_values)}")
        print(f"Mean correct halts: {np.mean(correct_halts)}")
        print(f"Mean missed halts: {np.mean(missed_halts)}")
        print("\n")

        # step 5: return the results
        # return halt_iterations, true_halt_values, correct_halts, missed_halts ### you can use this if needed

    # Adjust layout and display the histograms.
    plt.tight_layout()
    plt.show()

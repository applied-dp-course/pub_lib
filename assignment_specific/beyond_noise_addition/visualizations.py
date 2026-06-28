from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def make_effect_of_new_student_figure(
    students_heights: list,
    new_student_height: float,
    dp_function: Callable,
    epsilon: float,
    experiments_number: int,
    compute_exact_height_of_shortest: Callable,
):
    students_heights_with_new = students_heights + [new_student_height]

    exact_height_of_shortest_without_new = compute_exact_height_of_shortest(students_heights)
    exact_height_of_shortest_with_new = compute_exact_height_of_shortest(students_heights_with_new)

    dp_height_of_shortest_without_new = [
        dp_function(students_heights, epsilon) for _ in range(experiments_number)
    ]
    dp_height_of_shortest_with_new = [
        dp_function(students_heights_with_new, epsilon) for _ in range(experiments_number)
    ]

    bins = np.linspace(1.47, 2.12, 14)
    fig, ax = plt.subplots()
    ax.hist(
        dp_height_of_shortest_without_new,
        alpha=0.7,
        label="DP result - w.o. the student",
        color="blue",
        bins=bins,
    )
    ax.hist(
        dp_height_of_shortest_with_new,
        alpha=0.7,
        label="DP result - w. the student",
        color="orange",
        bins=bins,
    )
    ax.axvline(
        x=exact_height_of_shortest_without_new,
        label="exact result - w.o. the student",
        color="blue",
    )
    ax.axvline(
        x=exact_height_of_shortest_with_new, label="exact result - w. the student", color="orange"
    )
    ax.set_title(f"Results Distribution - epsilon={epsilon}")
    ax.set_xlabel("Height of Shortest")
    ax.set_ylabel("Number of Results")
    ax.legend(loc="upper left")
    return fig


def make_results_figure(Binary_search_estimate, Naive_Noise_addition_estimate, ReportNoisyMax_median, X):
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    axs[0].hist(Binary_search_estimate, bins=50, alpha=0.5, label='Binary Search Estimate')
    axs[0].set_xlabel('Estimate')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Binary Search Estimate Distribution')
    axs[0].axvline(x=np.median(X), color='r', linestyle='--', label='True Value')

    axs[1].hist(
        Naive_Noise_addition_estimate, bins=50, alpha=0.5, label='Naive Noise Addition Estimate'
    )
    axs[1].set_xlabel('Estimate')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Naive Noise Addition Estimate Distribution')
    axs[1].axvline(x=np.median(X), color='r', linestyle='--', label='True Value')

    axs[2].hist(ReportNoisyMax_median, bins=50, alpha=0.5, label='ReportNoisyMax Estimate')
    axs[2].set_xlabel('Estimate')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('ReportNoisyMax Estimate Distribution')
    axs[2].axvline(x=np.median(X), color='r', linestyle='--', label='True Value')

    fig.tight_layout()
    return fig


def make_exponential_noise_figure(values, epsilon, seed=None):
    rng = np.random.default_rng(seed)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')

    noise = rng.exponential(scale=2 / epsilon, size=len(values))
    noisy_values = np.array(values) + noise
    x = np.arange(len(values))

    orig_bars = ax.bar(x - 0.2, values, width=0.4, color='blue', alpha=0.7, label='Original')
    ax.bar(x + 0.2, noisy_values, width=0.4, color='red', alpha=0.5, label='Noisy')

    tallest_index = np.argmax(values)
    orig_bars[tallest_index].set_color('green')

    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title(f'Exponential Noise Visualization (ε = {epsilon:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(range(len(values)))
    ax.legend()
    fig.tight_layout()
    return fig

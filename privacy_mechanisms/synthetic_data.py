"""Synthetic data generation utilities."""

import numpy as np
import pandas as pd


def generate_private_gaussian_data(n, mu, sd, epsilon, seed=None):
    rng = np.random.default_rng(seed)

    # Generate original data
    data = rng.normal(mu, sd, n)

    # Add Laplace noise to the mean estimate
    private_mu_hat = np.mean(data) + rng.laplace(scale=(1 / n) / epsilon)

    # Generate synthetic data using the noisy mean
    private_data = rng.normal(private_mu_hat, sd, n)

    return data, private_data, private_mu_hat


def create_noisy_histogram(data, epsilon, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    columns = data.columns
    n = data.shape[0]

    # Create noisy counts for each column
    final_data = {}
    for column in columns:
        noisy_counts = {}
        values = data[column].unique()

        for value in values:
            # Calculate true proportion
            count = data[data[column] == value].shape[0] / n
            # Add Laplace noise
            noisy_counts[value] = count + rng.laplace(scale=1 / (epsilon * n))

        # Post-process: ensure non-negative and normalize
        for value in noisy_counts.keys():
            if noisy_counts[value] < 0:
                noisy_counts[value] = 0
            if noisy_counts[value] > 1:
                noisy_counts[value] = 1

        # Normalize to ensure probabilities sum to 1
        sum_col = sum(noisy_counts.values())
        if sum_col > 0:
            for value in noisy_counts.keys():
                noisy_counts[value] = noisy_counts[value] / sum_col

        final_data[column] = noisy_counts

    return final_data


def sample_synthetic_data(noisy_counts, n, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    synthetic_data = pd.DataFrame()

    for column, counts in noisy_counts.items():
        values = list(counts.keys())
        probs = list(counts.values())

        # Sample from the noisy distribution
        new_column = rng.choice(values, size=n, p=probs)
        synthetic_data[column] = new_column

    return synthetic_data


def calculate_domain_size(data):
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to dictionary format
        data_dict = {col: data[col].unique() for col in data.columns}
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise ValueError("Data must be a DataFrame or dictionary")

    num_combinations = 1
    for column_values in data_dict.values():
        num_combinations *= len(column_values)

    return num_combinations


def generate_private_synthetic_dataset(data, epsilon_per_column=0.1, seed=None):
    rng = np.random.default_rng(seed)

    # Create noisy histogram
    noisy_counts = create_noisy_histogram(data, epsilon_per_column, rng=rng)

    # Generate synthetic data with same size as original
    synthetic_data = sample_synthetic_data(noisy_counts, len(data), rng=rng)

    # Calculate domain size for reference
    domain_size = calculate_domain_size(data)

    return synthetic_data, noisy_counts, domain_size


def compare_datasets(original, synthetic, columns_to_compare=None):
    if columns_to_compare is None:
        columns_to_compare = original.columns

    comparison = {}

    for column in columns_to_compare:
        if column in original.columns and column in synthetic.columns:
            # Value counts for both datasets
            orig_counts = original[column].value_counts(normalize=True).sort_index()
            synth_counts = synthetic[column].value_counts(normalize=True).sort_index()

            # Align the indices
            all_values = sorted(set(orig_counts.index) | set(synth_counts.index))
            orig_aligned = orig_counts.reindex(all_values, fill_value=0)
            synth_aligned = synth_counts.reindex(all_values, fill_value=0)

            # Calculate metrics
            comparison[column] = {
                'original_distribution': orig_aligned.to_dict(),
                'synthetic_distribution': synth_aligned.to_dict(),
                'total_variation_distance': 0.5 * np.sum(np.abs(orig_aligned - synth_aligned)),
                'unique_values_original': len(orig_counts),
                'unique_values_synthetic': len(synth_counts),
            }

    return comparison

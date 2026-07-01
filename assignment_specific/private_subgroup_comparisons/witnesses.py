"""Witness datasets and neighbor pairs for private subgroup comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from libdpy.assignment_specific.private_estimation.utils import (
    DEFAULT_SEED,
    extract_income,
    load_fulton,
)
from libdpy.assignment_specific.private_subgroup_comparisons.mechanisms import (
    GROUP_A,
    GROUP_B,
    subgroup_counts,
)

PUBLIC_CLIP_LOWER = 0.0
PUBLIC_CLIP_UPPER = 141_450.0
PUBLIC_REFERENCE_MEAN = 33_423.95303888846
PUBLIC_VALUE_UPPER = PUBLIC_CLIP_UPPER / PUBLIC_REFERENCE_MEAN
DEFAULT_TAU = 5.0
DEFAULT_SUPPORT_THRESHOLD = 30
DEFAULT_EPS_TOTAL = 1.0
DEFAULT_EPS_COUNT = 0.25
DEFAULT_EPS_SUM = 0.25


def scale_clipped_salary_by_reference_mean(
    y: np.ndarray,
    L: float,
    U: float,
    reference_mean: float,
) -> np.ndarray:
    """Clip salary dollars and scale them by a public reference mean."""

    if U <= L:
        raise ValueError("clip upper bound must exceed lower bound")
    if reference_mean <= 0:
        raise ValueError("reference mean must be positive")
    clipped = np.clip(np.asarray(y, dtype=float), L, U)
    return clipped / reference_mean


def age_group_labels(age: np.ndarray) -> np.ndarray:
    """Public grouping: workers under 50 vs age 50 and above."""

    age = np.asarray(age, dtype=float)
    return np.where(age < 50, GROUP_A, GROUP_B)


def sex_group_labels(sex: np.ndarray) -> np.ndarray:
    """Public grouping: the dataset's two sex-code categories."""

    sex = np.asarray(sex, dtype=int)
    return np.where(sex == 0, GROUP_A, GROUP_B)


def latino_group_labels(latino: np.ndarray) -> np.ndarray:
    """Public grouping: Latino workers versus everyone else (~5% in Fulton)."""

    latino = np.asarray(latino, dtype=int)
    return np.where(latino == 1, GROUP_B, GROUP_A)


def prepare_fulton_subgroup_frame(
    *,
    n_rows: int = 1000,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return a Fulton subset with mean-scaled salary and public subgroup labels."""

    source = load_fulton()
    if n_rows < len(source):
        df = source.sample(n=n_rows, random_state=seed).sort_index().copy()
    else:
        df = source.head(n_rows).copy()
    income = extract_income(df)
    df["x"] = scale_clipped_salary_by_reference_mean(
        income,
        PUBLIC_CLIP_LOWER,
        PUBLIC_CLIP_UPPER,
        PUBLIC_REFERENCE_MEAN,
    )
    df["sex_group"] = sex_group_labels(df["sex"].to_numpy(dtype=int))
    df["latino_group"] = latino_group_labels(df["latino"].to_numpy(dtype=int))
    df["age_group"] = age_group_labels(df["age"].to_numpy(dtype=float))
    df["group"] = df["sex_group"]
    df["income"] = income
    return df


def frame_to_arrays(
    frame: pd.DataFrame,
    *,
    group_column: str = "group",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean-scaled values and group labels from a prepared frame."""

    return frame["x"].to_numpy(dtype=float), frame[group_column].to_numpy()


def build_balanced_subgroup_sample(
    n_per_group: int = 500,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic balanced sample with moderate group means on a bounded scale."""

    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    groups = np.array([GROUP_A] * n_per_group + [GROUP_B] * n_per_group)
    x = np.empty(n, dtype=float)
    x[groups == GROUP_A] = rng.uniform(0.35, 0.45, size=n_per_group)
    x[groups == GROUP_B] = rng.uniform(0.55, 0.65, size=n_per_group)
    perm = rng.permutation(n)
    return x[perm], groups[perm]


def build_imbalanced_subgroup_sample(
    *,
    n_large: int = 900,
    n_small: int = 100,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic imbalanced sample."""

    rng = np.random.default_rng(seed)
    groups = np.array([GROUP_A] * n_large + [GROUP_B] * n_small)
    x = np.empty(n_large + n_small, dtype=float)
    x[groups == GROUP_A] = rng.uniform(0.35, 0.45, size=n_large)
    x[groups == GROUP_B] = rng.uniform(0.55, 0.65, size=n_small)
    perm = rng.permutation(len(groups))
    return x[perm], groups[perm]


def build_sparse_subgroup_sample(
    *,
    n_large: int = 990,
    n_small: int = 10,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Engineered sparse sample for denominator fragility demos."""

    return build_imbalanced_subgroup_sample(
        n_large=n_large,
        n_small=n_small,
        seed=seed,
    )


def build_oracle_ls_failure_neighbor_pair(
    *,
    m_small: int = 3,
    n_large: int = 500,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, int], dict[str, int]]:
    """Neighbor pair where one replacement removes a row from the smaller group.

    Returns ``(x, groups, x_prime, groups_prime, out_idx, counts_D, counts_D_prime)``.
    """

    if m_small < 2:
        raise ValueError("m_small must be at least 2")
    rng = np.random.default_rng(seed)
    groups = np.array([GROUP_B] * m_small + [GROUP_A] * n_large)
    x = np.empty(m_small + n_large, dtype=float)
    x[groups == GROUP_B] = rng.uniform(0.55, 0.65, size=m_small)
    x[groups == GROUP_A] = rng.uniform(0.35, 0.45, size=n_large)
    out_idx = int(np.where(groups == GROUP_B)[0][0])
    groups_prime = groups.copy()
    x_prime = x.copy()
    groups_prime[out_idx] = GROUP_A
    x_prime[out_idx] = 0.40
    return (
        x,
        groups,
        x_prime,
        groups_prime,
        out_idx,
        subgroup_counts(groups),
        subgroup_counts(groups_prime),
    )


def support_threshold_grid(*, start: int = 5, stop: int = 80, step: int = 5) -> np.ndarray:
    """Public grid of support thresholds for PTR/smooth comparison figures."""

    return np.arange(start, stop + 1, step, dtype=int)

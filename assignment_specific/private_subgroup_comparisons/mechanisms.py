"""Mechanisms for private subgroup mean-difference comparisons on normalized salaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from libdpy.privacy_mechanisms.noise import laplace_noise

GROUP_A = "young"
GROUP_B = "older"
GLOBAL_SENSITIVITY_NORMALIZED = 2.0
DEFAULT_DENOMINATOR_FLOOR = 1.0
COUNT_SENSITIVITY = 1.0
SUM_SENSITIVITY = 1.0


def normalize_clipped_salary(y: np.ndarray, L: float, U: float) -> np.ndarray:
    """Map clipped salaries into ``[0, 1]`` using public bounds ``[L, U]``."""

    if U <= L:
        raise ValueError("clip upper bound must exceed lower bound")
    clipped = np.clip(np.asarray(y, dtype=float), L, U)
    return (clipped - L) / (U - L)


def subgroup_counts(
    groups: np.ndarray,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, int]:
    """Return exact counts for the two comparison groups."""

    groups = np.asarray(groups)
    return {
        "A": int(np.sum(groups == group_a)),
        "B": int(np.sum(groups == group_b)),
    }


def subgroup_sums(
    x: np.ndarray,
    groups: np.ndarray,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, float]:
    """Return normalized sum queries for each group."""

    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    return {
        "A": float(np.sum(x[groups == group_a])),
        "B": float(np.sum(x[groups == group_b])),
    }


def subgroup_difference(
    x: np.ndarray,
    groups: np.ndarray,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> float:
    """Difference of group means on normalized values; raises if either group is empty."""

    counts = subgroup_counts(groups, group_a, group_b)
    if counts["A"] == 0 or counts["B"] == 0:
        raise ValueError("subgroup_difference requires both groups to be nonempty")
    sums = subgroup_sums(x, groups, group_a, group_b)
    return sums["A"] / counts["A"] - sums["B"] / counts["B"]


def subgroup_difference_total(
    x: np.ndarray,
    groups: np.ndarray,
    tau: float = DEFAULT_DENOMINATOR_FLOOR,
    *,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> float:
    """Total-function subgroup difference using a public denominator floor ``tau``."""

    if tau <= 0:
        raise ValueError("denominator floor tau must be positive")
    counts = subgroup_counts(groups, group_a, group_b)
    sums = subgroup_sums(x, groups, group_a, group_b)
    mu_a = sums["A"] / max(counts["A"], tau)
    mu_b = sums["B"] / max(counts["B"], tau)
    return mu_a - mu_b


def replacement_ls_bound(
    count_a: int,
    count_b: int,
    *,
    value_range: float = 1.0,
) -> float:
    """Conservative replacement-adjacency local sensitivity bound for a mean difference."""

    return value_range * (
        1.0 / max(count_a - 1, 1) + 1.0 / max(count_b - 1, 1)
    )


def _release_dict(
    estimate: float,
    *,
    privacy_status: str,
    epsilon: dict[str, float] | float,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "estimate": float(estimate),
        "privacy_status": privacy_status,
        "epsilon": epsilon,
    }
    payload.update(extra)
    return payload


def global_sensitivity_release(
    x: np.ndarray,
    groups: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    tau: float = DEFAULT_DENOMINATOR_FLOOR,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """Laplace release with the pessimistic global sensitivity bound ``GS <= 2``.

    Uses :func:`subgroup_difference_total` so the statistic is defined on neighboring
    datasets where a group may become empty.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    delta = subgroup_difference_total(x, groups, tau, group_a=group_a, group_b=group_b)
    noise_scale = GLOBAL_SENSITIVITY_NORMALIZED / epsilon
    noisy = delta + laplace_noise(noise_scale, rng=rng)
    counts = subgroup_counts(groups, group_a, group_b)
    return _release_dict(
        noisy,
        privacy_status="valid",
        epsilon={"release": epsilon},
        true_delta=delta,
        noise_scale=noise_scale,
        counts=counts,
        tau=tau,
        mechanism="global_sensitivity",
    )


def oracle_local_sensitivity_release(
    x: np.ndarray,
    groups: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """Scale noise by the true local sensitivity — diagnostic only, not DP."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    counts = subgroup_counts(groups, group_a, group_b)
    ls = replacement_ls_bound(counts["A"], counts["B"])
    delta = subgroup_difference(x, groups, group_a, group_b)
    noise_scale = ls / epsilon
    noisy = delta + laplace_noise(noise_scale, rng=rng)
    return _release_dict(
        noisy,
        privacy_status="oracle diagnostic",
        epsilon={"release": epsilon},
        true_delta=delta,
        local_sensitivity=ls,
        noise_scale=noise_scale,
        counts=counts,
        mechanism="oracle_local_sensitivity",
    )


def noisy_count_sum_release(
    x: np.ndarray,
    groups: np.ndarray,
    eps_n_a: float,
    eps_n_b: float,
    eps_s_a: float,
    eps_s_b: float,
    tau: float,
    rng: np.random.Generator,
    *,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """Release four Laplace queries and post-process into a private mean difference."""

    eps_values = (eps_n_a, eps_n_b, eps_s_a, eps_s_b)
    if any(eps <= 0 for eps in eps_values):
        raise ValueError("all epsilon budget components must be positive")
    if tau <= 0:
        raise ValueError("denominator floor tau must be positive")

    counts = subgroup_counts(groups, group_a, group_b)
    sums = subgroup_sums(x, groups, group_a, group_b)
    noisy_n_a = counts["A"] + laplace_noise(COUNT_SENSITIVITY / eps_n_a, rng=rng)
    noisy_n_b = counts["B"] + laplace_noise(COUNT_SENSITIVITY / eps_n_b, rng=rng)
    noisy_s_a = sums["A"] + laplace_noise(SUM_SENSITIVITY / eps_s_a, rng=rng)
    noisy_s_b = sums["B"] + laplace_noise(SUM_SENSITIVITY / eps_s_b, rng=rng)
    mu_a = noisy_s_a / max(noisy_n_a, tau)
    mu_b = noisy_s_b / max(noisy_n_b, tau)
    noisy_delta = mu_a - mu_b
    epsilon_total = float(sum(eps_values))
    return _release_dict(
        noisy_delta,
        privacy_status="valid",
        epsilon={
            "n_A": eps_n_a,
            "n_B": eps_n_b,
            "S_A": eps_s_a,
            "S_B": eps_s_b,
            "total": epsilon_total,
        },
        counts=counts,
        sums=sums,
        noisy_counts={"A": noisy_n_a, "B": noisy_n_b},
        noisy_sums={"A": noisy_s_a, "B": noisy_s_b},
        noisy_means={"A": mu_a, "B": mu_b},
        tau=tau,
        mechanism="noisy_count_sum",
    )


def ptr_support_release(
    x: np.ndarray,
    groups: np.ndarray,
    m: int,
    eps_test: float,
    eps_release: float,
    delta: float,
    rng: np.random.Generator,
    *,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """Conceptual PTR: test distance from insufficient-support datasets, then release or abstain.

    The test adds Laplace noise to the distance ``max(min(n_A, n_B) - m + 1, 0)``.
    If the noisy distance clears a public buffer, release ``Delta`` with sensitivity
    ``2 / (m - 1)`` on the normalized scale. Otherwise abstain.
    """

    if m < 2:
        raise ValueError("support threshold m must be at least 2")
    if eps_test <= 0 or eps_release <= 0:
        raise ValueError("epsilon budgets must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")

    counts = subgroup_counts(groups, group_a, group_b)
    min_count = min(counts["A"], counts["B"])
    distance = max(min_count - m + 1, 0)
    test_sensitivity = 1.0
    noisy_distance = distance + laplace_noise(test_sensitivity / eps_test, rng=rng)
    buffer = np.log(2.0 / delta) / eps_test
    accepted = noisy_distance > buffer
    release_sensitivity = 2.0 / (m - 1)
    if accepted:
        delta_stat = subgroup_difference(x, groups, group_a, group_b)
        estimate = delta_stat + laplace_noise(release_sensitivity / eps_release, rng=rng)
        status = "released"
    else:
        estimate = float("nan")
        status = "abstained"

    return _release_dict(
        estimate,
        privacy_status="conceptual PTR",
        epsilon={"test": eps_test, "release": eps_release, "total": eps_test + eps_release},
        counts=counts,
        min_count=min_count,
        distance=distance,
        noisy_distance=noisy_distance,
        buffer=buffer,
        accepted=accepted,
        release_sensitivity=release_sensitivity,
        status=status,
        mechanism="ptr_support",
        delta_failure=delta,
    )


def smooth_sensitivity_bound(min_count: int, beta: float) -> float:
    """Conservative smooth-sensitivity envelope on the normalized scale."""

    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if beta <= 0:
        raise ValueError("beta must be positive")

    bound = 0.0
    for k in range(min_count + 2):
        envelope = 2.0 / max(min_count - k - 1, 1)
        bound = max(bound, np.exp(-beta * k) * envelope)
    return float(bound)


def conceptual_smooth_sensitivity_release(
    x: np.ndarray,
    groups: np.ndarray,
    epsilon: float,
    beta: float,
    rng: np.random.Generator,
    *,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """Conceptual smooth-sensitivity comparison — labeled, not a formal pure-DP guarantee."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    counts = subgroup_counts(groups, group_a, group_b)
    min_count = min(counts["A"], counts["B"])
    smooth_bound = smooth_sensitivity_bound(min_count, beta)
    delta = subgroup_difference(x, groups, group_a, group_b)
    noisy = delta + laplace_noise(smooth_bound / epsilon, rng=rng)
    local_bound = replacement_ls_bound(counts["A"], counts["B"])
    return _release_dict(
        noisy,
        privacy_status="conceptual smooth sensitivity",
        epsilon={"release": epsilon},
        true_delta=delta,
        local_sensitivity=local_bound,
        smooth_sensitivity=smooth_bound,
        beta=beta,
        counts=counts,
        mechanism="conceptual_smooth_sensitivity",
    )

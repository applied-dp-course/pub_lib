"""Static figure builders for the private subgroup comparisons lecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from libdpy.assignment_specific.private_estimation.lecture_figures import (
    make_accuracy_leaderboard_figure,
)
from libdpy.assignment_specific.private_estimation.utils import (
    DEFAULT_SEED,
)
from libdpy.assignment_specific.private_estimation.visualizations import (
    PROVENANCE_PALETTE,
    adaptive_histogram_bin_edges,
    add_histogram_density_line,
)
from libdpy.privacy_mechanisms.above_threshold import above_threshold
from libdpy.visualization.plot_styles import MPL_BOUND, MPL_PRIMARY, MPL_REFERENCE, MPL_SECONDARY
from libdpy.assignment_specific.private_subgroup_comparisons.mechanisms import (
    conceptual_smooth_sensitivity_release,
    noisy_count_sum_release,
    ptr_abstention_probability,
    ptr_false_accept_probability,
    ptr_support_distance,
    ptr_support_release,
    ptr_test_buffer,
    subgroup_counts,
    subgroup_difference,
)
from libdpy.assignment_specific.private_subgroup_comparisons.witnesses import (
    DEFAULT_EPS_COUNT,
    DEFAULT_EPS_SUM,
    DEFAULT_EPS_TOTAL,
    DEFAULT_SUPPORT_THRESHOLD,
    DEFAULT_TAU,
    GROUP_A,
    GROUP_B,
    PUBLIC_VALUE_UPPER,
    age_group_labels,
    build_balanced_subgroup_sample,
    build_imbalanced_subgroup_sample,
    build_oracle_ls_failure_neighbor_pair,
    frame_to_arrays,
    prepare_fulton_subgroup_frame,
)

_DPI = 100
DEFAULT_BETA = 0.35
ORACLE_EPS = 1.0
PTR_DELTA = 0.05
_SUBGROUP_LEADERBOARD_KWARGS = {
    "estimate_target": "subgroup_difference",
    "error_metric": "value",
    "x_label": "signed error in normalized average salary gap",
    "std_fmt": ".4f",
    "x_tick_fmt": ".3f",
    "round_decimals": 3,
}


@dataclass(frozen=True)
class SubgroupSamplingArtifact:
    """Sampling distribution for the non-private subgroup statistic."""

    comparison_label: str
    population_size: int
    population_gap: float
    sample_gaps: np.ndarray
    sample_size: int
    n_samples: int


@dataclass(frozen=True)
class OracleScaleLeakArtifact:
    """Neighbor pair where oracle local-sensitivity noise changes with support."""

    x: np.ndarray
    groups: np.ndarray
    x_prime: np.ndarray
    groups_prime: np.ndarray
    out_idx: int
    counts_d: dict[str, int]
    counts_d_prime: dict[str, int]
    epsilon: float


@dataclass(frozen=True)
class CountSumArtifact:
    """Accuracy of noisy count/sum releases under explicit budget splits."""

    x: np.ndarray
    groups: np.ndarray
    tau: float
    eps_total: float
    budget_rows: list[dict[str, float]]
    rmse_by_split: dict[str, float]


@dataclass(frozen=True)
class SupportComparisonArtifact:
    """Repair behavior as true minimum subgroup support varies."""

    min_support_values: np.ndarray
    count_sum_error: np.ndarray
    ptr_abstention: np.ndarray
    ptr_error: np.ndarray
    ptr_release_counts: np.ndarray
    smooth_error: np.ndarray
    ptr_threshold: int
    beta: float


@dataclass(frozen=True)
class AboveThresholdSupportArtifact:
    """Real AboveThreshold run over a public coarsening hierarchy."""

    coarsening_labels: list[str]
    support_values: list[float]
    threshold: float
    halt_index: int
    noisy_prefix: list[bool]
    epsilon: float


@dataclass(frozen=True)
class PtrFailureArtifact:
    """Analytic PTR abstention/failure probabilities vs true minimum support."""

    support_values: np.ndarray
    abstention_probability: np.ndarray
    distance_values: np.ndarray
    ptr_threshold: int
    eps_test: float
    delta: float
    buffer: float
    false_accept_at_zero: float


def build_subgroup_sampling_artifact(
    population_values: np.ndarray,
    population_groups: np.ndarray,
    *,
    sample_size: int,
    n_samples: int,
    comparison_label: str = "subgroup comparison",
    seed: int = DEFAULT_SEED,
) -> SubgroupSamplingArtifact:
    """Resample fixed-size datasets and compute the subgroup gap on each draw."""

    population_values = np.asarray(population_values, dtype=float)
    population_groups = np.asarray(population_groups)
    population_size = len(population_values)
    if sample_size > population_size:
        raise ValueError("sample_size cannot exceed the population size")
    population_gap = subgroup_difference(population_values, population_groups)
    rng = np.random.default_rng(seed)
    sample_gaps = []
    for _ in range(n_samples):
        idx = rng.integers(0, population_size, size=sample_size)
        sample_gaps.append(
            subgroup_difference(population_values[idx], population_groups[idx])
        )
    return SubgroupSamplingArtifact(
        comparison_label=comparison_label,
        population_size=population_size,
        population_gap=population_gap,
        sample_gaps=np.asarray(sample_gaps, dtype=float),
        sample_size=sample_size,
        n_samples=n_samples,
    )


def build_oracle_ls_failure_artifact(seed: int = DEFAULT_SEED) -> OracleScaleLeakArtifact:
    """Sparse neighbor pair where one replacement changes the oracle noise scale."""

    x, groups, x_prime, groups_prime, out_idx, counts_d, counts_d_prime = (
        build_oracle_ls_failure_neighbor_pair(seed=seed)
    )
    return OracleScaleLeakArtifact(
        x=x,
        groups=groups,
        x_prime=x_prime,
        groups_prime=groups_prime,
        out_idx=out_idx,
        counts_d=counts_d,
        counts_d_prime=counts_d_prime,
        epsilon=ORACLE_EPS,
    )


def build_typical_oracle_neighbor_pair(
    *,
    seed: int,
    n_per_group: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    """Balanced neighbor pair where oracle-LS Gaussian scale barely changes."""

    x, groups = build_balanced_subgroup_sample(n_per_group=n_per_group, seed=seed)
    out_idx = int(np.where(groups == GROUP_B)[0][0])
    x_prime = x.copy()
    groups_prime = groups.copy()
    groups_prime[out_idx] = GROUP_A
    x_prime[out_idx] = float(np.median(x[groups == GROUP_A]))
    return (
        x,
        groups,
        x_prime,
        groups_prime,
        subgroup_counts(groups),
        subgroup_counts(groups_prime),
    )


def build_count_sum_artifact(
    x: np.ndarray,
    groups: np.ndarray,
    *,
    eps_total: float,
    tau: float,
    seed: int = DEFAULT_SEED,
    n_trials: int = 200,
) -> CountSumArtifact:
    """Evaluate noisy count/sum accuracy under explicit budget splits."""

    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    target = subgroup_difference(x, groups)
    splits = {
        "half counts / half sums": (0.25, 0.25, 0.25, 0.25),
    }
    rmse_by_split: dict[str, float] = {}
    budget_rows: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for label, split in splits.items():
        eps_n_a, eps_n_b, eps_s_a, eps_s_b = [eps_total * w for w in split]
        budget_rows.append(
            {
                "split": label,
                "eps_n_A": eps_n_a,
                "eps_n_B": eps_n_b,
                "eps_S_A": eps_s_a,
                "eps_S_B": eps_s_b,
                "eps_total": eps_n_a + eps_n_b + eps_s_a + eps_s_b,
            }
        )
        estimates = []
        for _ in range(n_trials):
            run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            result = noisy_count_sum_release(
                x,
                groups,
                eps_n_a,
                eps_n_b,
                eps_s_a,
                eps_s_b,
                tau,
                run_rng,
                value_bound=PUBLIC_VALUE_UPPER,
            )
            estimates.append(result["estimate"])
        rmse_by_split[label] = float(np.sqrt(np.mean((np.asarray(estimates) - target) ** 2)))
    return CountSumArtifact(
        x=x,
        groups=groups,
        tau=tau,
        eps_total=eps_total,
        budget_rows=budget_rows,
        rmse_by_split=rmse_by_split,
    )


def build_support_comparison_artifact(
    *,
    min_support_values: np.ndarray,
    ptr_threshold: int,
    n_large: int,
    n_trials: int,
    eps_total: float,
    tau: float,
    seed: int = DEFAULT_SEED,
) -> SupportComparisonArtifact:
    """Compare repairs as true minimum group support varies."""

    min_support_values = np.asarray(min_support_values, dtype=int)
    rng = np.random.default_rng(seed)
    count_sum_error: list[float] = []
    ptr_abstention: list[float] = []
    ptr_error: list[float] = []
    ptr_release_counts: list[int] = []
    smooth_error: list[float] = []
    eps_n = eps_total / 4.0
    eps_s = eps_total / 4.0

    for idx, n_small in enumerate(min_support_values):
        x, groups = build_imbalanced_subgroup_sample(
            n_large=n_large,
            n_small=int(n_small),
            seed=seed + idx,
        )
        target = subgroup_difference(x, groups)
        count_sum_estimates = []
        ptr_estimates = []
        smooth_estimates = []
        abstained = 0
        for _ in range(n_trials):
            count_sum_result = noisy_count_sum_release(
                x,
                groups,
                eps_n,
                eps_n,
                eps_s,
                eps_s,
                tau,
                np.random.default_rng(rng.integers(0, 2**32 - 1)),
                value_bound=PUBLIC_VALUE_UPPER,
            )
            count_sum_estimates.append(count_sum_result["estimate"])

            ptr_result = ptr_support_release(
                x,
                groups,
                ptr_threshold,
                eps_test=0.4 * eps_total,
                eps_release=0.6 * eps_total,
                delta=PTR_DELTA,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
                value_bound=PUBLIC_VALUE_UPPER,
            )
            if not ptr_result["accepted"]:
                abstained += 1
            else:
                ptr_estimates.append(ptr_result["estimate"])

            smooth_result = conceptual_smooth_sensitivity_release(
                x,
                groups,
                epsilon=eps_total,
                beta=DEFAULT_BETA,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
                value_bound=PUBLIC_VALUE_UPPER,
            )
            smooth_estimates.append(smooth_result["estimate"])

        count_sum_error.append(
            float(np.mean(np.abs(np.asarray(count_sum_estimates) - target)))
        )
        ptr_abstention.append(abstained / n_trials)
        ptr_release_counts.append(len(ptr_estimates))
        ptr_error.append(
            float(np.mean(np.abs(np.asarray(ptr_estimates) - target)))
            if ptr_estimates
            else float("nan")
        )
        smooth_error.append(float(np.mean(np.abs(np.asarray(smooth_estimates) - target))))

    return SupportComparisonArtifact(
        min_support_values=min_support_values,
        count_sum_error=np.asarray(count_sum_error, dtype=float),
        ptr_abstention=np.asarray(ptr_abstention, dtype=float),
        ptr_error=np.asarray(ptr_error, dtype=float),
        ptr_release_counts=np.asarray(ptr_release_counts, dtype=int),
        smooth_error=np.asarray(smooth_error, dtype=float),
        ptr_threshold=ptr_threshold,
        beta=DEFAULT_BETA,
    )


def build_ptr_smooth_artifact(seed: int = DEFAULT_SEED) -> SupportComparisonArtifact:
    """Compatibility builder for older tests; uses explicit public defaults."""

    return build_support_comparison_artifact(
        min_support_values=np.array([3, 5, 8, 12, 20, 30, 50, 80, 120]),
        ptr_threshold=DEFAULT_SUPPORT_THRESHOLD,
        n_large=700,
        n_trials=120,
        eps_total=DEFAULT_EPS_TOTAL,
        tau=DEFAULT_TAU,
        seed=seed,
    )


def build_above_threshold_support_artifact(seed: int = DEFAULT_SEED) -> AboveThresholdSupportArtifact:
    """Run AboveThreshold on public coarsenings with queries that read the database."""

    frame = prepare_fulton_subgroup_frame(seed=seed)
    coarsening_labels = [
        "sex x educ x age bin",
        "sex x educ",
        "education",
        "age < 50",
    ]

    def min_group_count(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        return float(np.min(counts))

    def coarsening_arrays(db: list[dict[str, Any]]) -> list[np.ndarray]:
        records = pd.DataFrame.from_records(db)
        age = records["age"].to_numpy(dtype=float)
        educ = records["educ"].to_numpy(dtype=int)
        sex = records["sex"].to_numpy(dtype=int)
        age_bin = np.digitize(age, [25, 35, 45, 55, 65], right=True)
        age_groups = np.where(age_group_labels(age) == GROUP_B, 1, 0)
        return [
            sex * 10000 + educ * 100 + age_bin,
            sex * 100 + educ,
            educ,
            age_groups,
        ]

    threshold = 50.0
    database = frame.to_dict("records")
    support_values = [min_group_count(labels) for labels in coarsening_arrays(database)]
    queries = [
        (lambda db, level=level: min_group_count(coarsening_arrays(db)[level]))
        for level in range(len(coarsening_labels))
    ]

    epsilon = 0.5
    noisy_prefix = above_threshold(
        database,
        queries,
        threshold,
        epsilon,
        rng=np.random.default_rng(seed),
    )
    halt_index = (
        len(noisy_prefix) - 1
        if noisy_prefix and noisy_prefix[-1]
        else max(len(noisy_prefix) - 1, 0)
    )
    return AboveThresholdSupportArtifact(
        coarsening_labels=coarsening_labels,
        support_values=support_values,
        threshold=threshold,
        halt_index=halt_index,
        noisy_prefix=noisy_prefix,
        epsilon=epsilon,
    )


def build_ptr_failure_artifact(
    *,
    support_values: np.ndarray | None = None,
    ptr_threshold: int = DEFAULT_SUPPORT_THRESHOLD,
    eps_test: float = 0.4,
    delta: float = PTR_DELTA,
) -> PtrFailureArtifact:
    """Analytic PTR abstention curve as true minimum support varies."""

    if ptr_threshold < 2:
        raise ValueError("ptr_threshold must be at least 2")
    if support_values is None:
        support_values = np.arange(1, 121, dtype=int)
    support_values = np.asarray(support_values, dtype=int)
    abstention = np.array(
        [
            ptr_abstention_probability(int(n), ptr_threshold, eps_test, delta)
            for n in support_values
        ],
        dtype=float,
    )
    distances = np.array(
        [ptr_support_distance(int(n), ptr_threshold) for n in support_values],
        dtype=int,
    )
    return PtrFailureArtifact(
        support_values=support_values,
        abstention_probability=abstention,
        distance_values=distances,
        ptr_threshold=ptr_threshold,
        eps_test=eps_test,
        delta=delta,
        buffer=ptr_test_buffer(eps_test, delta),
        false_accept_at_zero=ptr_false_accept_probability(eps_test, delta),
    )


def evaluate_subgroup_accuracy(
    population_df: pd.DataFrame,
    mechanism: Callable[..., dict[str, Any]],
    *,
    n: int,
    n_datasets: int,
    n_runs: int,
    seed: int,
    group_fn: Callable[[pd.DataFrame], np.ndarray],
    value_fn: Callable[[pd.DataFrame], np.ndarray],
    method: str,
    privacy_status: str,
    epsilon_total: float,
) -> pd.DataFrame:
    """Evaluate subgroup mechanisms with the Lecture 5 accuracy-row schema."""

    rng = np.random.default_rng(seed)
    population_values = value_fn(population_df)
    population_groups = group_fn(population_df)
    population_target = subgroup_difference(population_values, population_groups)
    rows: list[dict[str, Any]] = []
    sample_size = min(n, len(population_df))
    for dataset_id in range(n_datasets):
        for _ in range(50):
            idx = rng.choice(len(population_df), size=sample_size, replace=False)
            sample_df = population_df.iloc[idx]
            x = value_fn(sample_df)
            groups = group_fn(sample_df)
            counts = subgroup_counts(groups)
            if counts["A"] > 0 and counts["B"] > 0:
                break
        else:
            continue
        sample_target = subgroup_difference(x, groups)
        for run_id in range(n_runs):
            run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            result = mechanism(x, groups, run_rng)
            estimate = float(result["estimate"])
            if not np.isfinite(estimate):
                continue
            for reference, target in (("sample", sample_target), ("population", population_target)):
                error = estimate - target
                rows.append(
                    {
                        "method": method,
                        "privacy_status": privacy_status,
                        "estimate_target": "subgroup_difference",
                        "error_metric": "value",
                        "reference": reference,
                        "epsilon_total": epsilon_total,
                        "dataset_id": dataset_id,
                        "run_id": run_id,
                        "estimate": estimate,
                        "target": target,
                        "error": error,
                        "abs_error": abs(error),
                        "squared_error": error * error,
                        "notes": "",
                    }
                )
    return pd.DataFrame(rows)


def make_subgroup_sampling_distribution_figure(
    artifact: SubgroupSamplingArtifact,
    *,
    title: str,
    x_label: str = "normalized average salary gap",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Plot sample-to-sample variation around the population normalized salary gap."""

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=_DPI)
    edges = adaptive_histogram_bin_edges(artifact.sample_gaps)
    add_histogram_density_line(
        ax,
        artifact.sample_gaps,
        edges,
        label=(
            f"{artifact.n_samples} resamples "
            f"(draw n={artifact.sample_size:,} from N={artifact.population_size:,})"
        ),
        color=PROVENANCE_PALETTE["typical"],
        linewidth=2.0,
    )
    ax.axvline(
        artifact.population_gap,
        color=PROVENANCE_PALETTE["engineered"],
        linestyle=MPL_BOUND,
        linewidth=2,
        label=f"population gap = {artifact.population_gap:.3f}",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("density")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def make_subgroup_accuracy_leaderboard_figure(
    leaderboard: pd.DataFrame,
    *,
    title: str,
) -> Figure:
    """Lecture 5-style signed-error leaderboard for one subgroup mechanism."""

    return make_accuracy_leaderboard_figure(
        leaderboard,
        title=title,
        **_SUBGROUP_LEADERBOARD_KWARGS,
    )


def make_ptr_failure_probability_figure(
    artifact: PtrFailureArtifact,
    *,
    title: str,
    marker_min_count: int | None = None,
    marker_label: str | None = None,
) -> Figure:
    """Plot analytic PTR test-failure (abstention) probability vs true support."""

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=_DPI)
    ax.plot(
        artifact.support_values,
        artifact.abstention_probability,
        color=PROVENANCE_PALETTE["engineered"],
        linestyle=MPL_PRIMARY,
        linewidth=2.0,
        label="P(test fails → abstain)",
    )
    ax.axhline(
        artifact.delta,
        color="#555555",
        linestyle=MPL_REFERENCE,
        linewidth=1.5,
        label=f"chosen failure probability δ={artifact.delta:g}",
    )
    ax.axhline(
        artifact.false_accept_at_zero,
        color=PROVENANCE_PALETTE["extreme_real"],
        linestyle=MPL_SECONDARY,
        linewidth=1.2,
        label=f"P(release | distance=0)={artifact.false_accept_at_zero:.3f}",
    )
    ax.axvline(
        artifact.ptr_threshold,
        color=PROVENANCE_PALETTE["typical"],
        linestyle=MPL_BOUND,
        linewidth=1.5,
        label=f"PTR threshold m={artifact.ptr_threshold}",
    )
    if marker_min_count is not None:
        marker_prob = float(
            np.interp(
                marker_min_count,
                artifact.support_values.astype(float),
                artifact.abstention_probability,
            )
        )
        ax.scatter(
            [marker_min_count],
            [marker_prob],
            color=PROVENANCE_PALETTE["typical"],
            s=36,
            zorder=5,
            label=marker_label or f"this sample min n={marker_min_count}",
        )
    ax.set_xlabel("true minimum group support min(n_A, n_B)")
    ax.set_ylabel("PTR test failure probability")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def make_support_comparison_figure(
    artifact: SupportComparisonArtifact,
    *,
    min_ptr_releases: int = 10,
) -> Figure:
    """Plot repair behavior on a common true-support x-axis."""

    fig, curve_ax = plt.subplots(figsize=(7.5, 4.5), dpi=_DPI)
    ax2 = curve_ax.twinx()
    x = artifact.min_support_values
    ptr_error = np.where(
        artifact.ptr_release_counts >= min_ptr_releases,
        artifact.ptr_error,
        np.nan,
    )
    series = [
        (artifact.count_sum_error, PROVENANCE_PALETTE["typical"], "o", MPL_PRIMARY, "count/sum |error|"),
        (artifact.ptr_abstention, PROVENANCE_PALETTE["engineered"], "s", MPL_REFERENCE, "PTR abstention"),
        (
            ptr_error,
            PROVENANCE_PALETTE["extreme_real"],
            "^",
            MPL_SECONDARY,
            f"PTR |error| when released (m={artifact.ptr_threshold})",
        ),
        (artifact.smooth_error, PROVENANCE_PALETTE["typical"], "v", MPL_BOUND, "smooth |error|"),
    ]
    for idx, (values, color, marker, linestyle, label) in enumerate(series):
        axis = curve_ax if idx == 1 else ax2
        axis.plot(
            x,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=2.0,
        )
    rare_release_mask = artifact.ptr_release_counts < min_ptr_releases
    if np.any(rare_release_mask):
        ax2.scatter(
            x[rare_release_mask],
            np.zeros(np.count_nonzero(rare_release_mask)),
            marker="x",
            color=PROVENANCE_PALETTE["extreme_real"],
            label=f"PTR error hidden: <{min_ptr_releases} releases",
            zorder=4,
        )
    curve_ax.set_xlabel("true minimum group count min(n_A,n_B)")
    curve_ax.set_ylabel("PTR abstention probability", color=PROVENANCE_PALETTE["engineered"])
    ax2.set_ylabel("mean absolute error of Delta", color=PROVENANCE_PALETTE["typical"])
    curve_ax.set_title("Repairs vs support")
    curve_ax.set_xscale("log")
    curve_ax.set_xticks(x)
    curve_ax.set_xticklabels([str(int(v)) for v in x])
    lines1, labels1 = curve_ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    curve_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig

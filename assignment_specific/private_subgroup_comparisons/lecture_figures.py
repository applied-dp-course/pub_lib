"""Static matplotlib figure builders for the private subgroup comparisons lecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from libdpy.assignment_specific.private_estimation.lecture_figures import (
    make_accuracy_leaderboard_figure,
    make_composition_budget_figure,
    make_sparse_group_warning_figure,
)
from libdpy.assignment_specific.private_estimation.utils import (
    DEFAULT_DELTA,
    DEFAULT_SEED,
    AuditPanel,
    DataProvenance,
    FIGURE_FREEZE_N_TRIALS_AUDIT,
    audit_panel,
)
from libdpy.assignment_specific.private_estimation.visualizations import (
    PROVENANCE_PALETTE,
    adaptive_histogram_bin_edges,
    add_histogram_density_line,
    audit_panel_figure,
    audit_panels_comparison_figure,
    sorted_neighbor_bars_figure,
)
from libdpy.visualization.plot_styles import MPL_BOUND, MPL_PRIMARY, MPL_REFERENCE, MPL_SECONDARY
from libdpy.privacy_mechanisms.above_threshold import above_threshold
from libdpy.assignment_specific.private_subgroup_comparisons.mechanisms import (
    GLOBAL_SENSITIVITY_NORMALIZED,
    conceptual_smooth_sensitivity_release,
    noisy_count_sum_release,
    oracle_local_sensitivity_release,
    ptr_support_release,
    replacement_ls_bound,
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
    PUBLIC_CLIP_LOWER,
    PUBLIC_CLIP_UPPER,
    age_group_labels,
    build_balanced_subgroup_sample,
    build_imbalanced_subgroup_sample,
    build_oracle_ls_failure_neighbor_pair,
    build_sparse_subgroup_sample,
    frame_to_arrays,
    prepare_fulton_subgroup_frame,
)

_DPI = 100
DEFAULT_BETA = 0.35
ORACLE_EPS = 1.0
PTR_DELTA = 0.05
_SUBGROUP_ACCURACY_SAMPLE_N = 400
_SUBGROUP_LEADERBOARD_KWARGS = {
    "estimate_target": "subgroup_difference",
    "error_metric": "value",
    "x_label": "signed error in normalized Δ",
    "std_fmt": ".4f",
    "x_tick_fmt": ".3f",
    "round_decimals": 3,
}


def _finite_leaderboard_rows(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Drop abstentions and other non-finite releases before accuracy plotting."""

    return leaderboard[np.isfinite(leaderboard["estimate"])].copy()


@dataclass(frozen=True)
class SubgroupSamplingArtifact:
    x: np.ndarray
    groups: np.ndarray
    true_delta: float
    bootstrap_deltas: np.ndarray


@dataclass(frozen=True)
class LocalSensitivityArtifact:
    support_values: np.ndarray
    ls_bounds: np.ndarray


@dataclass(frozen=True)
class OracleLocalSensitivityArtifact:
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]]
    epsilon: float
    rmse_by_scenario: dict[str, float]


@dataclass(frozen=True)
class OracleScaleLeakArtifact:
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
    x: np.ndarray
    groups: np.ndarray
    tau: float
    eps_total: float
    budget_rows: list[dict[str, float]]
    budget_ledger: dict[str, dict[str, float]]
    rmse_by_split: dict[str, float]


@dataclass(frozen=True)
class SupportComparisonArtifact:
    min_support_values: np.ndarray
    count_sum_error: np.ndarray
    ptr_abstention: np.ndarray
    ptr_error: np.ndarray
    smooth_error: np.ndarray
    ptr_threshold: int
    beta: float


# Alias for earlier artifact builders and tests.
PtrSmoothArtifact = SupportComparisonArtifact


@dataclass(frozen=True)
class AboveThresholdSupportArtifact:
    coarsening_labels: list[str]
    support_values: list[float]
    threshold: float
    halt_index: int
    noisy_prefix: list[bool]
    epsilon: float


def build_subgroup_sampling_artifact(seed: int = DEFAULT_SEED) -> SubgroupSamplingArtifact:
    frame = prepare_fulton_subgroup_frame(seed=seed)
    x, groups = frame_to_arrays(frame)
    true_delta = subgroup_difference(x, groups)
    rng = np.random.default_rng(seed)
    n = len(x)
    bootstrap = []
    for _ in range(400):
        idx = rng.integers(0, n, size=n)
        bootstrap.append(subgroup_difference(x[idx], groups[idx]))
    return SubgroupSamplingArtifact(
        x=x,
        groups=groups,
        true_delta=true_delta,
        bootstrap_deltas=np.asarray(bootstrap, dtype=float),
    )


def build_local_sensitivity_artifact(seed: int = DEFAULT_SEED) -> LocalSensitivityArtifact:
    _ = seed
    support_values = np.arange(2, 101, dtype=int)
    ls_bounds = np.array(
        [
            replacement_ls_bound(n, n)
            for n in support_values
        ],
        dtype=float,
    )
    return LocalSensitivityArtifact(support_values=support_values, ls_bounds=ls_bounds)


def _oracle_rmse(
    x: np.ndarray,
    groups: np.ndarray,
    *,
    epsilon: float,
    seed: int,
    n_trials: int = 200,
) -> float:
    target = subgroup_difference(x, groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_trials):
        run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        result = oracle_local_sensitivity_release(x, groups, epsilon, run_rng)
        estimates.append(result["estimate"])
    estimates = np.asarray(estimates, dtype=float)
    return float(np.sqrt(np.mean((estimates - target) ** 2)))


def build_oracle_ls_utility_artifact(seed: int = DEFAULT_SEED) -> OracleLocalSensitivityArtifact:
    scenarios = {
        "balanced": build_balanced_subgroup_sample(seed=seed),
        "imbalanced": build_imbalanced_subgroup_sample(seed=seed + 1),
        "sparse": build_sparse_subgroup_sample(seed=seed + 2),
    }
    rmse = {
        name: _oracle_rmse(x, groups, epsilon=ORACLE_EPS, seed=seed + idx)
        for idx, (name, (x, groups)) in enumerate(scenarios.items())
    }
    return OracleLocalSensitivityArtifact(
        scenarios=scenarios,
        epsilon=ORACLE_EPS,
        rmse_by_scenario=rmse,
    )


def build_oracle_ls_failure_artifact(seed: int = DEFAULT_SEED) -> OracleScaleLeakArtifact:
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


def build_count_sum_artifact(seed: int = DEFAULT_SEED) -> CountSumArtifact:
    x, groups = build_balanced_subgroup_sample(seed=seed)
    target = subgroup_difference(x, groups)
    splits = {
        "equal quarters": (0.25, 0.25, 0.25, 0.25),
        "counts heavy": (0.40, 0.40, 0.10, 0.10),
        "sums heavy": (0.10, 0.10, 0.40, 0.40),
    }
    rmse_by_split: dict[str, float] = {}
    budget_rows: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for label, split in splits.items():
        eps_n_a, eps_n_b, eps_s_a, eps_s_b = [DEFAULT_EPS_TOTAL * w for w in split]
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
        for _ in range(200):
            run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            result = noisy_count_sum_release(
                x,
                groups,
                eps_n_a,
                eps_n_b,
                eps_s_a,
                eps_s_b,
                DEFAULT_TAU,
                run_rng,
            )
            estimates.append(result["estimate"])
        rmse_by_split[label] = float(np.sqrt(np.mean((np.asarray(estimates) - target) ** 2)))
    eps_n_a = DEFAULT_EPS_TOTAL * 0.25
    eps_n_b = DEFAULT_EPS_TOTAL * 0.25
    eps_s_a = DEFAULT_EPS_TOTAL * 0.25
    eps_s_b = DEFAULT_EPS_TOTAL * 0.25
    budget_ledger = {
        "epsilon": {
            "n_A": eps_n_a,
            "n_B": eps_n_b,
            "S_A": eps_s_a,
            "S_B": eps_s_b,
        }
    }
    return CountSumArtifact(
        x=x,
        groups=groups,
        tau=DEFAULT_TAU,
        eps_total=DEFAULT_EPS_TOTAL,
        budget_rows=budget_rows,
        budget_ledger=budget_ledger,
        rmse_by_split=rmse_by_split,
    )


def build_support_comparison_artifact(
    seed: int = DEFAULT_SEED,
    *,
    ptr_threshold: int = DEFAULT_SUPPORT_THRESHOLD,
    n_large: int = 700,
    n_trials: int = 120,
) -> SupportComparisonArtifact:
    """Compare repairs as true minimum group support varies (common x-axis)."""

    min_support_values = np.array([3, 5, 8, 12, 20, 30, 50, 80, 120], dtype=int)
    rng = np.random.default_rng(seed)
    count_sum_error: list[float] = []
    ptr_abstention: list[float] = []
    ptr_error: list[float] = []
    smooth_error: list[float] = []
    eps_n = DEFAULT_EPS_COUNT
    eps_s = DEFAULT_EPS_SUM

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
            run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            count_sum_result = noisy_count_sum_release(
                x,
                groups,
                eps_n,
                eps_n,
                eps_s,
                eps_s,
                DEFAULT_TAU,
                run_rng,
            )
            count_sum_estimates.append(count_sum_result["estimate"])

            ptr_result = ptr_support_release(
                x,
                groups,
                ptr_threshold,
                eps_test=0.4,
                eps_release=0.6,
                delta=PTR_DELTA,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
            )
            if not ptr_result["accepted"]:
                abstained += 1
            else:
                ptr_estimates.append(ptr_result["estimate"])

            smooth_result = conceptual_smooth_sensitivity_release(
                x,
                groups,
                epsilon=DEFAULT_EPS_TOTAL,
                beta=DEFAULT_BETA,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
            )
            smooth_estimates.append(smooth_result["estimate"])

        count_sum_error.append(
            float(np.mean(np.abs(np.asarray(count_sum_estimates) - target)))
        )
        ptr_abstention.append(abstained / n_trials)
        if ptr_estimates:
            ptr_error.append(float(np.mean(np.abs(np.asarray(ptr_estimates) - target))))
        else:
            ptr_error.append(float("nan"))
        smooth_error.append(float(np.mean(np.abs(np.asarray(smooth_estimates) - target))))

    return SupportComparisonArtifact(
        min_support_values=min_support_values,
        count_sum_error=np.asarray(count_sum_error, dtype=float),
        ptr_abstention=np.asarray(ptr_abstention, dtype=float),
        ptr_error=np.asarray(ptr_error, dtype=float),
        smooth_error=np.asarray(smooth_error, dtype=float),
        ptr_threshold=ptr_threshold,
        beta=DEFAULT_BETA,
    )


def build_ptr_smooth_artifact(seed: int = DEFAULT_SEED) -> SupportComparisonArtifact:
    """Alias for :func:`build_support_comparison_artifact`."""

    return build_support_comparison_artifact(seed=seed)


def build_above_threshold_support_artifact(seed: int = DEFAULT_SEED) -> AboveThresholdSupportArtifact:
    """Run AboveThreshold on a public coarsening hierarchy with real support queries."""

    frame = prepare_fulton_subgroup_frame(seed=seed)
    coarsening_labels = [
        "sex × educ × age quartile",
        "sex × educ",
        "education only",
        "age < 50 vs ≥ 50",
    ]
    age = frame["age"].to_numpy(dtype=float)
    educ = frame["educ"].to_numpy(dtype=int)
    sex = frame["sex"].to_numpy(dtype=int)
    age_quartile = np.digitize(age, [25, 35, 45, 55, 65], right=True)
    age_groups = np.where(age_group_labels(age) == GROUP_B, 1, 0)
    label_levels = [
        sex * 10000 + educ * 100 + age_quartile,
        sex * 100 + educ,
        educ,
        age_groups,
    ]

    def min_group_count(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        return float(np.min(counts))

    support_values = [min_group_count(labels) for labels in label_levels]
    threshold = 50.0
    database = frame.to_dict("records")
    queries = []
    for labels in label_levels:
        fixed_labels = labels

        def query(db, labels=fixed_labels):
            _ = db
            return min_group_count(labels)

        queries.append(query)

    epsilon = 0.5
    rng = np.random.default_rng(seed)
    noisy_prefix = above_threshold(database, queries, threshold, epsilon, rng=rng)
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


def evaluate_subgroup_accuracy(
    population_df: pd.DataFrame,
    mechanism: Callable[..., dict[str, Any]],
    *,
    n: int = 1000,
    n_datasets: int = 50,
    n_runs: int = 1,
    seed: int = DEFAULT_SEED,
    group_fn: Callable[[pd.DataFrame], np.ndarray],
    value_fn: Callable[[pd.DataFrame], np.ndarray],
    method: str,
    privacy_status: str,
    epsilon_total: float,
) -> pd.DataFrame:
    """Evaluate subgroup mechanisms with the same leaderboard schema as Lecture 5."""

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


def make_subgroup_proposals_diagnoses_repairs_table() -> list[dict]:
    """Proposal → diagnosis → repair table for the subgroup-comparison lecture."""

    return [
        {
            "proposal": "Oracle local-sensitivity noise",
            "why_tempting": "adapts noise to typical denominator fragility",
            "witness": "replacement shrinks smaller group support",
            "diagnosis": "noise scale leaks hidden support",
            "repair": "noisy counts/sums or PTR abstention",
        },
        {
            "proposal": "Global sensitivity on Δ",
            "why_tempting": "one valid Laplace scale everywhere",
            "witness": "engineered tiny group",
            "diagnosis": "pays for sparse worst case",
            "repair": "local bound motivates adaptive repairs",
        },
        {
            "proposal": "Noisy count/sum release",
            "why_tempting": "valid composition; publishes support",
            "witness": "noisy denominator destabilizes mean",
            "diagnosis": "budget spent on denominators",
            "repair": "public τ floor; budget ledger",
        },
        {
            "proposal": "PTR support certification",
            "why_tempting": "abstain instead of fragile release",
            "witness": "borderline min(n_A, n_B)",
            "diagnosis": "conceptual unless theorem stated",
            "repair": "smooth sensitivity (continuous alternative)",
        },
    ]


def build_subgroup_mechanism_leaderboard(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Accuracy rows comparing oracle, count/sum, PTR, and smooth sensitivity."""

    frame = prepare_fulton_subgroup_frame(seed=seed)
    eps = DEFAULT_EPS_TOTAL
    tau = DEFAULT_TAU
    rows: list[pd.DataFrame] = []

    def oracle_mechanism(values, group_labels, run_rng):
        return oracle_local_sensitivity_release(values, group_labels, ORACLE_EPS, run_rng)

    def count_sum_mechanism(values, group_labels, run_rng):
        quarter = eps / 4.0
        return noisy_count_sum_release(
            values, group_labels, quarter, quarter, quarter, quarter, tau, run_rng
        )

    def ptr_mechanism(values, group_labels, run_rng):
        return ptr_support_release(
            values,
            group_labels,
            DEFAULT_SUPPORT_THRESHOLD,
            eps_test=0.4,
            eps_release=0.6,
            delta=PTR_DELTA,
            rng=run_rng,
        )

    def smooth_mechanism(values, group_labels, run_rng):
        return conceptual_smooth_sensitivity_release(
            values, group_labels, epsilon=eps, beta=DEFAULT_BETA, rng=run_rng
        )

    configs = [
        ("oracle local sensitivity", "oracle diagnostic", oracle_mechanism, ORACLE_EPS),
        ("noisy count/sum", "valid", count_sum_mechanism, eps),
        ("PTR support (conceptual)", "conceptual PTR", ptr_mechanism, eps),
        ("smooth sensitivity (conceptual)", "conceptual smooth sensitivity", smooth_mechanism, eps),
    ]
    sample_n = min(_SUBGROUP_ACCURACY_SAMPLE_N, len(frame))
    for method, privacy_status, mechanism, epsilon_total in configs:
        rows.append(
            evaluate_subgroup_accuracy(
                frame,
                mechanism,
                n=sample_n,
                n_datasets=40,
                seed=seed,
                group_fn=lambda df: df["group"].to_numpy(),
                value_fn=lambda df: df["x"].to_numpy(dtype=float),
                method=method,
                privacy_status=privacy_status,
                epsilon_total=epsilon_total,
            )
        )
    return _finite_leaderboard_rows(pd.concat(rows, ignore_index=True))


def build_oracle_utility_leaderboard(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Oracle-only signed-error leaderboard across balanced, imbalanced, and sparse support."""

    scenarios = {
        "balanced": build_balanced_subgroup_sample(seed=seed),
        "imbalanced": build_imbalanced_subgroup_sample(seed=seed + 1),
        "sparse": build_sparse_subgroup_sample(seed=seed + 2),
    }
    rows: list[pd.DataFrame] = []

    def oracle_mechanism(values, group_labels, run_rng):
        return oracle_local_sensitivity_release(values, group_labels, ORACLE_EPS, run_rng)

    for idx, (label, (x, groups)) in enumerate(scenarios.items()):
        frame = pd.DataFrame({"x": x, "group": groups})
        sample_n = min(_SUBGROUP_ACCURACY_SAMPLE_N, len(frame))
        rows.append(
            evaluate_subgroup_accuracy(
                frame,
                oracle_mechanism,
                n=sample_n,
                n_datasets=40,
                seed=seed + idx,
                group_fn=lambda df: df["group"].to_numpy(),
                value_fn=lambda df: df["x"].to_numpy(dtype=float),
                method=f"oracle LS ({label})",
                privacy_status="oracle diagnostic",
                epsilon_total=ORACLE_EPS,
            )
        )
    return _finite_leaderboard_rows(pd.concat(rows, ignore_index=True))


def build_support_repair_leaderboard(
    seed: int = DEFAULT_SEED,
    *,
    sparse_min: int = 10,
    dense_min: int = 120,
    n_large: int = 700,
) -> pd.DataFrame:
    """Leaderboard comparing count/sum, PTR, and smooth sensitivity at sparse vs dense support."""

    eps = DEFAULT_EPS_TOTAL
    tau = DEFAULT_TAU
    rows: list[pd.DataFrame] = []

    def count_sum_mechanism(values, group_labels, run_rng):
        quarter = eps / 4.0
        return noisy_count_sum_release(
            values, group_labels, quarter, quarter, quarter, quarter, tau, run_rng
        )

    def ptr_mechanism(values, group_labels, run_rng):
        return ptr_support_release(
            values,
            group_labels,
            DEFAULT_SUPPORT_THRESHOLD,
            eps_test=0.4,
            eps_release=0.6,
            delta=PTR_DELTA,
            rng=run_rng,
        )

    def smooth_mechanism(values, group_labels, run_rng):
        return conceptual_smooth_sensitivity_release(
            values, group_labels, epsilon=eps, beta=DEFAULT_BETA, rng=run_rng
        )

    repair_configs = [
        ("noisy count/sum", "valid", count_sum_mechanism, eps),
        ("PTR support (conceptual)", "conceptual PTR", ptr_mechanism, eps),
        ("smooth sensitivity (conceptual)", "conceptual smooth sensitivity", smooth_mechanism, eps),
    ]
    for support_label, n_small in (("sparse", sparse_min), ("dense", dense_min)):
        x, groups = build_imbalanced_subgroup_sample(
            n_large=n_large,
            n_small=n_small,
            seed=seed + n_small,
        )
        frame = pd.DataFrame({"x": x, "group": groups})
        sample_n = min(_SUBGROUP_ACCURACY_SAMPLE_N, len(frame))
        for method, privacy_status, mechanism, epsilon_total in repair_configs:
            rows.append(
                evaluate_subgroup_accuracy(
                    frame,
                    mechanism,
                    n=sample_n,
                    n_datasets=30,
                    seed=seed + n_small,
                    group_fn=lambda df: df["group"].to_numpy(),
                    value_fn=lambda df: df["x"].to_numpy(dtype=float),
                    method=f"{method} [{support_label} min={n_small}]",
                    privacy_status=privacy_status,
                    epsilon_total=epsilon_total,
                )
            )
    return _finite_leaderboard_rows(pd.concat(rows, ignore_index=True))


def make_subgroup_mechanism_leaderboard_figure(
    leaderboard: pd.DataFrame | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Lecture 5-style signed-error leaderboard for subgroup mechanisms."""

    leaderboard = leaderboard if leaderboard is not None else build_subgroup_mechanism_leaderboard(seed=seed)
    return make_accuracy_leaderboard_figure(
        leaderboard,
        title="Subgroup mechanism accuracy on Fulton sample",
        **_SUBGROUP_LEADERBOARD_KWARGS,
    )


def make_subgroup_sampling_distribution_figure(
    artifact: SubgroupSamplingArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    artifact = artifact or build_subgroup_sampling_artifact(seed=seed)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=_DPI)
    edges = adaptive_histogram_bin_edges(artifact.bootstrap_deltas)
    add_histogram_density_line(
        ax,
        artifact.bootstrap_deltas,
        edges,
        label="bootstrap resamples",
        color=PROVENANCE_PALETTE["typical"],
        linewidth=2.0,
    )
    ax.axvline(
        artifact.true_delta,
        color=PROVENANCE_PALETTE["engineered"],
        linestyle=MPL_BOUND,
        linewidth=2,
        label=f"sample Δ = {artifact.true_delta:.3f}",
    )
    ax.set_xlabel("normalized subgroup difference")
    ax.set_ylabel("density")
    ax.set_title("Sampling variability before privacy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_local_sensitivity_vs_support_figure(
    artifact: LocalSensitivityArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    artifact = artifact or build_local_sensitivity_artifact(seed=seed)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=_DPI)
    ax.plot(
        artifact.support_values,
        artifact.ls_bounds,
        color=PROVENANCE_PALETTE["typical"],
        linestyle=MPL_PRIMARY,
        linewidth=2,
    )
    ax.set_xlabel("group count n (balanced groups)")
    ax.set_ylabel("conservative local sensitivity bound")
    ax.set_title("Denominator control: LS scales like 1/n_A + 1/n_B")
    fig.tight_layout()
    return fig


def make_global_sensitivity_baseline_figure(seed: int = DEFAULT_SEED) -> Figure:
    """Compare pessimistic global vs typical local Laplace scales on the Fulton sample."""

    frame = prepare_fulton_subgroup_frame(seed=seed)
    x, groups = frame_to_arrays(frame)
    counts = subgroup_counts(groups)
    local_bound = replacement_ls_bound(counts["A"], counts["B"])
    noise_scales = np.array(
        [local_bound / ORACLE_EPS, GLOBAL_SENSITIVITY_NORMALIZED / ORACLE_EPS],
        dtype=float,
    )
    labels = ["local LS bound", "global GS bound"]
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=_DPI)
    bars = ax.bar(
        labels,
        noise_scales,
        color=[PROVENANCE_PALETTE["typical"], PROVENANCE_PALETTE["engineered"]],
        alpha=0.85,
    )
    for bar, value in zip(bars, noise_scales):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Laplace noise scale on normalized Δ (ε = 1)")
    ax.set_title("Global sensitivity is pessimistic on balanced Fulton support")
    fig.tight_layout()
    return fig


def make_oracle_ls_utility_figure(
    artifact: OracleLocalSensitivityArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Lecture 5 signed-error leaderboard: oracle LS across support profiles (not DP)."""

    _ = artifact
    return make_accuracy_leaderboard_figure(
        build_oracle_utility_leaderboard(seed=seed),
        title="Oracle local sensitivity utility by support profile (not DP)",
        **_SUBGROUP_LEADERBOARD_KWARGS,
    )


def make_sparse_subgroup_warning_figure(seed: int = DEFAULT_SEED) -> Figure:
    """Reuse Lecture 5 sparse-group warning (990/10 split) for denominator fragility."""

    return make_sparse_group_warning_figure(seed=seed)


def make_oracle_ls_neighbor_bars_figure(
    artifact: OracleScaleLeakArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Sorted-neighbor witness bars for the oracle-LS failure pair (Lecture 5 idiom)."""

    artifact = artifact or build_oracle_ls_failure_artifact(seed=seed)
    scale = PUBLIC_CLIP_UPPER - PUBLIC_CLIP_LOWER
    x_dollars = artifact.x * scale
    x_prime_dollars = artifact.x_prime * scale
    return sorted_neighbor_bars_figure(
        x_dollars,
        x_prime_dollars,
        title=(
            "Oracle LS witness: one replacement leaves the smaller group "
            f"(n_B: {artifact.counts_d['B']}→{artifact.counts_d_prime['B']})"
        ),
        provenance=DataProvenance.CONSTRUCTED_WITNESS,
        show_quantiles=False,
    )


def build_oracle_ls_privacy_failure_panel(
    artifact: OracleScaleLeakArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> tuple[AuditPanel, OracleScaleLeakArtifact]:
    """Empirical audit panel for the oracle-LS privacy failure witness."""

    artifact = artifact or build_oracle_ls_failure_artifact(seed=seed)

    def mechanism(bundle, rng, epsilon=artifact.epsilon):
        x_arr, group_arr = bundle
        return oracle_local_sensitivity_release(x_arr, group_arr, epsilon, rng)

    bundle_d = (artifact.x, artifact.groups)
    bundle_d_prime = (artifact.x_prime, artifact.groups_prime)
    panel = audit_panel(
        mechanism,
        bundle_d,
        bundle_d_prime,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        extractor=lambda result: float(result["estimate"]),
        adversary_statistic="oracle local-sensitivity release",
    )
    return panel, artifact


def build_oracle_vs_count_sum_audit_panels(
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> tuple[AuditPanel, AuditPanel, OracleScaleLeakArtifact]:
    """Audit panels for oracle LS (invalid) vs noisy count/sum (valid) on the same pair."""

    artifact = build_oracle_ls_failure_artifact(seed=seed)
    quarter = DEFAULT_EPS_TOTAL / 4.0

    def oracle_bundle_mechanism(bundle, rng, epsilon=artifact.epsilon):
        x_arr, group_arr = bundle
        return oracle_local_sensitivity_release(x_arr, group_arr, epsilon, rng)

    def count_sum_bundle_mechanism(bundle, rng):
        x_arr, group_arr = bundle
        return noisy_count_sum_release(
            x_arr, group_arr, quarter, quarter, quarter, quarter, DEFAULT_TAU, rng
        )

    bundle_d = (artifact.x, artifact.groups)
    bundle_d_prime = (artifact.x_prime, artifact.groups_prime)
    oracle_panel = audit_panel(
        oracle_bundle_mechanism,
        bundle_d,
        bundle_d_prime,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        extractor=lambda result: float(result["estimate"]),
        adversary_statistic="oracle local-sensitivity release",
    )
    count_sum_panel = audit_panel(
        count_sum_bundle_mechanism,
        bundle_d,
        bundle_d_prime,
        n_trials,
        DEFAULT_DELTA,
        seed=seed + 1,
        extractor=lambda result: float(result["estimate"]),
        adversary_statistic="noisy count/sum Δ release",
    )
    return oracle_panel, count_sum_panel, artifact


def make_oracle_ls_privacy_failure_figure(
    artifact: OracleScaleLeakArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> Figure:
    panel, artifact = build_oracle_ls_privacy_failure_panel(
        artifact=artifact,
        seed=seed,
        n_trials=n_trials,
    )
    scale_d = replacement_ls_bound(artifact.counts_d["A"], artifact.counts_d["B"]) / artifact.epsilon
    scale_d_prime = (
        replacement_ls_bound(artifact.counts_d_prime["A"], artifact.counts_d_prime["B"])
        / artifact.epsilon
    )
    title = (
        "Oracle local sensitivity leaks support through the release "
        f"(n_B: {artifact.counts_d['B']}→{artifact.counts_d_prime['B']}; "
        f"internal scale {scale_d:.4f}→{scale_d_prime:.4f})"
    )
    return audit_panel_figure(panel, title=title)


def make_oracle_vs_count_sum_audit_figure(
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> Figure:
    """Side-by-side audit panels: oracle LS (invalid) vs noisy count/sum (valid)."""

    oracle_panel, count_sum_panel, _artifact = build_oracle_vs_count_sum_audit_panels(
        seed=seed,
        n_trials=n_trials,
    )
    return audit_panels_comparison_figure(
        [oracle_panel, count_sum_panel],
        ["Oracle LS (not DP)", "Noisy count/sum"],
        title="Invalid oracle vs valid count/sum repair on the same neighbor pair",
    )


def make_noisy_count_sum_release_figure(
    artifact: CountSumArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    artifact = artifact or build_count_sum_artifact(seed=seed)
    return make_composition_budget_figure(artifact.budget_ledger)


def make_support_comparison_figure(
    artifact: SupportComparisonArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Abstention and error curves as true minimum group support varies."""

    artifact = artifact or build_support_comparison_artifact(seed=seed)
    fig, curve_ax = plt.subplots(figsize=(7.5, 4.5), dpi=_DPI)
    ax2 = curve_ax.twinx()
    x = artifact.min_support_values
    series = [
        (artifact.count_sum_error, PROVENANCE_PALETTE["typical"], "o", MPL_PRIMARY, "count/sum |error|"),
        (artifact.ptr_abstention, PROVENANCE_PALETTE["engineered"], "s", MPL_REFERENCE, "PTR abstention"),
        (
            artifact.ptr_error,
            PROVENANCE_PALETTE["extreme_real"],
            "^",
            MPL_SECONDARY,
            f"PTR |error| when released (m={artifact.ptr_threshold})",
        ),
        (artifact.smooth_error, PROVENANCE_PALETTE["typical"], "v", MPL_BOUND, "smooth sensitivity |error|"),
    ]
    for idx, (values, color, marker, linestyle, label) in enumerate(series):
        axis = ax2 if idx != 1 else curve_ax
        axis.plot(
            x,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=2.0,
        )
    curve_ax.set_xlabel("true minimum group support min(n_A, n_B)")
    curve_ax.set_ylabel("abstention probability", color=PROVENANCE_PALETTE["engineered"])
    ax2.set_ylabel("mean absolute error on normalized Δ", color=PROVENANCE_PALETTE["typical"])
    curve_ax.set_title("Common support comparison: count/sum, PTR, and smooth sensitivity")
    curve_ax.set_xscale("log")
    curve_ax.set_xticks(x)
    curve_ax.set_xticklabels([str(int(v)) for v in x])
    lines1, labels1 = curve_ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    curve_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def make_support_repair_leaderboard_figure(seed: int = DEFAULT_SEED) -> Figure:
    """Standalone sparse vs dense repair leaderboard."""

    return make_accuracy_leaderboard_figure(
        build_support_repair_leaderboard(seed=seed),
        title="Repair accuracy at sparse vs dense minimum support",
        **_SUBGROUP_LEADERBOARD_KWARGS,
    )


def make_ptr_vs_smooth_sensitivity_figure(
    artifact: SupportComparisonArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Alias for :func:`make_support_comparison_figure`."""

    return make_support_comparison_figure(artifact=artifact, seed=seed)


def make_above_threshold_support_search_figure(
    artifact: AboveThresholdSupportArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    artifact = artifact or build_above_threshold_support_artifact(seed=seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=_DPI)
    x_pos = np.arange(len(artifact.coarsening_labels))
    bars = ax.bar(
        x_pos,
        artifact.support_values,
        color=PROVENANCE_PALETTE["typical"],
        alpha=0.7,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(artifact.support_values) * 1.5)
    for bar, value in zip(bars, artifact.support_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(
        artifact.threshold,
        color=PROVENANCE_PALETTE["engineered"],
        linestyle=MPL_REFERENCE,
        label=f"threshold m={artifact.threshold:.0f}",
    )
    if artifact.halt_index < len(x_pos):
        ax.axvline(
            artifact.halt_index,
            color=PROVENANCE_PALETTE["extreme_real"],
            linestyle=MPL_BOUND,
            linewidth=2,
            label=f"AboveThreshold halt (ε={artifact.epsilon})",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(artifact.coarsening_labels, rotation=15, ha="right")
    ax.set_ylabel("true min group support q_j(D)")
    prefix = "".join("T" if bit else "F" for bit in artifact.noisy_prefix) or "none"
    ax.set_title(
        "AboveThreshold bridge: first public coarsening with enough support "
        f"(halt index {artifact.halt_index}; noisy prefix {prefix})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

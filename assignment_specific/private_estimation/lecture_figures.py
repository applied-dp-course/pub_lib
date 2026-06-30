"""Static matplotlib figure builders for private estimation and search lectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from matplotlib.figure import Figure

from libdpy.assignment_specific.private_estimation.utils import (
    DEFAULT_DELTA,
    DEFAULT_EPS_TOTAL,
    DEFAULT_K_SIGMA,
    DEFAULT_N,
    DEFAULT_N_TRIALS_AUDIT,
    DEFAULT_SEED,
    FIGURE_FREEZE_N_TRIALS_AUDIT,
    FIGURE_FREEZE_REPAIR_N_SEEDS,
    FIGURE_FREEZE_SELECTION_REPEATS,
    FigureMode,
    PUBLIC_ABOVE_THRESHOLD_FLOORS,
    PUBLIC_INCOME_BIN_EDGES,
    PUBLIC_INCOME_CANDIDATES,
    PUBLIC_INVERSE_SENSITIVITY_THRESHOLDS,
    PUBLIC_LOG1P_BIN_EDGES,
    PUBLIC_ORDERED_INCOME_THRESHOLDS,
    PUBLIC_SCALE_DIFF_BIN_EDGES,
    PUBLIC_AGE_SCALE,
    PUBLIC_TARGET_UPPER,
    audit_panel,
    above_threshold_halt_index,
    bounded_feature_score,
    build_mu_sigma_clipping_witness,
    build_raw_mean_audit_datasets,
    construct_median_gap_subset,
    construct_quantile_gap_subset,
    empirical_mu_sigma_clipped_mean,
    empirical_quantile_clipped_mean,
    extract_income,
    fixed_bin_noisy_histogram,
    gaussian_histogram_mean,
    gaussian_noise_std,
    load_fulton,
    median_value_plus_noise,
    private_baseline_predictor,
    private_quantile_clipped_mean,
    private_location_from_scale,
    private_scale_from_pairwise_diffs,
    raw_noisy_mean,
    replace_one_row,
    report_noisy_max,
    selection_score_sensitivity,
    split_base_and_pool,
    verify_repair_within_claimed_eps,
    verify_witness_separates,
    _disjoint_pairwise_abs_diffs,
)
from libdpy.assignment_specific.private_estimation.visualizations import (
    PROVENANCE_PALETTE,
    accuracy_component_density_line,
    accuracy_histogram_bin_edges,
    analytical_gaussian_shift_figure,
    audit_panel_figure,
    discrete_probability_line,
    histogram_comparison_figure,
    kde_probability_line,
    smooth_probability_mass,
)

_DPI = 100
_ACCURACY_DISPLAY_QUANTILES = (0.01, 0.99)
# Signed rank/CDF error lives in [-0.5, 0.5]; use a wider axis so edge mass is not clipped.
_RANK_SIGNED_ERROR_XLIM = (-1.0, 1.0)

# Lecture budget splits (sum visibly to stated totals)
EPS_TOTAL = DEFAULT_EPS_TOTAL
EPS_LOW_Q = 0.25
EPS_HIGH_Q = 0.25
EPS_MEAN = 0.50
EPS_HIST = 0.5
EPS_SELECT = 0.5
EPS_SCALE = 0.25
EPS_LOCATION = 0.25
EPS_GAUSS_MEAN = 0.50
EPS_BASELINE_GROUPS = 0.5
EPS_BASELINE_TOTAL = EPS_LOW_Q + EPS_HIGH_Q + EPS_MEAN + EPS_SELECT + EPS_BASELINE_GROUPS

RAW_MEAN_EPS = EPS_TOTAL
RAW_MEAN_PUBLIC_RANGE = 100_000_000.0
RAW_MEAN_NOISE_SCALE = gaussian_noise_std(
    RAW_MEAN_PUBLIC_RANGE / DEFAULT_N, RAW_MEAN_EPS, DEFAULT_DELTA
)


_DISCRETE_MARKER_THRESHOLD = 20
_DISCRETE_KDE_THRESHOLD = 40


def _plot_histogram_probability_line(
    ax,
    values: np.ndarray,
    bins: int | np.ndarray,
    *,
    label: str,
    color: str,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    data_range: tuple[float, float] | None = None,
    round_decimals: int = 0,
):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    n_unique = len(np.unique(np.round(finite, round_decimals)))
    if n_unique <= _DISCRETE_MARKER_THRESHOLD:
        centers, probability = discrete_probability_line(
            finite,
            round_decimals=round_decimals,
        )
        ax.vlines(
            centers,
            0.0,
            probability,
            colors=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        return ax.plot(
            centers,
            probability,
            linestyle="none",
            marker="o",
            markersize=5,
            color=color,
            alpha=alpha,
            label=label,
        )[0]
    if data_range is not None and n_unique > _DISCRETE_KDE_THRESHOLD:
        centers, probability = kde_probability_line(
            finite,
            data_range=data_range,
            round_decimals=round_decimals,
        )
    else:
        counts, edges = np.histogram(finite, bins=bins, density=False)
        probability = smooth_probability_mass(counts.astype(float) / finite.size)
        centers = 0.5 * (edges[:-1] + edges[1:])
    return ax.plot(
        centers,
        probability,
        label=label,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )[0]


def make_accuracy_leaderboard_figure(
    leaderboard: pd.DataFrame,
    *,
    estimate_target: str = "mean",
    error_metric: str = "value",
    title: str | None = None,
) -> Figure:
    """Plot signed-error decompositions for one target/metric facet.

    Each method panel overlays three line-style probability histograms:
    sample target minus full-data target, released estimate minus sample target,
    and released estimate minus full-data target.  This keeps the accuracy view
    in signed-error units.
    """

    subset = leaderboard[
        (leaderboard["estimate_target"] == estimate_target)
        & (leaderboard["error_metric"] == error_metric)
    ].copy()
    if subset.empty:
        raise ValueError("No leaderboard rows for the requested target/metric facet.")
    if "rank_denominator" not in subset.columns:
        subset["rank_denominator"] = np.nan

    methods = list(dict.fromkeys(subset["method"].tolist()))
    join_keys = [
        "method",
        "privacy_status",
        "estimate_target",
        "error_metric",
        "epsilon_total",
        "dataset_id",
        "run_id",
    ]
    sample = subset[subset["reference"] == "sample"][
        join_keys + ["estimate", "target", "error", "rank_denominator"]
    ].rename(
        columns={
            "estimate": "estimate_sample_metric",
            "target": "sample_target",
            "error": "estimate_minus_sample",
            "rank_denominator": "sample_rank_denominator",
        }
    )
    population = subset[subset["reference"] == "population"][
        join_keys + ["estimate", "target", "error", "rank_denominator"]
    ].rename(
        columns={
            "estimate": "estimate_population_metric",
            "target": "population_target",
            "error": "estimate_minus_population",
            "rank_denominator": "population_rank_denominator",
        }
    )
    merged = sample.merge(population, on=join_keys, how="inner")
    if merged.empty:
        raise ValueError(
            "Accuracy rows must include both sample and population references."
        )
    if error_metric == "rank":
        merged["sample_minus_population"] = (
            merged["estimate_population_metric"] - merged["estimate_sample_metric"]
        )
    else:
        merged["sample_minus_population"] = (
            merged["sample_target"] - merged["population_target"]
        )
    merged["estimate_minus_population_total"] = (
        merged["sample_minus_population"] + merged["estimate_minus_sample"]
    )

    n_methods = len(methods)
    n_cols = min(2, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10.5 * n_cols, 4.2 * n_rows),
        dpi=_DPI,
        sharey=False,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    if error_metric == "rank":
        def _denominator_label(values: pd.Series, fallback: str) -> str:
            finite = values.dropna().to_numpy(dtype=float)
            unique = np.unique(finite.astype(int)) if finite.size else np.array([])
            return f"{unique[0]:,}" if unique.size == 1 else fallback

        sample_denominator = _denominator_label(
            merged["sample_rank_denominator"], "|sample|"
        )
        population_denominator = _denominator_label(
            merged["population_rank_denominator"], "|population|"
        )
        line_specs = [
            (
                "sample_minus_population",
                (
                    f"rank(population, release)/{population_denominator} - "
                    f"rank(sample, release)/{sample_denominator}"
                ),
                "0.25",
            ),
            (
                "estimate_minus_sample",
                f"rank(sample, release)/{sample_denominator} - 0.5",
                "#2f6fbb",
            ),
            (
                "estimate_minus_population",
                f"rank(population, release)/{population_denominator} - 0.5",
                "#c84e2d",
            ),
        ]
        x_label = "signed CDF error: rank / reference size − 0.5"
        plot_scale = 1.0
        std_fmt = ".3f"
    else:
        empirical_label = f"empirical {estimate_target}"
        distribution_label = f"distribution {estimate_target}"
        line_specs = [
            ("sample_minus_population", f"{empirical_label} - {distribution_label}", "0.25"),
            ("estimate_minus_sample", f"released estimate - {empirical_label}", "#2f6fbb"),
            ("estimate_minus_population", f"released estimate - {distribution_label}", "#c84e2d"),
        ]
        x_label = "signed error in dollars"
        plot_scale = 1.0
        std_fmt = ",.0f"

    for ax, method in zip(axes, methods):
        method_df = merged[merged["method"] == method]
        status = str(method_df["privacy_status"].iloc[0])
        gray_vals = plot_scale * method_df["sample_minus_population"].to_numpy(dtype=float)
        blue_vals = plot_scale * method_df["estimate_minus_sample"].to_numpy(dtype=float)
        total_vals = plot_scale * method_df["estimate_minus_population_total"].to_numpy(
            dtype=float
        )
        plotted_series = [
            (gray_vals, line_specs[0][1], line_specs[0][2], "sample_minus_population"),
            (blue_vals, line_specs[1][1], line_specs[1][2], "estimate_minus_sample"),
            (total_vals, line_specs[2][1], line_specs[2][2], "estimate_minus_population"),
        ]
        finite = np.concatenate(
            [v[np.isfinite(v)] for v, *_ in plotted_series]
        )
        if finite.size == 0:
            continue
        component_ranges = []
        for values, *_ in plotted_series:
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue
            q_low, q_high = np.quantile(finite_values, _ACCURACY_DISPLAY_QUANTILES)
            central_values = finite_values[
                (finite_values >= q_low) & (finite_values <= q_high)
            ]
            center = float(np.median(central_values))
            std = (
                float(np.std(central_values, ddof=1))
                if central_values.size > 1
                else 0.0
            )
            if std > 0:
                component_ranges.append((center - 4.0 * std, center + 4.0 * std))
            component_ranges.append((float(q_low), float(q_high)))
        display_low = min(low for low, _ in component_ranges)
        display_high = max(high for _, high in component_ranges)
        if display_low == display_high:
            pad = max(abs(float(display_low)) * 0.01, 1.0)
        else:
            pad = 0.06 * float(display_high - display_low)
        hist_range = (float(display_low - pad), float(display_high + pad))
        if error_metric == "rank":
            hist_range = _RANK_SIGNED_ERROR_XLIM
        for values, label, color, column in plotted_series:
            finite_values = values[np.isfinite(values)]
            if error_metric == "rank":
                visible_values = finite_values
            else:
                visible_values = finite_values[
                    (finite_values >= hist_range[0]) & (finite_values <= hist_range[1])
                ]
            if visible_values.size == 0:
                continue
            line_std = float(np.std(finite_values)) if finite_values.size else float("nan")
            centers, density = accuracy_component_density_line(
                visible_values,
                hist_range,
                round_decimals=3 if error_metric == "rank" else 0,
            )
            ax.plot(
                centers,
                density,
                label=f"{label} (std {line_std:{std_fmt}})",
                color=color,
                linewidth=2.0,
                alpha=0.65
                if error_metric == "rank" and column == "sample_minus_population"
                else 1.0,
            )
        ax.set_xlim(hist_range)
        y_max = 0.0
        for line in ax.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if ydata.size:
                y_max = max(y_max, float(np.nanmax(ydata)))
        if y_max > 0:
            ax.set_ylim(0.0, 1.12 * y_max)
        ax.axvline(0, color="0.2", linestyle=":", linewidth=1)
        ax.set_title(f"{method} ({status})")
        ax.set_xlabel(x_label)
        if error_metric == "value":
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        elif error_metric == "rank":
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.set_ylabel("probability density")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
    for ax in axes[n_methods:]:
        ax.axis("off")
    fig.suptitle(
        title or f"Signed-error decomposition: {estimate_target}/{error_metric}"
    )
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class RawMeanSectionArtifact:
    D: np.ndarray
    D_prime: np.ndarray
    out_idx: int
    witness: float
    mean_D: float
    mean_D_prime: float
    noise_scale: float = RAW_MEAN_NOISE_SCALE
    epsilon: float = RAW_MEAN_EPS


@dataclass(frozen=True)
class QuantileClippingSectionArtifact:
    fulton_D: np.ndarray
    gap_D: np.ndarray
    D: np.ndarray
    D_prime: np.ndarray
    out_idx: int
    witness: float
    fulton_bounds: tuple[float, float]
    gap_bounds: tuple[float, float]


@dataclass(frozen=True)
class MedianSectionArtifact:
    D: np.ndarray
    D_prime: np.ndarray
    out_idx: int
    witness: float


@dataclass(frozen=True)
class AboveThresholdSectionArtifact:
    D: np.ndarray
    D_prime: np.ndarray
    floors: np.ndarray
    count_threshold: float
    oracle_table: list[dict]
    halt_index: float
    halt_floor: float
    out_idx: int
    witness: float


@dataclass(frozen=True)
class ReportNoisyMaxSectionArtifact:
    age: np.ndarray
    income: np.ndarray
    income_prime: np.ndarray
    out_idx: int
    witness: float
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]]
    exact_scores: dict[str, float]
    selected: str
    noisy_draw: dict[str, float]
    selection_counts: dict[str, int]
    sensitivity: float


@dataclass(frozen=True)
class MuSigmaSectionArtifact:
    D: np.ndarray
    D_prime: np.ndarray
    out_idx: int
    witness: float
    k: float = DEFAULT_K_SIGMA


@dataclass(frozen=True)
class BaselinePredictorSectionArtifact:
    age: np.ndarray
    income: np.ndarray
    D: np.ndarray
    D_prime: np.ndarray
    result: dict[str, Any]
    total_epsilon: float


def build_raw_mean_section_artifact(
    n_base: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> RawMeanSectionArtifact:
    D, D_prime, out_idx, witness = build_raw_mean_audit_datasets(n_base=n_base, seed=seed)
    return RawMeanSectionArtifact(
        D=D,
        D_prime=D_prime,
        out_idx=out_idx,
        witness=witness,
        mean_D=float(np.mean(D)),
        mean_D_prime=float(np.mean(D_prime)),
    )


def build_quantile_clipping_section_artifact(
    seed: int = DEFAULT_SEED,
    *,
    n_fulton: int = DEFAULT_N,
    n_gap: int = DEFAULT_N,
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> QuantileClippingSectionArtifact:
    fulton_D, _ = _fulton_income_subset(n_base=n_fulton, seed=seed)
    gap_D = construct_quantile_gap_subset(fulton_D, n=n_gap, seed=seed)
    bulk_count = max(2, int(round(0.99 * n_gap)))
    out_idx = bulk_count - 1
    witness = 200_000.0
    D_prime = replace_one_row(gap_D, out_idx, witness)
    return QuantileClippingSectionArtifact(
        fulton_D=fulton_D,
        gap_D=gap_D,
        D=gap_D,
        D_prime=D_prime,
        out_idx=out_idx,
        witness=witness,
        fulton_bounds=(float(np.quantile(fulton_D, low_q)), float(np.quantile(fulton_D, high_q))),
        gap_bounds=(float(np.quantile(gap_D, low_q)), float(np.quantile(gap_D, high_q))),
    )


def build_above_threshold_section_artifact(
    seed: int = DEFAULT_SEED,
    n_base: int = DEFAULT_N,
) -> AboveThresholdSectionArtifact:
    from libdpy.assignment_specific.private_estimation.utils import above_threshold_oracle_table

    D, pool = _fulton_income_subset(n_base=n_base, seed=seed)
    floors = PUBLIC_ABOVE_THRESHOLD_FLOORS
    count_threshold = 0.5 * len(D)
    rng = np.random.default_rng(seed)
    halt_idx = above_threshold_halt_index(D, floors, EPS_SELECT, rng, descending=False)
    halt_floor = float(floors[::-1][int(halt_idx) - 1])
    out_idx = int(np.argmin(D))
    witness = float(np.max(pool))
    D_prime = replace_one_row(D, out_idx, witness)
    return AboveThresholdSectionArtifact(
        D=D,
        D_prime=D_prime,
        floors=floors,
        count_threshold=count_threshold,
        oracle_table=above_threshold_oracle_table(D, floors),
        halt_index=halt_idx,
        halt_floor=halt_floor,
        out_idx=out_idx,
        witness=witness,
    )


def build_median_section_artifact(seed: int = DEFAULT_SEED) -> MedianSectionArtifact:
    gap_D = construct_median_gap_subset(np.array([]), n=DEFAULT_N, seed=seed)
    below_gap = gap_D[gap_D < 50_000]
    out_idx = int(np.where(gap_D == np.max(below_gap))[0][0])
    witness = 85_000.0
    D_prime = replace_one_row(gap_D, out_idx, witness)
    return MedianSectionArtifact(D=gap_D, D_prime=D_prime, out_idx=out_idx, witness=witness)


def build_report_noisy_max_section_artifact(
    seed: int = DEFAULT_SEED,
    n_rows: int = 1000,
    n_repeats: int = 80,
) -> ReportNoisyMaxSectionArtifact:
    df = load_fulton().head(n_rows)
    age = df["age"].to_numpy(dtype=float)
    income = extract_income(df)
    transforms = _public_age_transforms()
    n = len(income)
    sens = selection_score_sensitivity(n)
    exact_scores = {
        name: bounded_feature_score(age, income, phi, target_upper=PUBLIC_TARGET_UPPER)
        for name, phi in transforms.items()
    }
    rng = np.random.default_rng(seed)
    draw_rng = np.random.default_rng(seed + 1)
    noisy_draw = {
        name: exact_scores[name] + float(
            draw_rng.normal(0.0, gaussian_noise_std(sens, EPS_SELECT))
        )
        for name in transforms
    }
    selected = report_noisy_max(
        list(transforms.keys()),
        lambda name: bounded_feature_score(
            age, income, transforms[name], target_upper=PUBLIC_TARGET_UPPER
        ),
        EPS_SELECT,
        sens,
        rng,
    )
    counts: dict[str, int] = {name: 0 for name in transforms}
    for trial in range(n_repeats):
        pick = report_noisy_max(
            list(transforms.keys()),
            lambda name, t=trial: bounded_feature_score(
                age, income, transforms[name], target_upper=PUBLIC_TARGET_UPPER
            ),
            EPS_SELECT,
            sens,
            np.random.default_rng(seed + 10 + trial),
        )
        counts[pick] += 1
    # Witness: swap a low-income row for a high value to shift bounded scores
    out_idx = int(np.argmin(income))
    witness = float(np.max(income) * 1.5)
    income_prime = replace_one_row(income, out_idx, witness)
    return ReportNoisyMaxSectionArtifact(
        age=age,
        income=income,
        income_prime=income_prime,
        out_idx=out_idx,
        witness=witness,
        transforms=transforms,
        exact_scores=exact_scores,
        selected=selected,
        noisy_draw=noisy_draw,
        selection_counts=counts,
        sensitivity=sens,
    )


def build_mu_sigma_section_artifact(
    seed: int = DEFAULT_SEED,
    *,
    n_base: int = DEFAULT_N,
    k: float = DEFAULT_K_SIGMA,
) -> MuSigmaSectionArtifact:
    D, D_prime, out_idx, witness = build_mu_sigma_clipping_witness(n_base=n_base, seed=seed, k=k)
    return MuSigmaSectionArtifact(
        D=D,
        D_prime=D_prime,
        out_idx=out_idx,
        witness=witness,
        k=k,
    )


def build_baseline_predictor_section_artifact(
    seed: int = DEFAULT_SEED,
    n_rows: int = DEFAULT_N,
) -> BaselinePredictorSectionArtifact:
    df = load_fulton().head(n_rows)
    age = df["age"].to_numpy(dtype=float)
    income = extract_income(df)
    D = income.copy()
    D_prime = replace_one_row(D, 0, float(np.max(income)))
    transforms = _public_age_transforms()
    rng = np.random.default_rng(seed)
    result = private_baseline_predictor(
        age,
        D,
        list(transforms.keys()),
        transforms,
        candidates=PUBLIC_INCOME_CANDIDATES,
        eps_preprocess=(EPS_LOW_Q, EPS_HIGH_Q, EPS_MEAN),
        eps_select=EPS_SELECT,
        eps_groups=EPS_BASELINE_GROUPS,
        rng=rng,
    )
    return BaselinePredictorSectionArtifact(
        age=age,
        income=income,
        D=D,
        D_prime=D_prime,
        result=result,
        total_epsilon=EPS_BASELINE_TOTAL,
    )


def _fulton_income_subset(n_base: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    df = load_fulton()
    income = extract_income(df)
    return split_base_and_pool(income, n_base, seed)


def make_oracle_income_histogram_figure(
    *,
    log_scale: bool = False,
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Oracle EDA histogram — **not a private release** (``fig-oracle-income-histogram``)."""

    full_income = extract_income(load_fulton())
    if log_scale:
        D = extract_income(load_fulton(), transform="log1p")
        title = (
            f"Full Fulton income dataset (n={len(D):,}, log1p + public shift {15_000:.0f})"
            f" — exact histogram, not a private release"
        )
        edges = PUBLIC_LOG1P_BIN_EDGES
    else:
        D = full_income
        title = (
            f"Full Fulton income dataset (n={len(D):,})"
            f" — exact histogram, not a private release"
        )
        edges = PUBLIC_INCOME_BIN_EDGES
    fig = histogram_comparison_figure(D, edges, title=title)
    return fig


def make_analytical_raw_mean_figure(
    artifact: RawMeanSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Analytical Gaussian shift for raw noisy mean (``fig-analytical-raw-mean``)."""

    artifact = artifact or build_raw_mean_section_artifact(seed=seed)
    return analytical_gaussian_shift_figure(
        artifact.mean_D_prime,
        artifact.mean_D,
        artifact.noise_scale,
        title="Raw noisy mean: witness shift vs Gaussian std",
        xlabel="released noisy mean",
    )


def make_audit_raw_mean_figure(
    artifact: RawMeanSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> Figure:
    """Empirical audit of raw noisy mean (``fig-audit-raw-mean``)."""

    artifact = artifact or build_raw_mean_section_artifact(seed=seed)

    def mechanism(x, rng, epsilon=RAW_MEAN_EPS, noise_scale_override=RAW_MEAN_NOISE_SCALE):
        return raw_noisy_mean(x, epsilon, noise_scale_override, rng)

    panel = verify_witness_separates(
        mechanism,
        artifact.D,
        artifact.D_prime,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        claimed_eps=RAW_MEAN_EPS,
        adversary_statistic="released noisy mean",
    )
    return audit_panel_figure(panel, title="Raw noisy mean audit")


def make_empirical_quantile_bounds_table(
    artifact: QuantileClippingSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Before/after bounds for empirical quantile clipping (``tbl-empirical-quantile-bounds-before-after``)."""

    artifact = artifact or build_quantile_clipping_section_artifact(seed=seed)
    fulton_L, fulton_U = artifact.fulton_bounds
    gap_L, gap_U = artifact.gap_bounds
    return [
        {
            "dataset": "Fulton subset",
            "L": fulton_L,
            "U": fulton_U,
            "source": "empirical quantiles (non-private)",
        },
        {
            "dataset": "engineered N=1000 quantile-gap witness",
            "L": gap_L,
            "U": gap_U,
            "source": "constructed gap — witness exposes threshold leak",
        },
    ]


def make_audit_empirical_quantile_clipping_figure(
    artifact: QuantileClippingSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> Figure:
    """Audit empirical quantile clipping (``fig-audit-empirical-quantile-clipping``)."""

    artifact = artifact or build_quantile_clipping_section_artifact(seed=seed)

    def mechanism(x, rng, epsilon=EPS_TOTAL, low_q=0.01, high_q=0.99):
        return empirical_quantile_clipped_mean(x, epsilon, low_q, high_q, rng)

    panel = verify_witness_separates(
        mechanism,
        artifact.D,
        artifact.D_prime,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        claimed_eps=EPS_TOTAL,
        margin=0.05,
        adversary_statistic="released clipped mean",
    )
    return audit_panel_figure(panel, title="Empirical quantile clipping audit")


def make_median_ordinary_vs_worst_case_figure(seed: int = DEFAULT_SEED) -> Figure:
    """Ordinary vs worst-case median sensitivity (``fig-median-ordinary-vs-worst-case``)."""

    D, _ = _fulton_income_subset(seed=seed)
    gap_D = construct_median_gap_subset(D, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=_DPI)
    for ax, data, label, color in [
        (axes[0], D[:DEFAULT_N], "ordinary Fulton rows", "C0"),
        (
            axes[1],
            gap_D,
            "constructed median-gap subset",
            PROVENANCE_PALETTE["engineered"],
        ),
    ]:
        med = float(np.median(data))
        ax.hist(data, bins=30, alpha=0.7, color=color)
        ax.axvline(med, color=color, linewidth=2, label=f"median={med:.0f}")
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.suptitle("Median: robust in practice, unstable in worst case")
    fig.tight_layout()
    return fig


def make_rank_loss_median_figure(seed: int = DEFAULT_SEED) -> Figure:
    """Rank-loss utility over public candidates (``fig-rank-loss-median``)."""

    D, _ = _fulton_income_subset(n_base=DEFAULT_N, seed=seed)
    thresholds = PUBLIC_INVERSE_SENSITIVITY_THRESHOLDS
    alpha = 0.5
    n = len(D)
    target = alpha * n
    ranks = np.array([np.sum(D <= t) for t in thresholds])
    utilities = -np.abs(ranks - target)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=_DPI)
    ax.plot(thresholds / 1000, utilities, color="C0")
    ax.set_xlabel("threshold t (thousands $)")
    ax.set_ylabel("rank utility u(t)")
    ax.set_title("Rank-loss utility for median (α=0.5)")
    fig.tight_layout()
    return fig


def make_audit_median_value_noise_figure(
    artifact: MedianSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> Figure:
    """Audit range-calibrated median value+noise on median-gap subset."""

    artifact = artifact or build_median_section_artifact(seed=seed)

    def mechanism(x, rng, epsilon=EPS_TOTAL):
        return median_value_plus_noise(x, epsilon, rng)

    panel = audit_panel(
        mechanism,
        artifact.D,
        artifact.D_prime,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        claimed_eps=EPS_TOTAL,
        adversary_statistic="released noisy median value",
    )
    return audit_panel_figure(panel, title="Range-calibrated median value+noise")


def make_compare_empirical_vs_private_quantile_clipping_figure(
    seed: int = DEFAULT_SEED,
) -> Figure:
    """Compare broken empirical vs repaired private clipping (``fig-compare-empirical-vs-private-quantile-clipping``)."""

    D, _ = _fulton_income_subset(n_base=1000, seed=seed)
    rng = np.random.default_rng(seed)
    emp_L = float(np.quantile(D, 0.01))
    emp_U = float(np.quantile(D, 0.99))
    repaired = private_quantile_clipped_mean(
        D,
        PUBLIC_INVERSE_SENSITIVITY_THRESHOLDS,
        0.01,
        0.99,
        EPS_LOW_Q,
        EPS_HIGH_Q,
        EPS_MEAN,
        rng,
    )
    fig, ax = plt.subplots(figsize=(8, 3), dpi=_DPI)
    ax.barh(
        [0, 1],
        [emp_U - emp_L, repaired["U"] - repaired["L"]],
        color=["C3", "C2"],
        height=0.4,
    )
    ax.set_yticks([0, 1], ["empirical clip width", "private clip width"])
    ax.set_xlabel("interval width ($)")
    ax.set_title("Empirical vs private quantile clipping bounds")
    fig.tight_layout()
    return fig


def make_audit_empirical_mu_sigma_clipping_figure(
    artifact: MuSigmaSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
) -> Figure:
    """Audit μ±kσ clipping (``fig-audit-empirical-mu-sigma-clipping``).

    Uses the provided artifact. The notebook labels any smaller artifact as an
    engineered audit micro-example; accuracy runs stay on the fixed N contract.
    """

    artifact = artifact or build_mu_sigma_section_artifact(seed=seed)
    k = artifact.k

    def mechanism(x, rng, epsilon=EPS_TOTAL, k=k):
        return empirical_mu_sigma_clipped_mean(x, epsilon, k, rng)

    panel = verify_witness_separates(
        mechanism,
        artifact.D,
        artifact.D_prime,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        claimed_eps=EPS_TOTAL,
        adversary_statistic="released μ±4σ clipped mean",
    )
    return audit_panel_figure(panel, title=f"Empirical μ±4σ clipping audit (n={len(artifact.D)})")


def make_gaussian_scale_noisy_histogram_figure(
    seed: int = DEFAULT_SEED,
    *,
    epsilon_hist: float = EPS_HIST,
) -> Figure:
    """Gaussian scale localization (``fig-gaussian-scale-noisy-histogram``)."""

    rng = np.random.default_rng(seed)
    x = rng.normal(50_000, 15_000, size=DEFAULT_N)
    pairwise = _disjoint_pairwise_abs_diffs(x, rng)
    hist = fixed_bin_noisy_histogram(pairwise, PUBLIC_SCALE_DIFF_BIN_EDGES, epsilon_hist, rng=rng)
    centers = 0.5 * (hist["bin_edges"][:-1] + hist["bin_edges"][1:])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=_DPI)
    ax.bar(centers, hist["counts"], width=centers[1] - centers[0], alpha=0.7, color="C0")
    ax.set_xlabel("disjoint pairwise |difference|")
    ax.set_ylabel("noisy count")
    ax.set_title("Private scale via noisy pairwise-difference histogram")
    fig.tight_layout()
    return fig


def make_gaussian_location_noisy_histogram_figure(
    seed: int = DEFAULT_SEED,
    *,
    eps_scale: float = EPS_SCALE,
    eps_location: float = EPS_LOCATION,
) -> Figure:
    """Gaussian location localization (``fig-gaussian-location-noisy-histogram``)."""

    rng = np.random.default_rng(seed)
    mu, sigma = 50_000.0, 15_000.0
    x = rng.normal(mu, sigma, size=DEFAULT_N)
    scale_result = private_scale_from_pairwise_diffs(
        x, eps_scale, PUBLIC_SCALE_DIFF_BIN_EDGES, rng
    )
    sigma_tilde = scale_result["sigma_tilde"]
    loc_result = private_location_from_scale(
        x, sigma_tilde, eps_location, PUBLIC_INCOME_BIN_EDGES, rng
    )
    hist = loc_result["histogram"]
    centers = 0.5 * (hist["bin_edges"][:-1] + hist["bin_edges"][1:])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=_DPI)
    ax.bar(centers, hist["counts"], width=(centers[1] - centers[0]) * 0.9, alpha=0.7, color="C1")
    ax.axvline(mu, color="C3", linestyle="--", label=f"true μ={mu:,.0f}")
    ax.set_xlabel("income ($)")
    ax.set_ylabel("noisy count")
    ax.set_title("Private location via noisy histogram (original units)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_gaussian_method_comparison_figure(
    seed: int = DEFAULT_SEED,
    *,
    eps_scale: float = EPS_SCALE,
    eps_location: float = EPS_LOCATION,
    eps_mean: float = EPS_GAUSS_MEAN,
) -> Figure:
    """Synthetic Gaussian then income comparison (``fig-gaussian-method-comparison``)."""

    rng = np.random.default_rng(seed)
    synthetic = rng.normal(50_000, 15_000, size=DEFAULT_N)
    syn_result = gaussian_histogram_mean(
        synthetic,
        eps_scale,
        eps_location,
        eps_mean,
        PUBLIC_SCALE_DIFF_BIN_EDGES,
        PUBLIC_INCOME_BIN_EDGES,
        rng,
    )
    D, _ = _fulton_income_subset(n_base=1000, seed=seed)
    inc_result = gaussian_histogram_mean(
        D,
        eps_scale,
        eps_location,
        eps_mean,
        PUBLIC_SCALE_DIFF_BIN_EDGES,
        PUBLIC_INCOME_BIN_EDGES,
        np.random.default_rng(seed + 1),
    )
    fig, ax = plt.subplots(figsize=(8, 4), dpi=_DPI)
    ax.bar(
        ["synthetic Gaussian", "Fulton income"],
        [syn_result["estimate"], inc_result["estimate"]],
        color=["C2", "C3"],
        alpha=0.8,
    )
    ax.axhline(float(np.mean(synthetic)), color="C2", linestyle="--", label="true synthetic mean")
    ax.axhline(float(np.mean(D)), color="C3", linestyle="--", label="true income mean")
    ax.set_ylabel("private estimate")
    ax.set_title("Gaussian histogram-mean: model assumption matters")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_proposals_diagnoses_repairs_table() -> list[dict]:
    """Summary table (``tbl-proposals-diagnoses-repairs``)."""

    from libdpy.assignment_specific.private_estimation.utils import proposals_diagnoses_repairs_table

    return proposals_diagnoses_repairs_table()


# ---------------------------------------------------------------------------
# Lecture 6 figures
# ---------------------------------------------------------------------------


def make_above_threshold_audit_figure(
    artifact: AboveThresholdSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
    n_seeds: int = FIGURE_FREEZE_REPAIR_N_SEEDS,
) -> Figure:
    """AboveThreshold on high→low income-floor counts (``fig-above-threshold-audit``)."""

    artifact = artifact or build_above_threshold_section_artifact(seed=seed)

    def mechanism(x, rng, epsilon=EPS_SELECT):
        return above_threshold_halt_index(x, artifact.floors, epsilon, rng, descending=False)

    panel = verify_repair_within_claimed_eps(
        mechanism,
        artifact.D,
        artifact.D_prime,
        EPS_SELECT,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        n_seeds=n_seeds,
        adversary_statistic="AboveThreshold halt index (high→low count queries)",
    )
    return audit_panel_figure(
        panel, title="AboveThreshold: highest floor clearing τ (high→low)"
    )


def _public_age_transforms() -> dict[str, Callable]:
    """Public φ functions for bounded feature scores — no private normalization."""

    scale = PUBLIC_AGE_SCALE
    return {
        "identity": lambda x: x / scale,
        "sqrt": lambda x: np.sqrt(np.maximum(x, 0) / scale),
        "log1p": lambda x: np.log1p(np.maximum(x, 0)) / np.log1p(scale),
    }


def make_report_noisy_max_selection_figure(
    artifact: ReportNoisyMaxSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_repeats: int = FIGURE_FREEZE_SELECTION_REPEATS,
) -> Figure:
    """Report-noisy-max feature selection (``fig-report-noisy-max-selection``)."""

    artifact = artifact or build_report_noisy_max_section_artifact(
        seed=seed, n_repeats=n_repeats
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=_DPI)
    names = list(artifact.exact_scores.keys())
    axes[0].bar(names, [artifact.exact_scores[n] for n in names], color="C0", alpha=0.7, label="exact")
    axes[0].bar(
        names,
        [artifact.noisy_draw[n] for n in names],
        color="none",
        edgecolor="C3",
        linewidth=2,
        label="one noisy draw",
    )
    axes[0].axhline(artifact.exact_scores[artifact.selected], color="C3", linestyle="--")
    axes[0].set_ylabel("bounded score")
    axes[0].set_title("Exact scores vs one noisy draw")
    axes[0].legend(fontsize=8)

    freqs = [artifact.selection_counts[n] / sum(artifact.selection_counts.values()) for n in names]
    axes[1].bar(names, freqs, color="C2", alpha=0.8)
    axes[1].set_ylabel("selection frequency")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Repeated runs (selected now: {artifact.selected})")
    fig.suptitle("Report-noisy-max transform selection")
    fig.tight_layout()
    return fig


def make_sparse_group_warning_figure(seed: int = DEFAULT_SEED) -> Figure:
    """Sparse subgroup instability warning (``fig-sparse-group-warning``)."""

    rng = np.random.default_rng(seed)
    groups = np.array([0] * 990 + [1] * 10)
    rng.shuffle(groups)
    target = rng.normal(50_000, 10_000, size=DEFAULT_N)
    target[groups == 1] += 80_000  # tiny subgroup looks great
    group_means = [float(np.mean(target[groups == g])) for g in [0, 1]]
    group_sizes = [int(np.sum(groups == g)) for g in [0, 1]]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=_DPI)
    ax.bar(["group 0", "group 1"], group_means, color=["C0", "C3"])
    for i, n in enumerate(group_sizes):
        ax.text(i, group_means[i], f"n={n}", ha="center", va="bottom")
    ax.set_ylabel("exact group mean")
    ax.set_title(
        "Sparse groups: repair = public min support + private support check before scoring"
    )
    fig.tight_layout()
    return fig


def make_baseline_end_to_end_audit_figure(
    artifact: BaselinePredictorSectionArtifact | None = None,
    seed: int = DEFAULT_SEED,
    *,
    n_trials: int = FIGURE_FREEZE_N_TRIALS_AUDIT,
    n_seeds: int = FIGURE_FREEZE_REPAIR_N_SEEDS,
) -> Figure:
    """Minimal baseline predictor end-to-end audit (``fig-baseline-end-to-end-audit``)."""

    artifact = artifact or build_baseline_predictor_section_artifact(seed=seed)
    age = artifact.age
    transforms = _public_age_transforms()
    transform_names = list(transforms.keys())

    def baseline_pipeline(x, rng):
        result = private_baseline_predictor(
            age,
            x,
            transform_names,
            transforms,
            candidates=PUBLIC_INCOME_CANDIDATES,
            eps_preprocess=(EPS_LOW_Q, EPS_HIGH_Q, EPS_MEAN),
            eps_select=EPS_SELECT,
            eps_groups=EPS_BASELINE_GROUPS,
            rng=rng,
        )
        return float(transform_names.index(result["selected_transform"]))

    panel = verify_repair_within_claimed_eps(
        baseline_pipeline,
        artifact.D,
        artifact.D_prime,
        artifact.total_epsilon,
        n_trials,
        DEFAULT_DELTA,
        seed=seed,
        n_seeds=n_seeds,
        adversary_statistic="baseline selected transform index",
    )
    return audit_panel_figure(panel, title="Baseline predictor end-to-end audit")


def make_composition_budget_figure(ledger: dict) -> Figure:
    """Render a budget ledger as a composition slide table figure."""

    from libdpy.assignment_specific.private_estimation.utils import budget_ledger_table

    rows = budget_ledger_table(ledger)
    fig, ax = plt.subplots(figsize=(6, 2 + 0.4 * len(rows)), dpi=_DPI)
    ax.axis("off")
    cell_text = [[r["step"], f"{r['epsilon']:.2f}"] for r in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["step", "ε"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Composition budget ledger")
    return fig

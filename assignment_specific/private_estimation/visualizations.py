"""Interactive and reusable plot components for private estimation lectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from libdpy.assignment_specific.privacy_auditing.lecture_figures import (
    _add_dp_line,
    _format_eps,
)
from libdpy.assignment_specific.private_estimation.utils import (
    AuditPanel,
    DataProvenance,
    FigureMode,
    figure_mode_suffix,
)


@dataclass(frozen=True)
class ProvenanceStyle:
    """Visual style assigned to a data-provenance family."""

    family: str
    color: str
    background: str
    hatch: str
    marker: str


PROVENANCE_PALETTE = {
    "typical": "#2f6f9f",
    "extreme_real": "#c47f17",
    "engineered": "#b43b3b",
}

PROVENANCE_STYLE: dict[DataProvenance, ProvenanceStyle] = {
    DataProvenance.REAL_DATA: ProvenanceStyle(
        "typical", PROVENANCE_PALETTE["typical"], "#eef5fb", "", "o"
    ),
    DataProvenance.PRIVATE_RUN: ProvenanceStyle(
        "typical", PROVENANCE_PALETTE["typical"], "#eef5fb", "", "o"
    ),
    DataProvenance.ORACLE_DIAGNOSTIC: ProvenanceStyle(
        "typical", PROVENANCE_PALETTE["typical"], "#eef5fb", "", "o"
    ),
    DataProvenance.EXTREME_REAL: ProvenanceStyle(
        "extreme_real", PROVENANCE_PALETTE["extreme_real"], "#fff4df", "..", "^"
    ),
    DataProvenance.CONSTRUCTED_WITNESS: ProvenanceStyle(
        "engineered", PROVENANCE_PALETTE["engineered"], "#fdeeee", "//", "s"
    ),
    DataProvenance.SYNTHETIC_WITNESS: ProvenanceStyle(
        "engineered", PROVENANCE_PALETTE["engineered"], "#fdeeee", "//", "s"
    ),
    DataProvenance.SYNTHETIC_MODEL: ProvenanceStyle(
        "engineered", PROVENANCE_PALETTE["engineered"], "#fdeeee", "//", "s"
    ),
}


def provenance_family(provenance: DataProvenance) -> str:
    """Return ``typical``, ``extreme_real``, or ``engineered`` for a provenance."""

    return PROVENANCE_STYLE[provenance].family


def adaptive_histogram_bin_edges(
    values: np.ndarray,
    *,
    min_bins: int = 20,
    max_bins: int = 180,
    min_observations_per_bin: int = 80,
    data_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Choose histogram edges from each distribution's own spread.

    Uses Scott's normal-reference rule so the bin width grows with the sample
    standard deviation and shrinks with sample size.  This keeps narrow
    components visible when overlaid with much noisier components.
    """

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float)
    if data_range is None:
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
    else:
        data_min = float(data_range[0])
        data_max = float(data_range[1])
        finite = finite[(finite >= data_min) & (finite <= data_max)]
        if finite.size == 0:
            return np.linspace(data_min, data_max, min_bins + 1)
    if data_min == data_max:
        pad = max(abs(data_min) * 0.01, 1.0)
        return np.linspace(data_min - pad, data_max + pad, min_bins + 1)
    data_range_width = data_max - data_min
    if finite.size < 2:
        n_bins = min_bins
    else:
        std = float(np.std(finite, ddof=1))
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        iqr = float(q75 - q25)
        scott_width = 3.5 * std / np.cbrt(finite.size) if std > 0 else 0.0
        fd_width = 2.0 * iqr / np.cbrt(finite.size) if iqr > 0 else 0.0
        positive_widths = [w for w in (scott_width, fd_width) if w > 0]
        bin_width = max(positive_widths) if positive_widths else 0.0
        variance_scaled_bins = (
            int(np.ceil(data_range_width / bin_width)) if bin_width > 0 else min_bins
        )
        occupancy_cap = max(min_bins, finite.size // min_observations_per_bin)
        n_bins = max(min_bins, min(max_bins, occupancy_cap, variance_scaled_bins))
    return np.linspace(data_min, data_max, n_bins + 1)


def accuracy_histogram_bin_edges(
    values: np.ndarray,
    *,
    data_range: tuple[float, float] | None = None,
    min_bins: int = 12,
    max_bins: int = 60,
    min_observations_per_bin: int = 120,
    discrete_round_decimals: int = 0,
) -> np.ndarray:
    """Histogram edges for signed-error accuracy figures.

  When a line has only a handful of distinct error values (typical for
  population-reference rank-quantile releases), bin edges are placed between
  support points instead of using a fine uniform grid that leaves empty bins.
  Otherwise falls back to ``adaptive_histogram_bin_edges`` on the display window.
    """

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float)
    if data_range is None:
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
    else:
        data_min = float(data_range[0])
        data_max = float(data_range[1])
        finite = finite[(finite >= data_min) & (finite <= data_max)]
        if finite.size == 0:
            pad = max(abs(data_min) * 0.01, 1.0)
            return np.linspace(data_min - pad, data_max + pad, min_bins + 1)
    if data_min == data_max:
        pad = max(abs(data_min) * 0.01, 1.0)
        return np.linspace(data_min - pad, data_max + pad, min_bins + 1)

    occupancy_cap = max(min_bins, finite.size // min_observations_per_bin)
    unique_sorted = np.unique(np.round(finite, discrete_round_decimals))
    n_unique = unique_sorted.size
    discrete_cap = min(max_bins, occupancy_cap, 40)
    if n_unique <= discrete_cap:
        if n_unique <= 1:
            return np.linspace(data_min, data_max, min_bins + 1)
        inner_edges = 0.5 * (unique_sorted[:-1] + unique_sorted[1:])
        edges = np.unique(
            np.clip(np.concatenate([[data_min], inner_edges, [data_max]]), data_min, data_max)
        )
        if edges.size < 3:
            return np.linspace(data_min, data_max, min_bins + 1)
        return edges.astype(float)

    return adaptive_histogram_bin_edges(
        finite,
        min_bins=min_bins,
        max_bins=min(max_bins, occupancy_cap),
        min_observations_per_bin=min_observations_per_bin,
        data_range=(data_min, data_max),
    )


def shared_accuracy_histogram_bin_edges(
    data_range: tuple[float, float],
    *,
    n_bins: int = 80,
) -> np.ndarray:
    """Uniform bin edges shared across overlaid accuracy-decomposition lines."""

    data_min, data_max = map(float, data_range)
    if data_min >= data_max:
        pad = max(abs(data_min) * 0.01, 1.0)
        return np.linspace(data_min - pad, data_max + pad, n_bins + 1)
    return np.linspace(data_min, data_max, n_bins + 1)


def probability_mass_per_bin_line(
    values: np.ndarray,
    bin_edges: np.ndarray,
    *,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Probability mass in each bin (sums to 1) on a fixed edge grid."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    counts, edges = np.histogram(finite, bins=bin_edges, density=False)
    probability = counts.astype(float) / finite.size
    if smooth and probability.size >= 3:
        probability = smooth_probability_mass(probability)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, probability


def _smooth_density_curve(
    centers: np.ndarray, density: np.ndarray
) -> np.ndarray:
    """Light smoothing for a density curve while preserving unit integral."""

    if density.size < 5:
        return density
    if centers.size > 1:
        dx = float(np.median(np.diff(centers)))
    else:
        dx = 1.0
    if dx <= 0:
        return density
    mass = density * dx
    mass = smooth_probability_mass(mass)
    return mass / dx


def accuracy_component_density_line(
    values: np.ndarray,
    panel_range: tuple[float, float],
    *,
    round_decimals: int = 0,
    min_bins: int = 12,
    max_bins: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Density curve for one signed-error component with its own bin grid.

    Each decomposition line chooses bin edges from its own spread (not a shared
  panel-wide grid), then reports a probability density so overlays stay comparable.
    """

    panel_min, panel_max = map(float, panel_range)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    finite = finite[(finite >= panel_min) & (finite <= panel_max)]
    if finite.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if finite.size == 1:
        q_low = q_high = float(finite[0])
    else:
        q_low, q_high = np.quantile(finite, [0.01, 0.99])
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if std > 0:
        pad = 4.0 * std
    else:
        pad = max(abs(float(np.median(finite))) * 0.01, 1.0)
    line_range = (float(q_low - pad), float(q_high + pad))
    line_range = (max(panel_min, line_range[0]), min(panel_max, line_range[1]))
    if line_range[1] <= line_range[0]:
        line_range = (panel_min, panel_max)

    n_unique = len(np.unique(np.round(finite, round_decimals)))
    if n_unique <= 20:
        centers, mass = discrete_probability_line(
            finite,
            round_decimals=round_decimals,
        )
        if centers.size > 1:
            width = float(np.min(np.diff(centers)))
        else:
            width = max((line_range[1] - line_range[0]) / max_bins, 1.0)
        return centers, mass / max(width, 1.0)

    bin_edges = accuracy_histogram_bin_edges(
        finite,
        data_range=line_range,
        discrete_round_decimals=round_decimals,
        min_bins=min_bins,
        max_bins=max_bins,
    )
    density, edges = np.histogram(finite, bins=bin_edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, _smooth_density_curve(centers, density)


def smooth_probability_mass(probability: np.ndarray) -> np.ndarray:
    """Light smoothing for line-style probability histograms."""

    if probability.size < 5:
        return probability
    kernel = (
        np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        if probability.size >= 25
        else np.array([1.0, 3.0, 6.0, 10.0, 6.0, 3.0, 1.0], dtype=float)
    )
    kernel /= kernel.sum()
    smoothed = np.convolve(probability, kernel, mode="same")
    total = float(np.sum(smoothed))
    if total > 0:
        smoothed /= total
    return smoothed


def kde_probability_line(
    values: np.ndarray,
    *,
    data_range: tuple[float, float],
    n_grid: int = 180,
    round_decimals: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-smoothed probability mass for sparse signed-error supports."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    data_min, data_max = data_range
    finite = finite[(finite >= data_min) & (finite <= data_max)]
    if finite.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    grid = np.linspace(data_min, data_max, n_grid)
    if finite.size == 1:
        probability = np.zeros_like(grid)
        probability[int(np.argmin(np.abs(grid - finite[0])))] = 1.0
        return grid, probability

    n_unique = len(np.unique(np.round(finite, round_decimals)))
    span = max(data_max - data_min, 1.0)
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if std > 0:
        bandwidth = std * finite.size ** (-1.0 / 5.0)
    else:
        bandwidth = span / max(n_unique, 8)
    bandwidth = max(bandwidth, span / 80.0)

    diffs = (grid[:, None] - finite[None, :]) / bandwidth
    pdf = np.exp(-0.5 * diffs * diffs).sum(axis=1) / (
        finite.size * bandwidth * np.sqrt(2.0 * np.pi)
    )
    probability = pdf * (grid[1] - grid[0])
    total = float(probability.sum())
    if total > 0:
        probability /= total
    return grid, probability


def discrete_probability_line(
    values: np.ndarray,
    *,
    round_decimals: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact probability mass at each distinct signed-error value."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    rounded = np.round(finite, round_decimals)
    unique, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(unique)
    centers = unique[order].astype(float)
    probability = counts[order].astype(float) / finite.size
    return centers, probability


def histogram_density_line(
    values: np.ndarray,
    bins: int | str | np.ndarray = 80,
    *,
    range: tuple[float, float] | None = None,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers and density values for a line-style histogram."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    density, edges = np.histogram(finite, bins=bins, range=range, density=True)
    if smooth and density.size >= 5:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        kernel /= kernel.sum()
        density = np.convolve(density, kernel, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def add_histogram_density_line(
    ax,
    values: np.ndarray,
    bins: int | np.ndarray = 80,
    *,
    range: tuple[float, float] | None = None,
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 2.0,
    smooth: bool = True,
    **plot_kwargs,
):
    """Add a histogram-density line through bin centers to an axis."""

    centers, density = histogram_density_line(values, bins, range=range, smooth=smooth)
    if centers.size == 0:
        return None
    return ax.plot(
        centers,
        density,
        label=label,
        color=color,
        linewidth=linewidth,
        **plot_kwargs,
    )[0]


def audit_panel_figure(
    panel: AuditPanel,
    *,
    title: str = "",
    figure_mode: FigureMode = FigureMode.EMPIRICAL_AUDIT,
) -> Figure:
    """Two-panel audit figure reusing the privacy-auditing ROC visual language."""

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=100)
    ax_hist, ax_roc = axes
    suffix = figure_mode_suffix(figure_mode)
    add_histogram_density_line(
        ax_hist,
        panel.samples_neg,
        adaptive_histogram_bin_edges(panel.samples_neg),
        label="D",
        color="C0",
        linewidth=2.0,
    )
    add_histogram_density_line(
        ax_hist,
        panel.samples_pos,
        adaptive_histogram_bin_edges(panel.samples_pos),
        label="D'",
        color="C1",
        linewidth=2.0,
    )
    ax_hist.axvline(panel.tau_star, color="0.25", linestyle=":", linewidth=1.2)
    ax_hist.set_xlabel(panel.adversary_statistic)
    ax_hist.set_ylabel("density")
    base_title = title or f"Audit: {panel.adversary_statistic}"
    ax_hist.set_title(f"{base_title}{suffix}")
    ax_hist.grid(axis="y", alpha=0.2)
    ax_hist.legend(fontsize=8)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax_roc.plot(panel.fpr, panel.tpr, color="C2", linewidth=2.0, label="empirical ROC")
    ax_roc.scatter(
        [panel.governing_fpr],
        [panel.governing_tpr],
        color="C3",
        s=35,
        zorder=3,
        label=f"selected $\\widehat{{\\varepsilon}}={_format_eps(panel.eps_plug)}$",
    )
    _add_dp_line(
        ax_roc,
        panel.eps_plug,
        panel.audit_result.delta,
        color="C3",
    )
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_aspect("equal", adjustable="box")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(
        f"Compute $\\varepsilon$: plug-in $\\widehat{{\\varepsilon}}="
        f"{_format_eps(panel.eps_plug)}${suffix}"
    )
    ax_roc.grid(alpha=0.2)
    ax_roc.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def audit_panels_comparison_figure(
    panels: list[AuditPanel],
    labels: list[str],
    *,
    title: str = "",
    figure_mode: FigureMode = FigureMode.EMPIRICAL_AUDIT,
) -> Figure:
    """Stack multiple empirical audit panels for mechanism comparison."""

    if len(panels) != len(labels):
        raise ValueError("panels and labels must have the same length.")
    if not panels:
        raise ValueError("At least one audit panel is required.")

    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(11.5, 3.6 * len(panels)),
        dpi=100,
        sharex=False,
    )
    suffix = figure_mode_suffix(figure_mode)
    for row, (panel, label) in enumerate(zip(panels, labels)):
        ax_hist, ax_roc = axes[row]
        add_histogram_density_line(
            ax_hist,
            panel.samples_neg,
            adaptive_histogram_bin_edges(panel.samples_neg),
            label="D",
            color="C0",
            linewidth=2.0,
        )
        add_histogram_density_line(
            ax_hist,
            panel.samples_pos,
            adaptive_histogram_bin_edges(panel.samples_pos),
            label="D'",
            color="C1",
            linewidth=2.0,
        )
        ax_hist.axvline(panel.tau_star, color="0.25", linestyle=":", linewidth=1.2)
        ax_hist.set_xlabel(panel.adversary_statistic)
        ax_hist.set_ylabel("density")
        ax_hist.set_title(f"{label}: sampled outputs{suffix}")
        ax_hist.grid(axis="y", alpha=0.2)
        ax_hist.legend(fontsize=8)

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax_roc.plot(panel.fpr, panel.tpr, color="C2", linewidth=2.0, label="empirical ROC")
        ax_roc.scatter(
            [panel.governing_fpr],
            [panel.governing_tpr],
            color="C3",
            s=35,
            zorder=3,
            label=f"selected $\\widehat{{\\varepsilon}}={_format_eps(panel.eps_plug)}$",
        )
        _add_dp_line(
            ax_roc,
            panel.eps_plug,
            panel.audit_result.delta,
            color="C3",
        )
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_aspect("equal", adjustable="box")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title(f"{label}: empirical ROC{suffix}")
        ax_roc.grid(alpha=0.2)
        ax_roc.legend(fontsize=8, loc="lower right")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _sorted_row_position(data: np.ndarray, row_index: int) -> int:
    """Return the position of ``row_index`` after sorting ``data`` ascending."""

    if row_index < 0 or row_index >= len(data):
        raise IndexError("row_index out of range for data")
    return int(np.where(np.argsort(data) == row_index)[0][0])


def sorted_neighbor_bars_figure(
    x: np.ndarray,
    x_prime: np.ndarray,
    *,
    title: str,
    provenance: DataProvenance = DataProvenance.REAL_DATA,
    low_q: float = 0.01,
    high_q: float = 0.99,
    yscale: str = "linear",
    show_quantiles: bool = True,
    show_median: bool = False,
    mu_sigma_bounds_fn: Callable[[np.ndarray], tuple[float, float, float, float]] | None = None,
    mu_sigma_k: float = 4.0,
) -> Figure:
    """Sorted-neighbor bar plot with provenance styling and optional clip lines."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=110, sharey=True)
    style = PROVENANCE_STYLE[provenance]
    datasets = [
        ("D", np.asarray(x, dtype=float)),
        ("D'", np.asarray(x_prime, dtype=float)),
    ]
    y_max = max(float(np.max(np.clip(data, 0, None))) for _, data in datasets)
    y_min = min(float(np.min(data)) for _, data in datasets)

    for ax, (label, data) in zip(axes, datasets):
        sorted_values = np.sort(data)
        ranks = np.arange(len(sorted_values))
        if provenance is DataProvenance.CONSTRUCTED_WITNESS:
            ax.fill_between(
                ranks,
                sorted_values,
                step="post",
                color=style.color,
                alpha=0.62,
                linewidth=0.0,
            )
            bulk_count = max(2, int(round(0.99 * len(sorted_values))))
            if bulk_count < len(sorted_values):
                ax.axvline(
                    bulk_count - 0.5,
                    color="0.35",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                )
        else:
            ax.bar(
                ranks,
                sorted_values,
                width=1.0,
                align="edge",
                color=style.color,
                alpha=0.62,
                edgecolor=style.color,
                linewidth=0.0,
            )

        if mu_sigma_bounds_fn is not None:
            mean_value, sigma, _, upper = mu_sigma_bounds_fn(data)
            ax.axhline(
                mean_value,
                color="black",
                linestyle="-",
                linewidth=1.2,
                label=f"mean {mean_value:,.0f}",
            )
            ax.axhline(
                upper,
                color="C4",
                linestyle="-.",
                linewidth=1.1,
                label=(
                    f"μ+{mu_sigma_k:g}σ {upper:,.0f} "
                    f"(σ={sigma:,.0f})"
                ),
            )
        else:
            mean_value = float(np.mean(data))
            ax.axhline(
                mean_value,
                color="black",
                linestyle="-",
                linewidth=1.2,
                label=f"mean {mean_value:,.0f}",
            )

        if show_quantiles:
            low_value = float(np.quantile(data, low_q))
            high_value = float(np.quantile(data, high_q))
            ax.axhline(
                low_value,
                color="C2",
                linestyle=":",
                linewidth=1.2,
                label=f"q{low_q:g} {low_value:,.0f}",
            )
            ax.axhline(
                high_value,
                color="C3",
                linestyle="--",
                linewidth=1.2,
                label=f"q{high_q:g} {high_value:,.0f}",
            )

        if show_median:
            median_value = float(np.median(data))
            ax.axhline(
                median_value,
                color="C3",
                linestyle=":",
                linewidth=1.3,
                label=f"median {median_value:,.0f}",
            )

        ax.set_title(label)
        ax.set_xlabel("sorted row index")
        ax.set_yscale(yscale)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.grid(axis="y", alpha=0.18)
        ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel("income")
    if y_max > 0:
        if yscale == "log":
            positive_min = min(
                float(np.min(data[data > 0]))
                for _, data in datasets
                if np.any(data > 0)
            )
            axes[0].set_ylim(bottom=positive_min * 0.92, top=y_max * 1.08)
        else:
            display_bottom = min(0.0, y_min * 1.08)
            axes[0].set_ylim(bottom=display_bottom, top=y_max * 1.08)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def sorted_dataset_bars_figure(
    x: np.ndarray,
    *,
    title: str,
    provenance: DataProvenance = DataProvenance.REAL_DATA,
    low_q: float = 0.01,
    high_q: float = 0.99,
    yscale: str = "linear",
    show_quantiles: bool = True,
    show_median: bool = True,
) -> Figure:
    """Single-dataset companion to ``sorted_neighbor_bars_figure``."""

    data = np.asarray(x, dtype=float)
    sorted_values = np.sort(data)
    ranks = np.arange(len(sorted_values))
    style = PROVENANCE_STYLE[provenance]
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=110)

    y_max = float(np.max(np.clip(data, 0, None)))
    y_min = float(np.min(data))

    if len(sorted_values) > 1500:
        ax.fill_between(
            ranks,
            sorted_values,
            step="post",
            color=style.color,
            alpha=0.62,
            linewidth=0.0,
        )
    else:
        ax.bar(
            ranks,
            sorted_values,
            width=1.0,
            align="edge",
            color=style.color,
            alpha=0.62,
            edgecolor=style.color,
            linewidth=0.0,
        )

    mean_value = float(np.mean(data))
    ax.axhline(
        mean_value,
        color="black",
        linestyle="-",
        linewidth=1.2,
        label=f"mean {mean_value:,.0f}",
    )

    if show_quantiles:
        low_value = float(np.quantile(data, low_q))
        high_value = float(np.quantile(data, high_q))
        ax.axhline(
            low_value,
            color="C2",
            linestyle=":",
            linewidth=1.2,
            label=f"q{low_q:g} {low_value:,.0f}",
        )
        ax.axhline(
            high_value,
            color="C3",
            linestyle="--",
            linewidth=1.2,
            label=f"q{high_q:g} {high_value:,.0f}",
        )
        if y_max > high_value * 1.05:
            ax.axhline(
                y_max,
                color="C4",
                linestyle="-.",
                linewidth=1.2,
                label=f"max {y_max:,.0f}",
            )

    if show_median:
        median_value = float(np.median(data))
        ax.axhline(
            median_value,
            color="C3",
            linestyle=":",
            linewidth=1.3,
            label=f"median {median_value:,.0f}",
        )

    ax.set_title(title)
    ax.set_xlabel("sorted row index")
    ax.set_ylabel("income")
    ax.set_yscale(yscale)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=7, loc="upper left")

    if y_max > 0:
        if yscale == "log":
            positive = data[data > 0]
            bottom = float(np.min(positive)) * 0.92 if positive.size else 1.0
            ax.set_ylim(bottom=bottom, top=y_max * 1.08)
        else:
            ax.set_ylim(bottom=min(0.0, y_min * 1.08), top=y_max * 1.08)

    fig.tight_layout()
    return fig


def analytical_gaussian_shift_figure(
    center_neg: float,
    center_pos: float,
    std: float,
    *,
    title: str = "",
    figure_mode: FigureMode = FigureMode.ANALYTICAL_EXPLANATION,
    xlabel: str = "mechanism output",
) -> Figure:
    """Two shifted Gaussian densities for neighboring datasets."""

    from scipy.stats import norm

    span = 5 * std
    grid = np.linspace(
        min(center_neg, center_pos) - span,
        max(center_neg, center_pos) + span,
        400,
    )
    pdf_neg = norm.pdf(grid, loc=center_neg, scale=std)
    pdf_pos = norm.pdf(grid, loc=center_pos, scale=std)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(grid, pdf_neg, color="C0", linewidth=2, label="$D'$: N($\\mu'$, $\\sigma^2$)")
    ax.plot(grid, pdf_pos, color="C1", linewidth=2, label="$D$: N($\\mu$, $\\sigma^2$)")
    ax.axvline(center_neg, color="C0", linestyle=":", alpha=0.6)
    ax.axvline(center_pos, color="C1", linestyle=":", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    suffix = figure_mode_suffix(figure_mode)
    ax.set_title((title or "Analytical proposal: mean shift vs Gaussian std") + suffix)
    ax.annotate(
        f"$|\\mu-\\mu'|/\\sigma = {abs(center_pos - center_neg) / std:.2f}$",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def histogram_comparison_figure(
    values: np.ndarray,
    bin_edges: np.ndarray,
    *,
    title: str = "",
    figure_mode: FigureMode | None = None,
) -> Figure:
    """Exact histogram (teaching artifact — not a private release)."""

    counts, edges = np.histogram(values, bins=bin_edges)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    ax.bar(
        centers,
        counts,
        width=width * 0.9,
        alpha=0.7,
        color="C0",
        label="exact counts",
    )
    ax.set_xlabel("income")
    ax.set_ylabel("count")
    resolved_title = title or "Oracle histogram"
    if figure_mode is not None:
        resolved_title += figure_mode_suffix(figure_mode)
    ax.set_title(resolved_title)
    ax.legend()
    fig.tight_layout()
    return fig

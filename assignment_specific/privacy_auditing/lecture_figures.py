"""Static matplotlib figures for the privacy-auditing lecture."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.stats import rv_continuous as Distribution

from libdpy.assignment_specific.privacy_auditing.animations import (
    draw_confusion_matrix_grid,
    make_audit_two_panel_figure,
)
from libdpy.assignment_specific.privacy_auditing.utils import (
    AuditResult,
    audit_point,
    empirical_roc,
    epsilon_from_roc_point,
    one_sided_epsilon_from_roc_points,
    roc_point_at_threshold,
    selected_threshold_from_empirical_roc,
)
from libdpy.privacy_mechanisms.noise import sample_outputs
from libdpy.visualization.roc_plots import (
    likelihood_ratio_unbounded,
    one_sided_privacy_bound,
    optimal_roc_curve,
)

_FIGSIZE_TWO_PANEL = (10.0, 4.0)
_FIGSIZE_SQUARE = (5.0, 5.0)
_DPI = 100


def _format_eps(epsilon: float) -> str:
    if not math.isfinite(epsilon):
        return "∞"
    if epsilon <= 0:
        return "0"
    return f"{epsilon:.3g}"


def _add_dp_line(ax, epsilon: float, delta: float, *, color: str = "C0", label: str | None = None):
    if not math.isfinite(epsilon):
        return
    x, y = one_sided_privacy_bound(epsilon, delta, res=200)
    ax.plot(x, y, color=color, linewidth=2, label=label or rf"$y=e^{{\varepsilon}}x+\delta$")


def _shade_forbidden_region(ax, epsilon: float, delta: float):
    x = np.linspace(0, 1, 200)
    if math.isfinite(epsilon):
        y_bound = np.exp(epsilon) * x + delta
    else:
        y_bound = np.ones_like(x)
    y_bound = np.clip(y_bound, 0, 1)
    ax.fill_between(x, y_bound, 1, alpha=0.15, color="C3", label="forbidden region")


def _roc_axes(ax, *, show_random_classifier_legend: bool = True):
    random_label = "random classifier" if show_random_classifier_legend else "_nolegend_"
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label=random_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_aspect("equal", adjustable="box")


def _zoom_roc_axes(
    ax,
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    min_span: float = 0.08,
    pad_fraction: float = 0.35,
) -> None:
    """Zoom ROC axes so small audit regions stay readable."""

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(x_max - x_min, min_span)
    y_span = max(y_max - y_min, min_span)
    x_pad = x_span * pad_fraction
    y_pad = y_span * pad_fraction
    ax.set_xlim(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad))
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))


def plot_dp_roc_line(epsilon: float, delta: float) -> Figure:
    """DP line as ROC bound (``fig-recall-dp-roc-line``)."""

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE, dpi=_DPI)
    _roc_axes(ax)
    _shade_forbidden_region(ax, epsilon, delta)
    _add_dp_line(ax, epsilon, delta, label=rf"$(\varepsilon={_format_eps(epsilon)},\ \delta={delta:.0g})$ bound")
    ax.set_title("The DP guarantee defines a forbidden region in the ROC plane")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def _exact_roc_panel(
    ax_pdf,
    ax_roc,
    dist_neg: Distribution,
    dist_pos: Distribution,
    delta: float,
    *,
    title: str,
    subtitle: str | None = None,
):
    resolution = 1000
    grid = np.linspace(
        min(dist_neg.ppf(0.001), dist_pos.ppf(0.001)),
        max(dist_neg.isf(0.001), dist_pos.isf(0.001)),
        resolution,
    )
    ax_pdf.plot(grid, dist_neg.pdf(grid), color="C0", label=r"$H_0$: absent")
    ax_pdf.plot(grid, dist_pos.pdf(grid), color="C1", label=r"$H_1$: present")
    ax_pdf.set_xlabel("output")
    ax_pdf.set_ylabel("density")
    ax_pdf.legend(fontsize=8)
    ax_pdf.set_title("Output distributions")

    fpr, tpr = optimal_roc_curve(dist_neg, dist_pos, resolution=resolution)
    lr_unbounded = likelihood_ratio_unbounded(dist_neg, dist_pos)
    if delta <= 0 and lr_unbounded:
        epsilon = float("inf")
        governing = None
    else:
        epsilon, governing = one_sided_epsilon_from_roc_points(fpr, tpr, delta)

    _roc_axes(ax_roc)
    ax_roc.plot(fpr, tpr, color="C2", linewidth=2, label="exact ROC")
    _add_dp_line(ax_roc, epsilon, delta, color="C0")
    if governing is not None:
        ax_roc.scatter(*governing, s=80, facecolors="none", edgecolors="C0", linewidths=2, zorder=5)
    panel_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax_roc.set_title(panel_title)
    ax_roc.legend(loc="lower right", fontsize=8)


def plot_exact_eps(
    dist_neg: Distribution,
    dist_pos: Distribution,
    delta: float | Sequence[float],
    title: str,
) -> Figure:
    """Exact ROC and tight DP line for analytic distributions."""

    deltas = [delta] if isinstance(delta, (int, float)) else list(delta)
    n_panels = len(deltas)
    fig, axes = plt.subplots(1, 2 * n_panels, figsize=(5 * n_panels, 4), dpi=_DPI)
    if n_panels == 1:
        ax_pdf, ax_roc = axes
        delta_value = deltas[0]
        fpr, tpr = optimal_roc_curve(dist_neg, dist_pos)
        lr_unbounded = likelihood_ratio_unbounded(dist_neg, dist_pos)
        if delta_value <= 0 and lr_unbounded:
            epsilon = float("inf")
        else:
            epsilon, _ = one_sided_epsilon_from_roc_points(fpr, tpr, delta_value)
        subtitle = rf"Exact ROC: $\varepsilon={_format_eps(epsilon)}$ at $\delta={delta_value:.0g}$"
        _exact_roc_panel(ax_pdf, ax_roc, dist_neg, dist_pos, delta_value, title=title, subtitle=subtitle)
    else:
        for index, delta_value in enumerate(deltas):
            ax_pdf = axes[2 * index]
            ax_roc = axes[2 * index + 1]
            fpr, tpr = optimal_roc_curve(dist_neg, dist_pos)
            lr_unbounded = likelihood_ratio_unbounded(dist_neg, dist_pos)
            if delta_value <= 0 and lr_unbounded:
                epsilon = float("inf")
            else:
                epsilon, _ = one_sided_epsilon_from_roc_points(fpr, tpr, delta_value)
            subtitle = rf"Exact ROC: $\varepsilon={_format_eps(epsilon)}$ at $\delta={delta_value:.0g}$"
            _exact_roc_panel(
                ax_pdf,
                ax_roc,
                dist_neg,
                dist_pos,
                delta_value,
                title=title,
                subtitle=subtitle,
            )

    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    return fig


def plot_sampled_densities(samples_neg: np.ndarray, samples_pos: np.ndarray) -> Figure:
    """Histograms of black-box output samples (``fig-two-logistic-sampled-densities``)."""

    fig, ax = plt.subplots(figsize=(7, 4), dpi=_DPI)
    bins = max(30, int(np.sqrt(len(samples_neg) + len(samples_pos))))
    ax.hist(samples_neg, bins=bins, density=True, alpha=0.55, color="C0", label=r"$H_0$: $X=Z$")
    ax.hist(samples_pos, bins=bins, density=True, alpha=0.55, color="C1", label=r"$H_1$: $X=1+Z$")
    ax.set_xlabel("output")
    ax.set_ylabel("density")
    ax.set_title("Samples from two shifted black-box output distributions")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_empirical_roc(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    *,
    extra_curves: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    highlight: bool = True,
) -> Figure:
    """Empirical ROC from finite samples (``fig-two-logistic-empirical-roc``)."""

    fpr, tpr, _ = empirical_roc(samples_neg, samples_pos)
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE, dpi=_DPI)
    _roc_axes(ax)
    if extra_curves:
        for curve_fpr, curve_tpr in extra_curves:
            ax.plot(curve_fpr, curve_tpr, color="0.75", linewidth=1, alpha=0.7)
    color = "C2" if highlight else "C0"
    linewidth = 2.5 if highlight else 1.5
    ax.plot(fpr, tpr, color=color, linewidth=linewidth, label="empirical ROC")
    ax.set_title("Empirical ROC from finite samples")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_empirical_roc_selected_threshold(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    delta: float,
) -> tuple[Figure, float, tuple[float, float], float]:
    """Seeded empirical ROC with governing point and ``tau_star``."""

    tau_star, (gov_fpr, gov_tpr), eps_plug = selected_threshold_from_empirical_roc(
        samples_neg, samples_pos, delta
    )
    fpr, tpr, _ = empirical_roc(samples_neg, samples_pos)

    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_TWO_PANEL, dpi=_DPI)
    bins = max(30, int(np.sqrt(len(samples_neg) + len(samples_pos))))
    axes[0].hist(samples_neg, bins=bins, density=True, alpha=0.5, color="C0", label=r"$H_0$")
    axes[0].hist(samples_pos, bins=bins, density=True, alpha=0.5, color="C1", label=r"$H_1$")
    axes[0].axvline(tau_star, color="black", linewidth=2, label=rf"$\tau_\star={tau_star:.3g}$")
    axes[0].set_xlabel("output")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Selected threshold from empirical ROC")

    _roc_axes(axes[1])
    axes[1].plot(fpr, tpr, color="C2", linewidth=2, label="empirical ROC")
    _add_dp_line(axes[1], eps_plug, delta, color="C0")
    axes[1].scatter(gov_fpr, gov_tpr, s=90, facecolors="none", edgecolors="C0", linewidths=2)
    axes[1].set_title(
        rf"Compute $\varepsilon$: plug-in $\widehat{{\varepsilon}}={_format_eps(eps_plug)}$, "
        rf"$\tau_\star={tau_star:.3g}$"
    )
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig, tau_star, (gov_fpr, gov_tpr), eps_plug


def plot_roc_resampling(
    sampler_neg: Callable[..., np.ndarray],
    sampler_pos: Callable[..., np.ndarray],
    n: int,
    seeds: Sequence[int],
    *,
    highlight_seed: int | None = None,
    highlight_samples: tuple[np.ndarray, np.ndarray] | None = None,
) -> Figure:
    """Empirical ROC variability across seeds (``fig-two-logistic-roc-resampling``)."""

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE, dpi=_DPI)
    _roc_axes(ax)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        neg = sampler_neg(n=n, rng=rng)
        pos = sampler_pos(n=n, rng=rng)
        fpr, tpr, _ = empirical_roc(neg, pos)
        is_highlight = highlight_seed is not None and seed == highlight_seed
        color = "C2" if is_highlight else "0.75"
        linewidth = 2.5 if is_highlight else 1.0
        alpha = 1.0 if is_highlight else 0.7
        ax.plot(fpr, tpr, color=color, linewidth=linewidth, alpha=alpha)

    if highlight_samples is not None:
        fpr, tpr, _ = empirical_roc(*highlight_samples)
        ax.plot(fpr, tpr, color="C2", linewidth=2.5, label="lecture seed")

    ax.set_title("Same mechanism, different samples: the empirical ROC moves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_threshold_event(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau_star: float,
) -> Figure:
    """Fixed threshold event visualization (``fig-threshold-event-two-logistic``)."""

    fpr, tpr = roc_point_at_threshold(samples_neg, samples_pos, tau_star)

    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_TWO_PANEL, dpi=_DPI)
    bins = max(30, int(np.sqrt(len(samples_neg) + len(samples_pos))))
    axes[0].hist(samples_neg, bins=bins, density=True, alpha=0.5, color="C0", label=r"$H_0$")
    axes[0].hist(samples_pos, bins=bins, density=True, alpha=0.5, color="C1", label=r"$H_1$")
    axes[0].axvline(tau_star, color="black", linewidth=2, label=rf"$\tau_\star={tau_star:.3g}$")
    for samples, color in ((samples_neg, "C0"), (samples_pos, "C1")):
        tail = samples[samples > tau_star]
        if len(tail):
            axes[0].hist(
                tail,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.35,
                color=color,
            )
    axes[0].set_xlabel("output")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[0].set_title(rf"Event $S_{{\tau_\star}}=\{{x>\tau_\star\}}$")

    _roc_axes(axes[1])
    axes[1].scatter(fpr, tpr, s=80, color="C2", zorder=5, label="empirical point")
    axes[1].set_title("Corresponding ROC point")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_audit_confusion_matrix(
    audit: AuditResult,
) -> Figure:
    """Confusion matrix and plug-in ROC point (``fig-audit-confusion-matrix``).

    Matches the final frame of ``fixed_threshold_audit_frames_from_samples``.
    """

    fig, axes = make_audit_two_panel_figure()
    draw_confusion_matrix_grid(axes[0], audit.tn, audit.fn, audit.fp, audit.tp)
    axes[0].set_title(rf"Fixed threshold $\tau_\star={audit.tau_star:.3g}$", fontsize=11)

    _roc_axes(axes[1])
    axes[1].scatter(
        audit.fpr_hat,
        audit.tpr_hat,
        s=100,
        color="C2",
        zorder=5,
        label="audit point",
    )
    _add_dp_line(
        axes[1],
        audit.eps_plug,
        audit.delta,
        color="C0",
        label=rf"plug-in $\widehat{{\varepsilon}}={_format_eps(audit.eps_plug)}$",
    )
    axes[1].set_title(
        rf"Final audit point; $\delta={audit.delta:.0g}$, "
        rf"$\widehat{{\varepsilon}}={_format_eps(audit.eps_plug)}$",
        fontsize=11,
    )
    axes[1].legend(loc="lower right", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.28)
    return fig


def plot_repeated_audit_points(
    runs: Sequence[AuditResult],
    reference: tuple[float, float, float] | None = None,
) -> Figure:
    """Repeated audit variability (``fig-repeated-audit-points``)."""

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE, dpi=_DPI)
    _roc_axes(ax)
    fprs = [run.fpr_hat for run in runs]
    tprs = [run.tpr_hat for run in runs]
    ax.scatter(fprs, tprs, s=25, alpha=0.6, color="C2", label="audit runs")
    if reference is not None:
        ref_fpr, ref_tpr, ref_eps = reference
        ax.scatter(
            ref_fpr,
            ref_tpr,
            s=120,
            marker="*",
            color="C3",
            label=rf"reference ($\varepsilon={_format_eps(ref_eps)}$)",
        )
    ax.set_title("Same mechanism and threshold; different finite audit samples")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_confidence_rectangle(audit: AuditResult) -> Figure:
    """Confidence rectangle geometry (``fig-confidence-rectangle``)."""

    if (
        audit.fpr_l is None
        or audit.fpr_u is None
        or audit.tpr_l is None
        or audit.tpr_u is None
        or audit.eps_audit is None
    ):
        raise ValueError("audit_result must include confidence-adjusted coordinates")

    fig, ax_full = plt.subplots(figsize=(7.0, 6.0), dpi=_DPI)
    _roc_axes(ax_full, show_random_classifier_legend=False)
    ax_full.scatter(
        audit.fpr_hat,
        audit.tpr_hat,
        s=45,
        color="C2",
        alpha=0.85,
        zorder=5,
    )
    overview_rect = Rectangle(
        (audit.fpr_l, audit.tpr_l),
        max(audit.fpr_u - audit.fpr_l, 0.0),
        max(audit.tpr_u - audit.tpr_l, 0.0),
        fill=False,
        edgecolor="C4",
        linewidth=1.5,
        linestyle=":",
    )
    ax_full.add_patch(overview_rect)
    ax_full.set_title("ROC plane with confidence region (overview)")

    ax_zoom = inset_axes(ax_full, width="58%", height="58%", loc="center right", borderpad=1.2)
    _roc_axes(ax_zoom, show_random_classifier_legend=False)
    confidence_rect = Rectangle(
        (audit.fpr_l, audit.tpr_l),
        max(audit.fpr_u - audit.fpr_l, 0.0),
        max(audit.tpr_u - audit.tpr_l, 0.0),
        fill=True,
        facecolor="C4",
        alpha=0.15,
        edgecolor="C4",
        linewidth=2.5,
        linestyle="--",
        label="CP rectangle",
    )
    ax_zoom.add_patch(confidence_rect)
    ax_zoom.scatter(
        audit.fpr_hat,
        audit.tpr_hat,
        s=80,
        color="C2",
        zorder=5,
        label="plug-in point",
    )
    ax_zoom.scatter(
        audit.fpr_u,
        audit.tpr_l,
        s=90,
        facecolors="none",
        edgecolors="C0",
        linewidths=2,
        label="safe corner",
    )
    _add_dp_line(
        ax_zoom,
        audit.eps_plug,
        audit.delta,
        color="C2",
        label=rf"plug-in line ($\widehat{{\varepsilon}}={_format_eps(audit.eps_plug)}$)",
    )
    _add_dp_line(
        ax_zoom,
        audit.eps_audit,
        audit.delta,
        color="C0",
        label=rf"safe line ($\varepsilon_{{\mathrm{{audit}}}}={_format_eps(audit.eps_audit)}$)",
    )
    _zoom_roc_axes(
        ax_zoom,
        [audit.fpr_l, audit.fpr_u],
        [audit.tpr_l, audit.tpr_u],
    )
    ax_zoom.set_title(
        rf"Zoom: safe $\varepsilon_{{\mathrm{{audit}}}}={_format_eps(audit.eps_audit)}$",
        fontsize=10,
    )
    mark_inset(ax_full, ax_zoom, loc1=2, loc2=4, fc="none", ec="0.45", linestyle="--", linewidth=1.2)
    handles, labels = ax_zoom.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=8,
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
    return fig


def plot_epsilon_histogram_naive_safe(
    plug: Sequence[float],
    safe: Sequence[float],
    reference: float,
) -> Figure:
    """Naive versus safe epsilon histogram (``fig-epsilon-histogram-naive-safe``)."""

    fig, ax = plt.subplots(figsize=(7, 4), dpi=_DPI)
    finite_plug = [value for value in plug if math.isfinite(value)]
    finite_safe = [value for value in safe if math.isfinite(value)]
    bins = 40
    if finite_plug:
        ax.hist(finite_plug, bins=bins, alpha=0.5, color="C0", label=r"plug-in $\widehat{\varepsilon}_{\mathrm{plug}}$")
    if finite_safe:
        ax.hist(finite_safe, bins=bins, alpha=0.5, color="C1", label=r"safe $\varepsilon_{\mathrm{audit}}$")
    ax.axvline(
        reference,
        color="C3",
        linewidth=2,
        label="reference (large simulation; not available to auditor)",
    )
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("frequency")
    ax.set_title("Naive versus safe audit lower bounds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_auditing_summary_diagram() -> Figure:
    """Workflow summary diagram (``fig-auditing-summary-diagram``)."""

    steps = [
        "black-box\nmechanism",
        "two neighboring\ndatasets",
        "fixed test",
        r"$(\widehat{\mathrm{FPR}},\widehat{\mathrm{TPR}})$",
        "confidence\nbounds",
        r"$\varepsilon$ lower bound",
    ]
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=_DPI)
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, step in enumerate(steps):
        x = index + 0.5
        ax.text(x, 0.55, step, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="0.92"))
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(index + 0.85, 0.55),
                xytext=(index + 0.15, 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_title("Privacy auditing workflow", pad=10)
    fig.tight_layout()
    return fig


def make_two_logistic_samplers(
    scale1: float = 1.0,
    scale2: float = 0.4,
):
    """Return callables that draw ``H_0`` and ``H_1`` two-logistic outputs."""

    from libdpy.privacy_mechanisms.noise import two_logistic_noise

    def sampler_neg(n: int, rng: np.random.Generator):
        return sample_outputs(
            lambda size, rng=rng: two_logistic_noise(size=size, scale1=scale1, scale2=scale2, rng=rng),
            mu=0.0,
            n=n,
            rng=rng,
        )

    def sampler_pos(n: int, rng: np.random.Generator):
        return sample_outputs(
            lambda size, rng=rng: two_logistic_noise(size=size, scale1=scale1, scale2=scale2, rng=rng),
            mu=1.0,
            n=n,
            rng=rng,
        )

    return sampler_neg, sampler_pos

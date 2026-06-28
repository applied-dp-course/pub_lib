"""Animation frames for the privacy-auditing lecture."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from libdpy.assignment_specific.privacy_auditing.utils import (
    balanced_audit_sequence,
    epsilon_from_roc_point,
    fixed_threshold_audit_state_sequence,
)
from libdpy.visualization.roc_plots import one_sided_privacy_bound

_FRAME_FIGSIZE = (13.0, 6.0)
_FRAME_DPI = 100


def make_audit_two_panel_figure() -> tuple[Figure, plt.Axes]:
    """Create a balanced two-panel audit canvas (confusion matrix + ROC)."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=_FRAME_FIGSIZE,
        dpi=_FRAME_DPI,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28},
    )
    axes[0].set_aspect("equal", adjustable="box")
    return figure, axes


def draw_confusion_matrix_grid(
    ax,
    tn: int,
    fn: int,
    fp: int,
    tp: int,
    *,
    highlight: tuple[int, int] | None = None,
) -> None:
    """Draw a deck-style confusion matrix on ``ax``.

    Columns are actual class (Negative, Positive); rows are guessed class.
    """

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    table_left = 0.20
    table_bottom = 0.04
    table_size = 0.76
    cell_size = table_size / 2.0
    table_center_x = table_left + table_size / 2.0
    table_center_y = table_bottom + table_size / 2.0
    table_top = table_bottom + table_size
    col_centers = [table_left + cell_size / 2.0, table_left + 1.5 * cell_size]
    row_centers = [table_bottom + 1.5 * cell_size, table_bottom + cell_size / 2.0]

    ax.text(table_center_x, 0.97, "Actual", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(col_centers[0], table_top + 0.065, "Negative", ha="center", va="center", fontsize=11)
    ax.text(col_centers[1], table_top + 0.065, "Positive", ha="center", va="center", fontsize=11)
    ax.text(0.035, table_center_y, "Guess", ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)
    ax.text(0.12, row_centers[0], "Negative", ha="center", va="center", fontsize=11, rotation=90)
    ax.text(0.12, row_centers[1], "Positive", ha="center", va="center", fontsize=11, rotation=90)

    cells = [
        ((1, 2), table_left, table_bottom + cell_size, "TN", tn),
        ((2, 2), table_left + cell_size, table_bottom + cell_size, "FN", fn),
        ((1, 1), table_left, table_bottom, "FP", fp),
        ((2, 1), table_left + cell_size, table_bottom, "TP", tp),
    ]
    for key, col, row, abbr, count in cells:
        facecolor = "0.95"
        edgecolor = "0.55"
        linewidth = 1.2
        if highlight is not None and key == highlight:
            facecolor = "0.82"
            edgecolor = "C0"
            linewidth = 2.0
        rect = plt.Rectangle(
            (col, row),
            cell_size,
            cell_size,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
        ax.text(
            col + cell_size / 2.0,
            row + 0.62 * cell_size,
            abbr,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            col + cell_size / 2.0,
            row + 0.36 * cell_size,
            str(count),
            ha="center",
            va="center",
            fontsize=17,
        )


def _roc_axes(ax):
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_aspect("equal", adjustable="box")


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
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2,
        label=label or rf"plug-in $\widehat{{\varepsilon}}={_format_eps(epsilon)}$",
    )


def _cell_for_update(label: int, positive_guess: bool) -> tuple[int, int]:
    if label == 0:
        return (1, 1) if positive_guess else (1, 2)
    return (2, 2) if not positive_guess else (2, 1)


def subsampled_audit_frame_indices(total_samples: int, n_frames: int) -> np.ndarray:
    """Return indices into a full audit sequence for ``n_frames`` evenly spaced steps."""

    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if n_frames >= total_samples:
        return np.arange(total_samples, dtype=int)
    return np.linspace(0, total_samples - 1, n_frames, dtype=int)


def fixed_threshold_audit_frames_from_samples(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau_star: float,
    *,
    seed: int,
    delta: float = 1e-2,
    n_frames: int = 200,
) -> Iterator[Figure]:
    """Yield audit animation frames from pre-drawn class samples.

    The audit path uses all supplied samples. ``seed`` shuffles the balanced
    experiment order and should match the seed used to draw ``samples_neg`` and
    ``samples_pos``. ``n_frames`` controls how many checkpoints are rendered from
    that path; when fewer frames than experiments are requested, the rendered
    counts jump between checkpoints but the final frame still uses all samples.
    """

    sequence_rng = np.random.default_rng(seed)

    samples_neg = np.asarray(samples_neg, dtype=float)
    samples_pos = np.asarray(samples_pos, dtype=float)
    n_per_class = len(samples_neg)
    if len(samples_pos) != n_per_class:
        raise ValueError("samples_neg and samples_pos must have the same length")
    if n_per_class == 0:
        raise ValueError("samples_neg and samples_pos must be non-empty")

    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    total_experiments = 2 * n_per_class
    rendered_frame_indices = np.linspace(
        0,
        total_experiments - 1,
        min(n_frames, total_experiments),
        dtype=int,
    )

    labels, values = balanced_audit_sequence(
        samples_neg,
        samples_pos,
        total_experiments,
        sequence_rng,
    )
    states = fixed_threshold_audit_state_sequence(
        samples_neg,
        samples_pos,
        tau_star,
        n_frames=total_experiments,
        seed=seed,
        labels=labels,
        values=values,
    )
    last_experiment_index = total_experiments - 1

    for frame_index in rendered_frame_indices:
        experiment_number = int(frame_index) + 1
        tn, fn, fp, tp, fpr, tpr = states[int(frame_index)]
        highlight = _cell_for_update(
            int(labels[frame_index]),
            bool(values[frame_index] > tau_star),
        )
        fig, axes = make_audit_two_panel_figure()
        draw_confusion_matrix_grid(
            axes[0],
            tn,
            fn,
            fp,
            tp,
            highlight=highlight,
        )
        axes[0].set_title(rf"Fixed threshold $\tau_\star={tau_star:.3g}$", fontsize=11)

        _roc_axes(axes[1])
        axes[1].scatter([fpr], [tpr], s=100, color="C2", zorder=5, label="audit point")
        if frame_index == last_experiment_index:
            eps_plug = epsilon_from_roc_point(fpr, tpr, delta)
            _add_dp_line(
                axes[1],
                eps_plug,
                delta,
                color="C0",
                label=rf"plug-in $\widehat{{\varepsilon}}={_format_eps(eps_plug)}$",
            )
            axes[1].legend(loc="lower right", fontsize=9)
            axes[1].set_title(
                rf"Final audit point; $\delta={delta:.0g}$, "
                rf"$\widehat{{\varepsilon}}={_format_eps(eps_plug)}$",
                fontsize=11,
            )
        else:
            axes[1].set_title(
                f"Audit point after {experiment_number} of {total_experiments} experiments",
                fontsize=11,
            )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.28)
        yield fig
        plt.close(fig)


def fixed_threshold_audit_frames(
    sampler_neg: Callable[..., np.ndarray],
    sampler_pos: Callable[..., np.ndarray],
    tau_star: float,
    *,
    seed: int,
    delta: float = 1e-2,
    n_frames: int = 200,
    n_per_class: int | None = None,
) -> Iterator[Figure]:
    """Yield figures for a fixed-threshold audit unfolding one sample at a time."""

    sample_rng = np.random.default_rng(seed)
    per_class = n_per_class if n_per_class is not None else n_frames // 2
    samples_neg = np.asarray(sampler_neg(n=per_class, rng=sample_rng), dtype=float)
    samples_pos = np.asarray(sampler_pos(n=per_class, rng=sample_rng), dtype=float)
    yield from fixed_threshold_audit_frames_from_samples(
        samples_neg,
        samples_pos,
        tau_star,
        seed=seed,
        delta=delta,
        n_frames=n_frames,
    )


def empirical_roc_accumulation_frames(
    sampler_neg: Callable[..., np.ndarray],
    sampler_pos: Callable[..., np.ndarray],
    n_frames: int = 200,
    rng: np.random.Generator | None = None,
) -> Iterator[Figure]:
    """Yield figures showing empirical ROC accumulation over ``n_frames`` steps."""

    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if rng is None:
        rng = np.random.default_rng()

    samples_neg = np.asarray(sampler_neg(n=n_frames, rng=rng), dtype=float)
    samples_pos = np.asarray(sampler_pos(n=n_frames, rng=rng), dtype=float)

    for frame_index in range(n_frames):
        count = frame_index + 1
        neg = samples_neg[:count]
        pos = samples_pos[:count]
        labels = np.concatenate([np.zeros(count, dtype=int), np.ones(count, dtype=int)])
        scores = np.concatenate([neg, pos])
        fpr, tpr, _ = roc_curve(labels, scores)

        fig, axes = plt.subplots(1, 2, figsize=_FRAME_FIGSIZE, dpi=_FRAME_DPI)
        axes[0].scatter(neg, np.zeros_like(neg), s=12, alpha=0.6, color="C0", label=r"$H_0$")
        axes[0].scatter(pos, np.ones_like(pos), s=12, alpha=0.6, color="C1", label=r"$H_1$")
        axes[0].set_xlabel("output")
        axes[0].set_yticks([])
        axes[0].set_title(f"Accumulating samples ({count} per class)")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
        axes[1].plot(fpr, tpr, color="C2", linewidth=2)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_title("Empirical ROC so far")
        fig.tight_layout()
        yield fig
        plt.close(fig)


def _rasterize_figure(figure: Figure) -> np.ndarray:
    """Render a figure to RGBA pixels using Agg (works in notebooks too)."""

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())


def frame_pixel_size() -> tuple[int, int]:
    """Return the raster size of animation frames for smoke checks."""

    width = int(_FRAME_FIGSIZE[0] * _FRAME_DPI)
    height = int(_FRAME_FIGSIZE[1] * _FRAME_DPI)
    return width, height


def fixed_threshold_audit_animation_html(
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau_star: float,
    *,
    seed: int,
    delta: float = 1e-2,
    n_frames: int = 200,
    fps: float = 24.0,
) -> str:
    """Return Jupyter-playable HTML for a play-once fixed-threshold audit animation."""

    from libdpy.visualization.animation_display import figure_animation_html

    return figure_animation_html(
        fixed_threshold_audit_frames_from_samples(
            samples_neg,
            samples_pos,
            tau_star,
            seed=seed,
            delta=delta,
            n_frames=n_frames,
        ),
        fps=fps,
        loop=False,
    )


_STANDALONE_PLAYER_TEMPLATE = (
    "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    "<style>html,body{{margin:0;padding:0;background:transparent}}"
    ".animation{{margin:0 auto;width:fit-content;max-width:100%}}"
    ".animation img{{max-width:100%;height:auto;display:block}}</style></head>"
    "<body>{body}</body></html>\n"
)


def write_fixed_threshold_audit_animation_player(
    output_path: str | Path,
    samples_neg: np.ndarray,
    samples_pos: np.ndarray,
    tau_star: float,
    *,
    seed: int,
    delta: float = 1e-2,
    n_frames: int = 200,
    fps: float = 24.0,
) -> Path:
    """Write a standalone interactive player HTML file for website embedding."""

    body = fixed_threshold_audit_animation_html(
        samples_neg,
        samples_pos,
        tau_star,
        seed=seed,
        delta=delta,
        n_frames=n_frames,
        fps=fps,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STANDALONE_PLAYER_TEMPLATE.format(body=body), encoding="utf-8")
    return path

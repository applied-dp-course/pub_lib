"""Plot-only visualizations for the reconstruction attacks lecture (V0–V10)."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from libdpy.attacks.reconstruction.instances import (
    corner_feasible,
)

try:
    from ipywidgets import (
        Checkbox,
        Dropdown,
        FloatSlider,
        HBox,
        HTML,
        Image as WidgetImage,
        Layout,
        VBox,
    )

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

_QUERY_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
_TRUE_B_COLOR = "#9467bd"

_BINARY_CORNERS: dict[str, list[float]] = {
    "(0, 0)": [0.0, 0.0],
    "(1, 0)": [1.0, 0.0],
    "(0, 1)": [0.0, 1.0],
    "(1, 1)": [1.0, 1.0],
}

__all__ = [
    "HAS_WIDGETS",
    "draw_2d_slabs",
    "interactive_2d_slab",
    "plot_candidate_elimination_panels",
    "plot_query_matrix_overview",
]

_PINNED_COLORS = {
    "pinned-0-correct": "#2166ac",
    "pinned-1-correct": "#b2182b",
    "pinned-incorrect": "#ff7f00",
    "unpinned": "#969696",
    "pinned-0": "#2166ac",
    "pinned-1": "#b2182b",
}


def _display_figure(fig: plt.Figure | None = None) -> None:
    """Display a figure in Jupyter without Agg ``plt.show()`` warnings."""
    if fig is None:
        fig = plt.gcf()
    try:
        get_ipython()  # type: ignore[name-defined]
        from IPython.display import display

        display(fig)
    except NameError:
        if not plt.get_backend().lower().endswith("agg"):
            plt.show()
    plt.close(fig)


def _exact_line_anchor_2d(
    q: Sequence[float],
    c: float,
    cube: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Midpoint of the line ``q @ x = c`` clipped to the square ``cube``."""
    q_arr = np.asarray(q, dtype=float)
    lo, hi = cube
    points: list[list[float]] = []

    for x in (lo, hi):
        if abs(q_arr[1]) > 1e-12:
            y = (c - q_arr[0] * x) / q_arr[1]
            if lo - 1e-9 <= y <= hi + 1e-9:
                points.append([x, y])
        elif abs(q_arr[0] * x - c) < 1e-9:
            points.extend([[x, lo], [x, hi]])

    for y in (lo, hi):
        if abs(q_arr[0]) > 1e-12:
            x = (c - q_arr[1] * y) / q_arr[0]
            if lo - 1e-9 <= x <= hi + 1e-9:
                points.append([x, y])
        elif abs(q_arr[1] * y - c) < 1e-9:
            points.extend([[lo, y], [hi, y]])

    if not points:
        return np.array([(lo + hi) / 2, (lo + hi) / 2])

    uniq: list[np.ndarray] = []
    for pt in points:
        arr = np.asarray(pt)
        if not any(np.linalg.norm(arr - u) < 1e-8 for u in uniq):
            uniq.append(arr)
    return np.vstack(uniq).mean(axis=0)


def _line_segment_2d(
    q: Sequence[float],
    c: float,
    cube: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray | None:
    """Endpoints of ``q @ x = c`` clipped to the square ``cube``."""
    q_arr = np.asarray(q, dtype=float)
    lo, hi = cube
    points: list[np.ndarray] = []

    for x in (lo, hi):
        if abs(q_arr[1]) > 1e-12:
            y = (c - q_arr[0] * x) / q_arr[1]
            if lo - 1e-9 <= y <= hi + 1e-9:
                points.append(np.array([x, y], dtype=float))
        elif abs(q_arr[0] * x - c) < 1e-9:
            points.extend(
                [np.array([x, lo], dtype=float), np.array([x, hi], dtype=float)]
            )

    for y in (lo, hi):
        if abs(q_arr[0]) > 1e-12:
            x = (c - q_arr[1] * y) / q_arr[0]
            if lo - 1e-9 <= x <= hi + 1e-9:
                points.append(np.array([x, y], dtype=float))
        elif abs(q_arr[1] * y - c) < 1e-9:
            points.extend(
                [np.array([lo, y], dtype=float), np.array([hi, y], dtype=float)]
            )

    uniq: list[np.ndarray] = []
    for pt in points:
        if not any(np.linalg.norm(pt - existing) < 1e-8 for existing in uniq):
            uniq.append(pt)

    if len(uniq) < 2:
        return None

    best_pair = (uniq[0], uniq[1])
    best_dist = -1.0
    for i, p_i in enumerate(uniq):
        for p_j in uniq[i + 1 :]:
            dist = float(np.linalg.norm(p_i - p_j))
            if dist > best_dist:
                best_dist = dist
                best_pair = (p_i, p_j)
    return np.vstack(best_pair)


def draw_2d_slabs(
    ax: Axes,
    slab_region: dict[str, Any],
    queries: Sequence[Sequence[float]],
    b: np.ndarray | None,
    alpha: float,
    *,
    query_color_indices: Sequence[int] | None = None,
    responses: Sequence[float] | None = None,
    show_true_b: bool = True,
    show_legend: bool = True,
) -> None:
    """
    Draw a precomputed 2D slab feasible region on ``ax``.

    Args:
        ax: Matplotlib axes to draw on.
        slab_region: Output of :func:`compute_2d_slab_region`.
        queries: Binary query vectors in ``{0, 1}^2``.
        b: True secret vector in ``{0, 1}^2`` (optional).
        alpha: Uniform accuracy bound for each query.
        query_color_indices: Stable palette index per query (matches checkbox colors).
        responses: Released answers ``r_i`` (defaults to exact ``q @ b``).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    X = slab_region["X"]
    Y = slab_region["Y"]
    feasible = slab_region["feasible"]
    contour_levels = slab_region["contour_levels"]

    if query_color_indices is None:
        query_color_indices = list(range(len(queries)))

    cube = slab_region.get("cube", (0.0, 1.0))
    lo_bound, hi_bound = cube
    pad = max(0.16 * (hi_bound - lo_bound), 0.12)
    b_arr = None if b is None else np.asarray(b, dtype=float)

    if b_arr is not None:
        ax.contourf(
            X,
            Y,
            feasible.astype(float),
            levels=[0.5, 1.5],
            alpha=0.25,
            colors=["#aec7e8"],
        )

    for idx, (q, (lo, hi)) in enumerate(zip(queries, contour_levels)):
        color = _QUERY_COLORS[query_color_indices[idx] % len(_QUERY_COLORS)]
        q_arr = np.asarray(q, dtype=float)
        exact = float(q_arr @ b_arr) if b_arr is not None else 0.0
        r_i = float(responses[idx]) if responses is not None else exact

        for boundary in (lo, hi):
            segment = _line_segment_2d(q_arr, boundary, cube=cube)
            if segment is not None:
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    zorder=4,
                )

        exact_segment = _line_segment_2d(q_arr, exact, cube=cube)
        if exact_segment is not None:
            ax.plot(
                exact_segment[:, 0],
                exact_segment[:, 1],
                color=color,
                linestyle="-",
                linewidth=2.2,
                zorder=5,
            )

        anchor = _exact_line_anchor_2d(q_arr, exact, cube=cube)
        q_norm = q_arr / (np.linalg.norm(q_arr) + 1e-12)
        arrow_len = 0.20 * (hi_bound - lo_bound)
        arrow = FancyArrowPatch(
            (anchor[0] - arrow_len * q_norm[0], anchor[1] - arrow_len * q_norm[1]),
            (anchor[0] + arrow_len * q_norm[0], anchor[1] + arrow_len * q_norm[1]),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=color,
            zorder=6,
            clip_on=False,
        )
        ax.add_patch(arrow)
        ax.annotate(
            f"r={r_i:.1f}",
            xy=(anchor[0], anchor[1]),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )

    ax.axvline(0.5, linestyle=":", linewidth=1, color="gray")
    ax.axhline(0.5, linestyle=":", linewidth=1, color="gray")

    for cx in (0, 1):
        for cy in (0, 1):
            if b_arr is None:
                ax.plot(
                    cx,
                    cy,
                    "o",
                    markersize=8,
                    fillstyle="none",
                    color="black",
                    alpha=0.35,
                    zorder=5,
                )
                continue
            ok = corner_feasible([cx, cy], queries, b_arr, alpha)
            if ok:
                ax.plot(cx, cy, "o", markersize=8, color="black", zorder=5)
            else:
                ax.plot(
                    cx,
                    cy,
                    "o",
                    markersize=8,
                    fillstyle="none",
                    color="black",
                    alpha=0.35,
                    zorder=5,
                )

    if show_true_b and b_arr is not None:
        ax.scatter(
            [float(b_arr[0])],
            [float(b_arr[1])],
            marker="x",
            s=90,
            c=_TRUE_B_COLOR,
            linewidths=2.2,
            zorder=7,
        )

    ax.set_xlim(lo_bound - pad, hi_bound + pad)
    ax.set_ylim(lo_bound - pad, hi_bound + pad)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    n_queries = len(queries)
    ax.set_title(
        f"{n_queries} query slab{'s' if n_queries != 1 else ''}, alpha={alpha:.2f}"
    )
    ax.set_aspect("equal")
    if show_legend:
        legend_handles = [
            Patch(facecolor="#aec7e8", alpha=0.5, label="feasible region"),
            Line2D([0], [0], color="black", linestyle="--", label="slab boundary"),
            Line2D([0], [0], color="black", linestyle="-", label="exact constraint"),
            Line2D(
                [0],
                [0],
                color="black",
                marker=">",
                linestyle="None",
                markersize=8,
                label="query normal",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markersize=8,
                label="feasible corner",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="none",
                linestyle="None",
                markersize=8,
                label="infeasible corner",
            ),
        ]
        if show_true_b and b_arr is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color=_TRUE_B_COLOR,
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=1.8,
                    label="true b",
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )


def interactive_2d_slab() -> None:
    """
    Launch an interactive 2D slab widget (revised lecture V1).

    Controls on the left; the slab picture on the right. Query slabs are
    drawn only after ``true b`` is selected, since the released answers are
    undefined before then.
    """
    if not HAS_WIDGETS:
        print("ipywidgets is not installed; 2D slab widget unavailable.")
        return

    import io

    import matplotlib
    from IPython.display import display

    from libdpy.attacks.reconstruction.instances import compute_2d_slab_region

    matplotlib.use("Agg")
    matplotlib.pyplot.ioff()

    query_specs: list[tuple[str, list[float]]] = [
        ("q = (1, 1)", [1, 1]),
        ("q = (1, 0)", [1, 0]),
        ("q = (0, 1)", [0, 1]),
    ]

    checkboxes: dict[str, Checkbox] = {}
    query_rows: list[Any] = []
    for spec_idx, (label, _q) in enumerate(query_specs):
        color = _QUERY_COLORS[spec_idx % len(_QUERY_COLORS)]
        cb = Checkbox(
            value=False,
            description=label,
            indent=False,
            layout=Layout(width="auto"),
        )
        checkboxes[label] = cb
        swatch = HTML(
            value=(
                f'<div style="width:14px;height:14px;background:{color};'
                f'border:1px solid #333;border-radius:2px;"></div>'
            )
        )
        query_rows.append(
            HBox([swatch, cb], layout=Layout(align_items="center", grid_gap="8px"))
        )

    secret_picker = Dropdown(
        options=[("(choose corner)", "")]
        + [(label, label) for label in _BINARY_CORNERS],
        value="",
        description="true b",
        layout=Layout(width="100%"),
        style={"description_width": "52px"},
    )
    alpha_slider = FloatSlider(
        min=0.0,
        max=1.0,
        step=0.05,
        value=0.3,
        description="alpha",
        readout=True,
        readout_format=".2f",
        continuous_update=False,
        layout=Layout(width="100%"),
        style={"description_width": "48px"},
    )
    legend_html = HTML(
        value=(
            "<div style='font-size:12px;line-height:1.45;border:1px solid #adb5bd;"
            "background:#fff;padding:8px 10px;width:100%;box-sizing:border-box;'>"
            "<b>Legend</b><br>"
            "<span style='display:inline-block;width:14px;height:10px;background:#aec7e8;"
            "opacity:.7;border:1px solid #8aa9c8;margin-right:6px'></span>feasible region<br>"
            "<span style='display:inline-block;width:18px;border-top:2px dashed #333;"
            "margin-right:6px'></span>slab boundary<br>"
            "<span style='display:inline-block;width:18px;border-top:2px solid #333;"
            "margin-right:6px'></span>exact constraint<br>"
            "<span style='display:inline-block;width:14px;margin-right:6px'>▶</span>query normal<br>"
            "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            "background:#111;margin-right:10px'></span>feasible corner<br>"
            "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            "border:1px solid #111;margin-right:10px'></span>infeasible corner<br>"
            f"<span style='display:inline-block;color:{_TRUE_B_COLOR};font-weight:bold;"
            "font-size:14px;margin-right:9px'>×</span>true b"
            "</div>"
        ),
        layout=Layout(width="100%"),
    )

    side_controls = VBox(
        [
            secret_picker,
            HTML("<b>Queries</b>"),
            VBox(query_rows),
            HTML("<b>Accuracy bound</b>"),
            alpha_slider,
            legend_html,
        ],
        layout=Layout(
            width="230px",
            min_width="230px",
            max_width="230px",
            flex="0 0 230px",
            padding="0",
            align_self="flex-start",
        ),
    )
    plot_image = WidgetImage(
        format="png",
        layout=Layout(width="380px", height="auto"),
    )
    plot_box = VBox(
        [plot_image],
        layout=Layout(
            border="1px solid #ced4da",
            padding="2px",
            align_self="flex-start",
        ),
    )

    state = {"blocked": True, "token": 0}

    def _show_figure(fig: plt.Figure) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        plot_image.value = buf.getvalue()

    def _update(*_: Any) -> None:
        if state["blocked"]:
            return
        state["token"] += 1
        token = state["token"]

        secret_key = secret_picker.value
        b = np.array(_BINARY_CORNERS[secret_key], dtype=float) if secret_key else None
        alpha = float(alpha_slider.value)
        active = [
            (label, q, spec_idx)
            for spec_idx, (label, q) in enumerate(query_specs)
            if checkboxes[label].value
        ]

        if token != state["token"]:
            return
        if b is None or not active:
            fig, ax = plt.subplots(figsize=(4.0, 3.4))
            ax.set_xlim(-0.16, 1.16)
            ax.set_ylim(-0.16, 1.16)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.axvline(0.5, linestyle=":", linewidth=1, color="gray")
            ax.axhline(0.5, linestyle=":", linewidth=1, color="gray")
            for cx in (0, 1):
                for cy in (0, 1):
                    ax.plot(
                        cx,
                        cy,
                        "o",
                        markersize=8,
                        fillstyle="none",
                        color="black",
                        alpha=0.35,
                        zorder=5,
                    )
            if b is not None:
                ax.scatter(
                    [float(b[0])],
                    [float(b[1])],
                    marker="x",
                    s=90,
                    c=_TRUE_B_COLOR,
                    linewidths=2.2,
                    zorder=7,
                )
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            title = (
                "Choose true b before drawing query slabs"
                if b is None
                else "Select at least one query"
            )
            ax.set_title(title)
            ax.set_aspect("equal")
            _show_figure(fig)
            return

        _labels, queries, color_indices = zip(*active)
        queries_list = list(queries)
        b_draw = b if b is not None else np.array([0.0, 0.0])
        responses = [float(np.dot(q, b_draw)) for q in queries_list]
        slab_region = compute_2d_slab_region(queries_list, b_draw, alpha)
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        draw_2d_slabs(
            ax,
            slab_region,
            queries_list,
            b,
            alpha,
            query_color_indices=list(color_indices),
            responses=responses,
            show_true_b=b is not None,
            show_legend=False,
        )
        _show_figure(fig)

    for cb in checkboxes.values():
        cb.observe(_update, names="value")
    secret_picker.observe(_update, names="value")
    alpha_slider.observe(_update, names="value")

    body = HBox(
        [plot_box, side_controls],
        layout=Layout(width="640px", align_items="flex-start", grid_gap="16px"),
    )
    display(body)
    state["blocked"] = False
    _update()


def plot_query_matrix_overview(
    Q: np.ndarray,
    b: np.ndarray,
    r: np.ndarray,
    exact: np.ndarray | None = None,
) -> None:
    """
    V0 overview: query matrix heatmap, secret barcode, and answer dot plot.

    Args:
        Q: Query matrix of shape ``(m, n)``.
        b: Secret binary vector of length ``n``.
        r: Noisy answers of length ``m`` (one per query row).
        exact: Optional exact answers ``Q @ b``.
    """
    if exact is None:
        exact = Q @ b

    m, n = Q.shape
    fig_h = max(2.0, 0.28 * m + 0.8)
    fig_w = max(6.0, 0.32 * n + 0.28 * max(n, 4) + 3.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 3, width_ratios=[n, max(2, n // 2), 4.2], wspace=0.55)

    ax_q = fig.add_subplot(gs[0])
    ax_q.imshow(
        Q,
        cmap="Greys",
        vmin=0,
        vmax=1,
        origin="upper",
        extent=(-0.5, n - 0.5, m - 0.5, -0.5),
        interpolation="nearest",
        aspect="equal",
    )
    ax_q.set_xlabel("record j")
    ax_q.set_ylabel("query i")
    ax_q.set_title(r"Query matrix $Q$")
    ax_q.set_xticks(range(n))
    ax_q.set_yticks(range(m))

    ax_b = fig.add_subplot(gs[1])
    ax_b.imshow(
        b.reshape(n, 1),
        cmap="Greys",
        vmin=0,
        vmax=1,
        origin="upper",
        aspect="equal",
        extent=(-0.5, 0.5, n - 0.5, -0.5),
        interpolation="nearest",
    )
    ax_b.set_xticks([])
    ax_b.set_yticks(range(n))
    ax_b.set_ylabel("record j")
    ax_b.set_title(r"Secret $b$")

    from matplotlib.lines import Line2D

    ax_r = fig.add_subplot(gs[2])
    rows = np.arange(m)
    ax_r.hlines(rows, exact, r, colors="gray", linestyles=":", linewidth=1, alpha=0.7)
    ax_r.scatter(r, rows, s=28, color="C0", zorder=3)
    ax_r.scatter(exact, rows, marker="x", s=36, color="C1", linewidths=1.2, zorder=4)
    ax_r.set_yticks(range(m))
    ax_r.set_xlabel("answer")
    ax_r.set_ylabel("query i")
    ax_r.set_title("released answers")
    ax_r.set_ylim(m - 0.5, -0.5)
    answer_max = int(max(np.max(r), np.max(exact), 1))
    ax_r.set_xticks(range(answer_max + 1))
    ax_r.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="C0",
                linestyle="None",
                markersize=6,
                label="released r",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="C1",
                linestyle="None",
                markersize=6,
                label="exact Qb",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        frameon=True,
    )

    fig.subplots_adjust(right=0.82)
    _display_figure(fig)


def plot_candidate_elimination_panels(
    hamming_dist: np.ndarray,
    max_residual: np.ndarray,
    feasible: np.ndarray,
    alpha: float,
    n: int,
    *,
    m: int | None = None,
    true_index: int | None = None,
) -> None:
    """
    Scatter of max ``\\ell_\\infty`` residual versus Hamming distance for all candidates.

    Args:
        hamming_dist: Hamming distance of each candidate from the truth.
        max_residual: Maximum residual of each candidate over all queries.
        feasible: Boolean mask of candidates within the accuracy slabs.
        alpha: Uniform accuracy bound (vertical reference line).
        n: Database size (for the y-axis limits).
        m: Number of queries (for the figure title).
        true_index: Optional index of the true database among candidates.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    infeasible = ~feasible

    ax.scatter(
        max_residual[infeasible],
        hamming_dist[infeasible],
        s=10,
        alpha=0.25,
        c="#bbbbbb",
        label="infeasible",
    )
    ax.scatter(
        max_residual[feasible],
        hamming_dist[feasible],
        s=14,
        alpha=0.65,
        c="#1f77b4",
        label="feasible",
    )
    if true_index is not None:
        ax.scatter(
            [max_residual[true_index]],
            [hamming_dist[true_index]],
            s=90,
            marker="*",
            c="#9467bd",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="true b",
        )

    ax.axvline(alpha, linestyle="--", color="black", linewidth=1, label=r"$\alpha$")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.5, n + 0.5)
    ax.set_xlabel(r"$\ell_\infty$")
    ax.set_ylabel("Hamming distance from true b")
    ax.set_title("Among feasible candidates, only near-truth survives")
    ax.legend(loc="upper left", fontsize=8)

    m_label = m if m is not None else "m"
    fig.suptitle(
        f"All {2**n} Boolean candidates at n={n}, m={m_label}, bounded accuracy α={alpha:g}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _display_figure(fig)



"""Plot-only 3D visualizations for the reconstruction attacks lecture (V2–V5, V8–V10)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from libdpy.attacks.reconstruction.geometry import (
    classify_corners_under_slabs,
    cube_corners,
    grid_points,
    plane_polygon,
    sample_feasible_cloud,
    slab_membership,
    slab_polygons,
)
from libdpy.attacks.reconstruction.instances import (
    compare_ols_downside2_estimators,
)
from libdpy.visualization.interactive import ControlSpec, InteractiveSpec
from libdpy.visualization.interactive_widgets import render_ipywidgets

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from ipywidgets import HTML, HBox, Layout, VBox

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

__all__ = [
    "HAS_PLOTLY",
    "HAS_WIDGETS",
    "interactive_3d_slabs",
    "plot_3d_out_of_cube_example",
]

_CUBE_EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]

_PLANE_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]
_CORNER_FEASIBLE = "#2ca02c"
_CORNER_INFEASIBLE = "#d62728"
_FEASIBLE_MESH_COLOR = "#aec7e8"
_FEASIBLE_MESH_OPACITY = 0.28
_TRUE_B_COLOR = "#9467bd"


def _require_plotly() -> None:
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for 3D reconstruction visualizations")


_SCENE_UIREVISION = "reconstruction-3d-slabs"

_CORNERS_3D: dict[str, np.ndarray] = {
    f"({int(c[0])},{int(c[1])},{int(c[2])})": c for c in cube_corners(3)
}


def _scene_layout() -> dict[str, Any]:
    return dict(
        xaxis=dict(range=[-0.05, 1.05], title="x₁", autorange=False),
        yaxis=dict(range=[-0.05, 1.05], title="x₂", autorange=False),
        zaxis=dict(range=[-0.05, 1.05], title="x₃", autorange=False),
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.55, y=1.55, z=1.15),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0),
        ),
    )


def _cube_wireframe_trace(
    color: str = "gray",
    width: float = 2.0,
    name: str = "cube",
) -> "go.Scatter3d":
    """Return a Plotly line trace for the unit-cube wireframe."""
    _require_plotly()
    corners = cube_corners(3)
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for i, j in _CUBE_EDGES:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
    )


def _fan_mesh3d(
    vertices: np.ndarray,
    color: str,
    opacity: float,
    name: str,
) -> "go.Mesh3d | None":
    """Triangulate a convex polygon fan-wise for Plotly Mesh3d."""
    _require_plotly()
    if vertices is None or len(vertices) < 3:
        return None
    center = vertices.mean(axis=0)
    pts = np.vstack([center, vertices])
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    n_verts = len(vertices)
    for t in range(1, n_verts):
        i_idx.append(0)
        j_idx.append(t)
        k_idx.append(1 if t == n_verts - 1 else t + 1)
    return go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=False,
        showscale=False,
    )


def _feasible_region_mesh_trace(
    Q: np.ndarray,
    r: np.ndarray,
    alpha: float,
    *,
    grid_level: int = 10,
    n_samples: int = 800,
) -> "go.Mesh3d | None":
    """
    Semi-transparent mesh approximating the LP-feasible region inside the cube.

    Uses a grid of slab-feasible points plus rejection samples, then their
    convex hull (the feasible set is convex).
    """
    _require_plotly()
    if alpha <= 1e-12:
        return None

    from scipy.spatial import ConvexHull

    pts = grid_points(grid_level, n=Q.shape[1])
    if pts is None:
        return None
    feasible_pts = pts[slab_membership(pts, Q, r, alpha)]

    if len(feasible_pts) < 4:
        rng = np.random.default_rng(0)
        feasible_pts = sample_feasible_cloud(Q, r, alpha, n_samples=n_samples, rng=rng)

    if len(feasible_pts) < 4:
        return None

    try:
        hull = ConvexHull(feasible_pts)
    except Exception:
        return None

    i_idx = hull.simplices[:, 0].tolist()
    j_idx = hull.simplices[:, 1].tolist()
    k_idx = hull.simplices[:, 2].tolist()
    return go.Mesh3d(
        x=feasible_pts[:, 0],
        y=feasible_pts[:, 1],
        z=feasible_pts[:, 2],
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=_FEASIBLE_MESH_COLOR,
        opacity=_FEASIBLE_MESH_OPACITY,
        name="feasible region",
        showlegend=False,
        showscale=False,
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1, roughness=0.9),
    )


def _closed_polygon_lines(
    vertices: np.ndarray,
    color: str,
    width: float,
    name: str,
) -> "go.Scatter3d | None":
    _require_plotly()
    if vertices is None or len(vertices) < 2:
        return None
    loop = np.vstack([vertices, vertices[:1]])
    return go.Scatter3d(
        x=loop[:, 0],
        y=loop[:, 1],
        z=loop[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
    )


def _normal_arrow_trace(
    q: np.ndarray,
    plane_verts: np.ndarray,
    color: str,
    name: str,
    scale: float = 0.25,
) -> "go.Scatter3d | None":
    _require_plotly()
    if plane_verts is None or len(plane_verts) == 0:
        return None
    q_arr = np.asarray(q, dtype=float)
    q_norm = q_arr / (np.linalg.norm(q_arr) + 1e-15)
    start = plane_verts.mean(axis=0)
    end = start + scale * q_norm
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode="lines+text",
        line=dict(color=color, width=4),
        text=["", ""],
        textposition="top center",
        name=name,
        showlegend=False,
    )


def _slab_legend_proxies(
    *,
    show_feasible_region: bool,
    show_true_b: bool,
) -> list["go.Scatter3d"]:
    """
    Legend-only traces mirroring the 2D slab plot (line types, not per-query).
    """
    _require_plotly()
    proxies: list[go.Scatter3d] = []
    if show_feasible_region:
        proxies.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=11, color=_FEASIBLE_MESH_COLOR, symbol="square", opacity=0.55),
                name="feasible region",
            )
        )
    proxies.extend(
        [
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="slab boundary",
            ),
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="lines",
                line=dict(color="black", width=2),
                name="exact constraint",
            ),
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="lines+markers",
                line=dict(color="black", width=2),
                marker=dict(size=5, color="black", symbol="diamond"),
                name="query normal",
            ),
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=8, color="black", symbol="circle"),
                name="feasible corner",
            ),
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=8, color="black", symbol="circle-open"),
                name="infeasible corner",
            ),
        ]
    )
    if show_true_b:
        proxies.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=7, color=_TRUE_B_COLOR, symbol="x", line=dict(width=2)),
                name="true b",
            )
        )
    return proxies


def _corner_scatter(
    corners: np.ndarray,
    feasible: np.ndarray | None = None,
    highlight: np.ndarray | None = None,
    size: float = 5.0,
) -> list["go.Scatter3d"]:
    _require_plotly()
    traces: list[go.Scatter3d] = []
    if feasible is None:
        traces.append(
            go.Scatter3d(
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                mode="markers",
                marker=dict(size=size, color="lightgray"),
                name="corners",
            )
        )
    else:
        for label, mask, marker_symbol in [
            ("feasible corner", feasible, "circle"),
            ("infeasible corner", ~feasible, "circle-open"),
        ]:
            if not np.any(mask):
                continue
            pts = corners[mask]
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=size + 2, color="black", symbol=marker_symbol),
                    name=label,
                    showlegend=False,
                )
            )
    if highlight is not None:
        traces.append(
            go.Scatter3d(
                x=[highlight[0]],
                y=[highlight[1]],
                z=[highlight[2]],
                mode="markers",
                marker=dict(size=7, color=_TRUE_B_COLOR, symbol="x", line=dict(width=2)),
                name="true b",
                showlegend=False,
            )
        )
    return traces


def _slab_plane_traces(
    Q: np.ndarray,
    r: np.ndarray,
    alpha: float,
    *,
    color_indices: Sequence[int] | None = None,
    show_planes_only: bool = False,
) -> list[Any]:
    """Build Plotly traces for exact planes or alpha-thickened slabs."""
    traces: list[Any] = []
    for idx, (q, ri) in enumerate(zip(Q, r)):
        palette_idx = color_indices[idx] if color_indices is not None else idx
        color = _PLANE_COLORS[palette_idx % len(_PLANE_COLORS)]
        if alpha <= 1e-12 or show_planes_only:
            verts = plane_polygon(q, float(ri))
            mesh = _fan_mesh3d(verts, color, 0.35, f"plane {idx + 1}")
            if mesh is not None:
                traces.append(mesh)
            outline = _closed_polygon_lines(verts, color, 2.0, f"plane {idx + 1} edge")
            if outline is not None:
                traces.append(outline)
            arrow = _normal_arrow_trace(q, verts, color, f"q{idx + 1}")
            if arrow is not None:
                traces.append(arrow)
        else:
            lo = float(ri - alpha)
            hi = float(ri + alpha)
            exact_verts = plane_polygon(q, float(ri))
            exact_outline = _closed_polygon_lines(exact_verts, color, 2.5, "exact constraint")
            if exact_outline is not None:
                traces.append(exact_outline)
            faces = slab_polygons(q, lo, hi)
            for face_name, verts in faces.items():
                mesh = _fan_mesh3d(verts, color, 0.18, f"slab {idx + 1} {face_name}")
                if mesh is not None:
                    traces.append(mesh)
                outline = _closed_polygon_lines(verts, color, 1.5, f"slab {idx + 1} {face_name}")
                if outline is not None:
                    traces.append(outline)
            if exact_verts is not None:
                arrow = _normal_arrow_trace(q, exact_verts, color, f"q{idx + 1}")
                if arrow is not None:
                    traces.append(arrow)
    return traces


def _build_3d_cube_only_figure(
    b: np.ndarray | None = None,
    *,
    show_true_b: bool = True,
    title: str = "Unit cube (select queries to add slabs)",
) -> "go.Figure":
    """Unit cube with corner markers; optional true-b highlight."""
    _require_plotly()
    fig = go.Figure()
    fig.add_trace(_cube_wireframe_trace())
    for trace in _corner_scatter(cube_corners(3), feasible=None, highlight=None):
        fig.add_trace(trace)
    if show_true_b and b is not None:
        b_arr = np.asarray(b, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=[b_arr[0]],
                y=[b_arr[1]],
                z=[b_arr[2]],
                mode="markers",
                marker=dict(size=7, color=_TRUE_B_COLOR, symbol="x", line=dict(width=2)),
                name="true b",
                showlegend=True,
            )
        )
    fig.update_layout(
        title=title,
        scene=_scene_layout(),
        width=750,
        height=650,
        uirevision=_SCENE_UIREVISION,
        showlegend=show_true_b and b is not None,
        margin=dict(r=40),
    )
    return fig


def _build_3d_slab_figure(
    b: np.ndarray,
    Q: np.ndarray,
    r: np.ndarray,
    alpha: float,
    *,
    active_query_indices: Sequence[int],
    show_true_b: bool = True,
    show_lp_region: bool = True,
) -> "go.Figure":
    """Build a fresh 3D slab figure for the selected query indices."""
    _require_plotly()
    if not active_query_indices:
        return _build_3d_cube_only_figure(
            b if show_true_b else None,
            show_true_b=show_true_b,
            title="Unit cube (select queries to add slabs)",
        )

    Q_use = Q[list(active_query_indices)]
    r_use = r[list(active_query_indices)]
    classified = classify_corners_under_slabs(Q_use, r_use, alpha)

    mesh = None
    has_feasible_mesh = False
    if show_lp_region and alpha > 1e-12:
        mesh = _feasible_region_mesh_trace(Q_use, r_use, alpha)
        has_feasible_mesh = mesh is not None

    fig = go.Figure()
    for trace in _slab_legend_proxies(
        show_feasible_region=has_feasible_mesh,
        show_true_b=show_true_b,
    ):
        fig.add_trace(trace)

    if has_feasible_mesh and mesh is not None:
        fig.add_trace(mesh)

    fig.add_trace(_cube_wireframe_trace())
    highlight = np.asarray(b, dtype=float) if show_true_b else None
    for trace in _corner_scatter(
        classified["corners"],
        feasible=classified["feasible"],
        highlight=highlight,
    ):
        fig.add_trace(trace)

    if alpha > 1e-12:
        for trace in _slab_plane_traces(
            Q_use,
            r_use,
            alpha,
            color_indices=list(active_query_indices),
        ):
            fig.add_trace(trace)
    else:
        for trace in _slab_plane_traces(
            Q_use,
            r_use,
            alpha=0.0,
            color_indices=list(active_query_indices),
            show_planes_only=True,
        ):
            fig.add_trace(trace)

    n_active = len(active_query_indices)
    fig.update_layout(
        title=f"Slabs (alpha={alpha:.2f}, {n_active} queries)",
        scene=_scene_layout(),
        width=750,
        height=650,
        uirevision=_SCENE_UIREVISION,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(r=160),
    )
    return fig


def _prepare_slab_widget_figure(fig: "go.Figure") -> "go.Figure":
    """Compact 3D slab figure for the notebook widget layout."""
    fig.update_layout(
        title=None,
        width=440,
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_layout(scene=dict(domain=dict(x=[0.0, 1.0], y=[0.0, 1.0])))
    return fig


def make_3d_slab_figure(
    true_b: str,
    alpha: float,
    query_0: bool = False,
    query_1: bool = False,
    query_2: bool = False,
    query_3: bool = False,
    query_4: bool = False,
    query_5: bool = False,
    query_6: bool = False,
    query_7: bool = False,
) -> "go.Figure":
    """Build one 3D slab state from named scalar controls."""

    Q = cube_corners(3)
    if not true_b:
        return _prepare_slab_widget_figure(
            _build_3d_cube_only_figure(
                None,
                show_true_b=False,
                title="Choose true b before drawing query slabs",
            )
        )
    if true_b not in _CORNERS_3D:
        raise ValueError(f"unknown cube corner: {true_b!r}")
    b = np.array(_CORNERS_3D[true_b], dtype=float)
    active = [
        index
        for index, enabled in enumerate(
            (
                query_0,
                query_1,
                query_2,
                query_3,
                query_4,
                query_5,
                query_6,
                query_7,
            )
        )
        if enabled
    ]
    return _prepare_slab_widget_figure(
        _build_3d_slab_figure(
            b,
            Q,
            Q @ b,
            float(alpha),
            active_query_indices=active,
            show_true_b=True,
        )
    )


def reconstruction_3d_slabs_spec() -> InteractiveSpec:
    Q = cube_corners(3)
    return InteractiveSpec(
        name="reconstruction_3d_slabs",
        artifact_name="reconstruction-3d-slabs",
        controls=(
            ControlSpec(
                name="true_b",
                kind="select",
                label="true b",
                default="",
                values=("", *_CORNERS_3D.keys()),
            ),
            *(
                ControlSpec(
                    name=f"query_{index}",
                    kind="checkbox",
                    label=f"q = ({int(q[0])},{int(q[1])},{int(q[2])})",
                    default=False,
                )
                for index, q in enumerate(Q)
            ),
            ControlSpec(
                name="alpha",
                kind="slider",
                label="alpha",
                default=0.2,
                min=0.0,
                max=1.0,
                step=0.05,
                readout_format=".2f",
            ),
        ),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets",),
        make_figure=make_3d_slab_figure,
        figure_factory=(
            "libdpy.assignment_specific.reconstruction."
            "reconstruction_3d_visualization:make_3d_slab_figure"
        ),
    )


def _slab_3d_widget_layout(figure, controls, _actions, errors):
    query_rows: list[Any] = []
    for index in range(8):
        color = _PLANE_COLORS[index % len(_PLANE_COLORS)]
        swatch = HTML(
            value=(
                f'<div style="width:14px;height:14px;background:{color};'
                f'border:1px solid #333;border-radius:2px;"></div>'
            )
        )
        query_rows.append(
            HBox(
                [swatch, controls[f"query_{index}"]],
                layout=Layout(align_items="center", grid_gap="8px"),
            )
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
            "<span style='display:inline-block;width:14px;margin-right:6px'>◆</span>query normal<br>"
            "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            "background:#111;margin-right:10px'></span>feasible corner<br>"
            "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            "border:1px solid #111;margin-right:10px'></span>infeasible corner<br>"
            f"<span style='display:inline-block;color:{_TRUE_B_COLOR};font-weight:bold;"
            "font-size:14px;margin-right:9px'>×</span>true b"
            "</div>"
        )
    )
    side_controls = VBox(
        [
            controls["true_b"],
            HTML("<b>Queries</b> (all binary q in {0,1}<sup>3</sup>)"),
            VBox(query_rows),
            HTML("<b>Accuracy bound</b>"),
            controls["alpha"],
            legend_html,
            errors,
        ],
        layout=Layout(
            width="240px",
            min_width="240px",
            max_width="240px",
            flex="0 0 240px",
            align_self="flex-start",
        ),
    )
    plot_box = VBox(
        [figure],
        layout=Layout(
            border="1px solid #ced4da",
            padding="2px",
            align_self="flex-start",
        ),
    )
    return HBox(
        [plot_box, side_controls],
        layout=Layout(width="710px", align_items="flex-start", grid_gap="16px"),
    )


def interactive_3d_slabs() -> Any:
    """
    Launch an interactive 3D slab explorer with per-query checkboxes.

    All eight binary subset-count queries in ``{0,1}^3`` are available.
    The cube is shown immediately; pick queries and ``true b`` to refine.
    """
    if not HAS_WIDGETS:
        print("ipywidgets is not installed; 3D slab widget unavailable.")
        return
    if not HAS_PLOTLY:
        print("plotly is not installed; 3D slab widget unavailable.")
        return

    from IPython.display import display

    rendered = render_ipywidgets(
        reconstruction_3d_slabs_spec(),
        layout=_slab_3d_widget_layout,
        preserve_ui_state=True,
    )
    display(rendered.root)
    return rendered.root


def _threshold_plane_trace(
    x2_value: float = 0.5,
    *,
    color: str = "#6c757d",
    opacity: float = 0.22,
) -> "go.Surface":
    """Plane ``x2 = x2_value`` clipped to the unit cube."""
    _require_plotly()
    x1 = np.array([[0.0, 1.0], [0.0, 1.0]])
    x2 = np.full_like(x1, x2_value)
    x3 = np.array([[0.0, 0.0], [1.0, 1.0]])
    return go.Surface(
        x=x1,
        y=x2,
        z=x3,
        surfacecolor=np.zeros_like(x1),
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        name="threshold x2=1/2",
        showlegend=True,
    )


def _loss_ellipsoid_trace(
    center: np.ndarray,
    Q: np.ndarray,
    level: float,
    *,
    color: str = "#17becf",
    opacity: float = 0.16,
    name: str = "least-squares level set",
) -> "go.Surface":
    """Ellipsoid ``||Qx-r||_2^2 = level`` centered at the OLS solution."""
    _require_plotly()
    eigvals, eigvecs = np.linalg.eigh(Q.T @ Q)
    radii = np.sqrt(level / np.maximum(eigvals, 1e-15))

    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    phi = np.linspace(0.0, np.pi, 18)
    sphere = np.array(
        [
            np.outer(np.sin(phi), np.cos(theta)),
            np.outer(np.sin(phi), np.sin(theta)),
            np.outer(np.cos(phi), np.ones_like(theta)),
        ]
    )
    scaled = radii[:, None, None] * sphere
    pts = np.einsum("ij,jkl->ikl", eigvecs, scaled) + center[:, None, None]

    return go.Surface(
        x=pts[0],
        y=pts[1],
        z=pts[2],
        surfacecolor=np.zeros_like(pts[0]),
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        name=name,
        showlegend=True,
    )


def plot_3d_out_of_cube_example(
    alpha: float = 0.2,
    *,
    show_loss_level: bool = False,
) -> None:
    """
    Visualize the OLS out-of-cube counterexample.

    The left panel gives the 3D cube context and the translucent displayed
    plane. The right panel shows the squared-loss landscape on the 2D plane
    through the true database, clipped OLS, and OLS:
    ``x = (1 + t, x2, -2t)``. OLS uses ``t > 0`` (hence ``x3 < 0``)
    to push ``x2`` above the rounding threshold, while the LP segment stays
    below it.

    Args:
        alpha: Uniform accuracy bound for the counterexample.
        show_loss_level: If True, draw the least-squares ellipsoid through
            the true database point.
    """
    _require_plotly()
    data = compare_ols_downside2_estimators(alpha=alpha)
    b = data["b"]
    Q = data["Q"]
    r = data["r"]
    x_ols = data["x_ols"]
    x_ols_clipped = data["x_ols_clipped"]
    lp_segment = data["lp_segment"]
    rounded_clipped = data["b_ols"]

    # Consistent encoding across both panels:
    #   colour = method  (OLS vs LP),  shape = pipeline stage.
    OLS_COLOR = "#d62728"
    LP_COLOR = "#1f77b4"
    TRUE_COLOR = _TRUE_B_COLOR
    SH_ORIG, SH_PROJ, SH_FINAL, SH_TRUE = "circle", "square", "diamond", "x"
    lp_orig = (
        np.asarray(lp_segment[0], dtype=float) + np.asarray(lp_segment[1], dtype=float)
    ) / 2.0
    lp_final = np.asarray(lp_segment[0], dtype=float)  # LP rounds onto true b=(1,0,0)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(
            "3D context: the displayed loss plane",
            "OLS loss on that plane",
        ),
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.08,
    )

    # 3D context panel.
    fig.add_trace(_cube_wireframe_trace(color="#495057", width=2.5), row=1, col=1)
    for trace in _corner_scatter(cube_corners(3), feasible=None):
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)
    plane_s = np.array([0.0, 1.0, 1.0, 0.0])
    plane_t = np.array([-0.5, -0.5, 0.25, 0.25])
    plane_pts = np.column_stack(
        [
            1.0 + plane_t,
            plane_s,
            -2.0 * plane_t,
        ]
    )
    fig.add_trace(
        go.Mesh3d(
            x=plane_pts[:, 0],
            y=plane_pts[:, 1],
            z=plane_pts[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#9467bd",
            opacity=0.16,
            name="displayed loss plane",
            showscale=False,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    plane_loop = np.vstack([plane_pts, plane_pts[:1]])
    fig.add_trace(
        go.Scatter3d(
            x=plane_loop[:, 0],
            y=plane_loop[:, 1],
            z=plane_loop[:, 2],
            mode="lines",
            line=dict(color="#9467bd", width=3),
            name="displayed loss plane edge",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    threshold_surface = _threshold_plane_trace(opacity=0.12)
    threshold_surface.showlegend = False
    fig.add_trace(threshold_surface, row=1, col=1)
    # OLS pipeline connectors (original -> projected -> final), drawn first so the
    # markers sit on top.
    fig.add_trace(
        go.Scatter3d(
            x=[x_ols[0], x_ols_clipped[0], rounded_clipped[0]],
            y=[x_ols[1], x_ols_clipped[1], rounded_clipped[1]],
            z=[x_ols[2], x_ols_clipped[2], rounded_clipped[2]],
            mode="lines",
            line=dict(color=OLS_COLOR, width=3),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=lp_segment[:, 0],
            y=lp_segment[:, 1],
            z=lp_segment[:, 2],
            mode="lines",
            line=dict(color=LP_COLOR, width=8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # OLS points: original, projected (clipped), final (rounded).
    for pt, sym in [
        (x_ols, SH_ORIG),
        (x_ols_clipped, SH_PROJ),
        (rounded_clipped, SH_FINAL),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[pt[0]],
                y=[pt[1]],
                z=[pt[2]],
                mode="markers",
                marker=dict(size=5, color=OLS_COLOR, symbol=sym),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    # LP points: original, final (rounded).
    for pt, sym in [(lp_orig, SH_ORIG), (lp_final, SH_FINAL)]:
        fig.add_trace(
            go.Scatter3d(
                x=[pt[0]],
                y=[pt[1]],
                z=[pt[2]],
                mode="markers",
                marker=dict(size=5, color=LP_COLOR, symbol=sym),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter3d(
            x=[b[0]],
            y=[b[1]],
            z=[b[2]],
            mode="markers",
            marker=dict(size=5, color=TRUE_COLOR, symbol=SH_TRUE, line=dict(width=2)),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if show_loss_level:
        true_loss = float(np.sum((Q @ b - r) ** 2))
        fig.add_trace(
            _loss_ellipsoid_trace(
                np.asarray(x_ols, dtype=float),
                Q,
                true_loss,
                name="LS level through true b",
            ),
            row=1,
            col=1,
        )

    # 2D plane panel: horizontal axis is x2; vertical axis is x3 (the third
    # coordinate). The displayed plane is x = (1 - x3/2, x2, x3); it contains
    # true b, clipped OLS, and OLS. OLS sits at x3 = -0.4, below the cube floor.
    x2_grid = np.linspace(-0.05, 1.05, 190)
    x3_grid = np.linspace(-0.85, 0.25, 190)
    X2, X3 = np.meshgrid(x2_grid, x3_grid)
    X1 = 1.0 - 0.5 * X3
    points = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)
    residuals = points @ Q.T - r
    loss = np.sum(residuals**2, axis=1).reshape(X2.shape)
    loss_floor = 1e-4
    loss_color = np.log10(loss + loss_floor)
    # Clip the low end of the log color range to 0.01: near-zero losses only
    # occur right at the OLS minimum, so spending most of the colormap there
    # leaves the bulk of the panel (losses ~1-4) saturated at one color. Anchoring
    # zmin at 0.01 lets that 0.01-4 band span most of the colormap and read as a
    # smooth gradient.
    loss_ticks = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 4.0])
    fig.add_trace(
        go.Heatmap(
            x=x2_grid,
            y=x3_grid,
            z=loss_color,
            customdata=loss,
            zmin=float(np.log10(0.01 + loss_floor)),
            zmax=float(np.log10(np.nanmax(loss) + loss_floor)),
            colorscale="Viridis",
            opacity=0.92,
            colorbar=dict(
                title="OLS loss<br>(log color)",
                x=1.01,
                len=0.58,
                tickmode="array",
                tickvals=np.log10(loss_ticks + loss_floor),
                ticktext=["≤0.01", "0.03", "0.1", "0.3", "1", "2", "4"],
            ),
            hovertemplate="x2=%{x:.2f}<br>x3=%{y:.2f}<br>loss=%{customdata:.3g}<extra></extra>",
            name="OLS loss landscape",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Contour(
            x=x2_grid,
            y=x3_grid,
            z=loss,
            contours=dict(
                coloring="lines",
                showlabels=False,
                start=0.01,
                end=0.05,
                size=0.04,
            ),
            line=dict(width=1.5, color="rgba(255,255,255,0.95)"),
            colorscale=[[0.0, "rgba(255,255,255,0.95)"], [1.0, "rgba(255,255,255,0.95)"]],
            showscale=False,
            showlegend=False,
            name="near-minimum loss contours",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Valid cube slice: outline only, so the loss heatmap shows through inside
    # the cube (where the OLS minimum is *not* — that is the whole point). On
    # this plane the cube is x2 in [0,1], x3 in [0,1]; its floor x3=0 is the
    # boundary OLS drops below.
    valid_x = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    valid_x3 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    fig.add_trace(
        go.Scatter(
            x=valid_x,
            y=valid_x3,
            mode="lines",
            line=dict(color="#f0f0f0", width=2.5),
            name="valid cube slice",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[lp_segment[0, 1], lp_segment[1, 1]],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(color=LP_COLOR, width=8),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.5, 0.5],
            y=[-0.8, 0.2],
            mode="lines",
            line=dict(color="#6c757d", dash="dot", width=2),
            name="rounding threshold x2=1/2",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # OLS pipeline connector (original -> projected -> final). Vertical axis = x3.
    fig.add_trace(
        go.Scatter(
            x=[x_ols[1], x_ols_clipped[1], rounded_clipped[1]],
            y=[x_ols[2], x_ols_clipped[2], rounded_clipped[2]],
            mode="lines",
            line=dict(color=OLS_COLOR, width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Points, with a legend that decodes colour (method) and shape (stage).
    for x_val, y_val, color, symbol, label in [
        (float(x_ols[1]), float(x_ols[2]), OLS_COLOR, SH_ORIG, "OLS — original"),
        (
            float(x_ols_clipped[1]),
            float(x_ols_clipped[2]),
            OLS_COLOR,
            SH_PROJ,
            "OLS — projected (clipped)",
        ),
        (
            float(rounded_clipped[1]),
            float(rounded_clipped[2]),
            OLS_COLOR,
            SH_FINAL,
            "OLS — final (rounded)",
        ),
        (float(lp_orig[1]), float(lp_orig[2]), LP_COLOR, SH_ORIG, "LP — original"),
        (float(lp_final[1]), float(lp_final[2]), LP_COLOR, SH_FINAL, "LP — final (rounded)"),
        (float(b[1]), float(b[2]), TRUE_COLOR, SH_TRUE, "true b"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[x_val],
                y=[y_val],
                mode="markers",
                marker=dict(size=11, color=color, symbol=symbol, line=dict(width=1)),
                name=label,
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="OLS fits the answers by leaving the cube",
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[-0.05, 1.28], title="x1", autorange=False),
            yaxis=dict(range=[-0.05, 1.05], title="x2", autorange=False),
            zaxis=dict(range=[-0.58, 1.05], title="x3", autorange=False),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.65, y=1.45, z=1.05),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        width=980,
        height=560,
        uirevision=_SCENE_UIREVISION,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.04,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=70, t=70, b=70),
    )
    fig.update_xaxes(
        title_text="x2",
        range=[-0.05, 1.05],
        zeroline=False,
        row=1,
        col=2,
    )
    # Vertical axis = x3, negative at the bottom: OLS escapes to x3 = -0.4 below
    # the cube floor (x3 = 0); the valid cube sits at x3 >= 0 above it.
    fig.update_yaxes(
        title_text="x₃  (cube floor at x₃ = 0)",
        range=[-0.8, 0.2],
        zeroline=False,
        row=1,
        col=2,
    )
    fig.show()

"""Notebook helpers for displaying matplotlib figure-sequence animations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from IPython.display import HTML, display
from matplotlib.animation import ArtistAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _rasterize_figure(figure: Figure) -> np.ndarray:
    """Render a figure to RGBA pixels using Agg (works in notebooks too)."""

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())


def figure_animation_html(
    frames: Iterable[Figure],
    *,
    fps: float = 24.0,
    loop: bool = False,
) -> str:
    """Return Jupyter-playable HTML for a matplotlib figure-sequence animation."""

    if fps <= 0:
        raise ValueError("fps must be positive")

    import matplotlib.pyplot as plt

    rasters: list[np.ndarray] = []
    for frame_figure in frames:
        if not isinstance(frame_figure, Figure):
            raise TypeError(f"expected matplotlib Figure, got {type(frame_figure)!r}")
        rasters.append(_rasterize_figure(frame_figure))
        plt.close(frame_figure)

    if not rasters:
        raise ValueError("animation produced no frames")

    height, width = rasters[0].shape[:2]
    dpi = 100.0
    host = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(host)
    axis = host.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.axis("off")
    artists = [[axis.imshow(array, animated=True)] for array in rasters]
    animation = ArtistAnimation(
        host,
        artists,
        interval=1000.0 / fps,
        blit=False,
        repeat=loop,
    )
    html = animation.to_jshtml(fps=fps, default_mode="loop" if loop else "once")
    plt.close(host)
    return html


def display_figure_animation(
    frames: Iterable[Figure] | Iterator[Figure],
    *,
    fps: float = 24.0,
    loop: bool = False,
) -> HTML:
    """Display a figure-sequence animation in a Jupyter notebook."""

    payload = HTML(figure_animation_html(frames, fps=fps, loop=loop))
    display(payload)
    return payload

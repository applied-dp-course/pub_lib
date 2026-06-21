"""Transitional ipywidgets renderer for pure Matplotlib figure factories."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, MutableMapping

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .interactive import InteractiveSpec
from .interactive_widgets import _control_widget

ImageLayout = Callable[
    [widgets.Image, Mapping[str, widgets.Widget], widgets.Output], widgets.Widget
]


@dataclass
class RenderedImageInteractive:
    """Live widget bundle for a Matplotlib image interactive."""

    root: widgets.Widget
    image: widgets.Image
    controls: Mapping[str, widgets.Widget]
    state: MutableMapping[str, Any]
    update: Callable[[], None]


def _default_layout(
    image: widgets.Image,
    controls: Mapping[str, widgets.Widget],
    errors: widgets.Output,
) -> widgets.Widget:
    return widgets.VBox(
        [
            image,
            widgets.HBox(
                list(controls.values()),
                layout=widgets.Layout(
                    width="100%",
                    display="flex",
                    flex_flow="row wrap",
                    grid_gap="8px",
                ),
            ),
            errors,
        ]
    )


def _figure_png(figure: Figure, *, dpi: int) -> bytes:
    buffer = io.BytesIO()
    FigureCanvasAgg(figure)
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    return buffer.getvalue()


def render_matplotlib_ipywidgets(
    spec: InteractiveSpec,
    *,
    layout: ImageLayout | None = None,
    dpi: int = 120,
) -> RenderedImageInteractive:
    """Render a Matplotlib factory without changing the global backend."""

    if "ipywidgets" not in spec.allowed_backends:
        raise ValueError(f"{spec.name} does not allow the ipywidgets backend")
    if spec.actions:
        raise NotImplementedError("Matplotlib image actions are not implemented")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    controls = {control.name: _control_widget(control) for control in spec.controls}
    state: MutableMapping[str, Any] = spec.new_state()
    image = widgets.Image(format="png")
    errors = widgets.Output()

    def current_values() -> dict[str, float | int | str | bool]:
        values: dict[str, float | int | str | bool] = {}
        specs = {control.name: control for control in spec.controls}
        for name, widget in controls.items():
            value = widget.value
            if isinstance(specs[name].default, int) and not isinstance(specs[name].default, bool):
                value = int(round(value))
            values[name] = value
        return values

    def update() -> None:
        try:
            figure = spec.make_figure(**spec.figure_kwargs(current_values(), state))
            if not isinstance(figure, Figure):
                raise TypeError("Matplotlib figure factories must return matplotlib.figure.Figure")
            image.value = _figure_png(figure, dpi=dpi)
        except Exception as error:
            errors.clear_output(wait=True)
            with errors:
                print(f"{type(error).__name__}: {error}")
            return
        errors.clear_output(wait=True)

    for control in controls.values():
        control.observe(lambda _change: update(), names="value")

    update()
    root = (layout or _default_layout)(image, controls, errors)
    return RenderedImageInteractive(
        root=root,
        image=image,
        controls=controls,
        state=state,
        update=update,
    )

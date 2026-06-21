"""Generic ipywidgets renderer for backend-neutral interactive plot specs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, MutableMapping

import ipywidgets as widgets
import plotly.graph_objects as go

from .interactive import ActionSpec, ControlSpec, InteractiveSpec

ControlLayout = Callable[
    [go.FigureWidget, Mapping[str, widgets.Widget], Mapping[str, widgets.Button], widgets.Output],
    widgets.Widget,
]


@dataclass
class RenderedInteractive:
    """Live widget bundle returned by :func:`render_ipywidgets`."""

    root: widgets.Widget
    figure: go.FigureWidget
    controls: Mapping[str, widgets.Widget]
    actions: Mapping[str, widgets.Button]
    state: MutableMapping[str, Any]
    update: Callable[[], None]


def _slider_widget(control: ControlSpec) -> widgets.Widget:
    values = (control.default, control.min, control.max, control.step)
    is_integer = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
    if control.slider_scale == "log":
        return widgets.FloatLogSlider(
            value=control.default,
            base=10,
            min=math.log10(control.min),
            max=math.log10(control.max),
            step=control.step,
            description=control.label,
            continuous_update=control.continuous,
            readout_format=control.readout_format
            or (".0f" if isinstance(control.default, int) else ".3g"),
            style={"description_width": "initial"},
            layout=widgets.Layout(width="auto", min_width="280px"),
        )
    slider_type = widgets.IntSlider if is_integer else widgets.FloatSlider
    kwargs: dict[str, Any] = {}
    if control.readout_format is not None:
        kwargs["readout_format"] = control.readout_format
    return slider_type(
        value=control.default,
        min=control.min,
        max=control.max,
        step=control.step,
        description=control.label,
        continuous_update=control.continuous,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto", min_width="220px"),
        **kwargs,
    )


def _control_widget(control: ControlSpec) -> widgets.Widget:
    if control.kind == "slider":
        return _slider_widget(control)
    if control.kind == "select":
        return widgets.Dropdown(
            options=list(control.values or ()),
            value=control.default,
            description=control.label,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="auto", min_width="220px"),
        )
    if control.kind == "checkbox":
        return widgets.Checkbox(
            value=bool(control.default),
            description=control.label,
            indent=False,
        )
    if control.kind == "button_group":
        return widgets.ToggleButtons(
            options=list(control.values or ()),
            value=control.default,
            description=control.label,
            style={"description_width": "initial"},
        )
    raise ValueError(f"unsupported control kind: {control.kind!r}")


def _capture_plotly_ui_state(figure: go.FigureWidget) -> dict[str, Any]:
    state: dict[str, Any] = {}
    layout = figure.layout.to_plotly_json()
    for key, value in layout.items():
        if key.startswith("scene") and isinstance(value, dict) and value.get("camera"):
            state[f"{key}.camera"] = value["camera"]
    return state


def replace_figure_widget(
    target: go.FigureWidget,
    source: go.Figure,
    *,
    preserve_ui_state: bool = True,
) -> None:
    """Replace a FigureWidget's complete figure while preserving useful UI state."""

    ui_state = _capture_plotly_ui_state(target) if preserve_ui_state else {}
    with target.batch_update():
        target.data = []
        target.add_traces(list(source.data))
        target.layout = source.layout
        try:
            target.frames = source.frames
        except (AttributeError, ValueError):
            pass
        for key, value in ui_state.items():
            scene_name, _, property_name = key.partition(".")
            scene = getattr(target.layout, scene_name, None)
            if scene is not None and property_name == "camera":
                scene.camera = value


def _default_layout(
    figure: go.FigureWidget,
    controls: Mapping[str, widgets.Widget],
    actions: Mapping[str, widgets.Button],
    errors: widgets.Output,
) -> widgets.Widget:
    control_row = widgets.HBox(
        [*controls.values(), *actions.values()],
        layout=widgets.Layout(
            width="100%",
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            grid_gap="8px",
        ),
    )
    return widgets.VBox(
        [figure, control_row, errors],
        layout=widgets.Layout(width="100%"),
    )


def render_ipywidgets(
    spec: InteractiveSpec,
    *,
    layout: ControlLayout | None = None,
    preserve_ui_state: bool = True,
    figure_transform: Callable[[go.Figure], go.Figure] | None = None,
) -> RenderedInteractive:
    """Render any supported ``InteractiveSpec`` as a live notebook widget."""

    if "ipywidgets" not in spec.allowed_backends:
        raise ValueError(f"{spec.name} does not allow the ipywidgets backend")

    controls = {control.name: _control_widget(control) for control in spec.controls}
    state: MutableMapping[str, Any] = spec.new_state()
    errors = widgets.Output()

    def current_values() -> dict[str, float | int | str | bool]:
        values: dict[str, float | int | str | bool] = {}
        specs_by_name = {control.name: control for control in spec.controls}
        for name, control in controls.items():
            value = control.value
            if isinstance(specs_by_name[name].default, int) and not isinstance(
                specs_by_name[name].default, bool
            ):
                value = int(round(value))
            values[name] = value
        return values

    def make_figure() -> go.Figure:
        figure = spec.make_figure(**spec.figure_kwargs(current_values(), state))
        if not isinstance(figure, go.Figure):
            raise TypeError("ipywidgets figure factories must return plotly.go.Figure")
        if figure_transform is not None:
            figure = figure_transform(figure)
        return figure

    figure_widget = go.FigureWidget(make_figure())

    def update() -> None:
        try:
            updated = make_figure()
        except Exception as error:
            errors.clear_output(wait=True)
            with errors:
                print(f"{type(error).__name__}: {error}")
            return
        errors.clear_output(wait=True)
        replace_figure_widget(
            figure_widget,
            updated,
            preserve_ui_state=preserve_ui_state,
        )

    for control in controls.values():
        control.observe(lambda _change: update(), names="value")

    action_widgets: dict[str, widgets.Button] = {}
    for action in spec.actions:
        button = widgets.Button(
            description=action.label,
            button_style=action.button_style,
        )

        def run_action(
            _button: widgets.Button,
            *,
            action_spec: ActionSpec = action,
        ) -> None:
            replacement = action_spec.handler(state, current_values())
            if replacement is not None:
                state.clear()
                state.update(replacement)
            update()

        button.on_click(run_action)
        action_widgets[action.name] = button

    root = (layout or _default_layout)(figure_widget, controls, action_widgets, errors)
    return RenderedInteractive(
        root=root,
        figure=figure_widget,
        controls=controls,
        actions=action_widgets,
        state=state,
        update=update,
    )


def one_slider_plot(
    make_figure: Callable[..., go.Figure],
    slider: ControlSpec,
    *,
    fixed_kwargs: Mapping[str, Any] | None = None,
    name: str = "one_slider_plot",
    artifact_name: str = "one-slider-plot",
) -> RenderedInteractive:
    """Render a one-slider figure factory through the shared engine."""

    if slider.kind != "slider":
        raise ValueError("one_slider_plot requires a slider ControlSpec")
    spec = InteractiveSpec(
        name=name,
        artifact_name=artifact_name,
        controls=(slider,),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets",),
        make_figure=make_figure,
        figure_factory=f"{make_figure.__module__}:{make_figure.__name__}",
        fixed_kwargs=dict(fixed_kwargs or {}),
    )
    return render_ipywidgets(spec)


def two_slider_plot(
    make_figure: Callable[..., go.Figure],
    first: ControlSpec,
    second: ControlSpec,
    *,
    fixed_kwargs: Mapping[str, Any] | None = None,
    name: str = "two_slider_plot",
    artifact_name: str = "two-slider-plot",
) -> RenderedInteractive:
    """Render a two-slider figure factory through the shared engine."""

    if first.kind != "slider" or second.kind != "slider":
        raise ValueError("two_slider_plot requires two slider ControlSpecs")
    if first.name == second.name:
        raise ValueError("slider names must be distinct")
    spec = InteractiveSpec(
        name=name,
        artifact_name=artifact_name,
        controls=(first, second),
        preferred_backend="ipywidgets",
        allowed_backends=("ipywidgets",),
        make_figure=make_figure,
        figure_factory=f"{make_figure.__module__}:{make_figure.__name__}",
        fixed_kwargs=dict(fixed_kwargs or {}),
    )
    return render_ipywidgets(spec)

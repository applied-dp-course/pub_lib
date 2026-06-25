"""Generic ipywidgets renderer for backend-neutral interactive plot specs."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BasePlotlyType

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


def _attach_formatted_slider_description(
    slider: widgets.FloatLogSlider | widgets.FloatSlider | widgets.IntSlider,
    control: ControlSpec,
) -> None:
    if control.readout_formatter is None:
        return

    def sync_description(change: dict[str, Any]) -> None:
        slider.description = (
            f"{control.label} = {control.readout_formatter(float(change['new']))}"
        )

    slider.readout = False
    slider.observe(sync_description, names="value")
    slider.description = f"{control.label} = {control.readout_formatter(float(control.default))}"


def _slider_widget(control: ControlSpec) -> widgets.Widget:
    values = (control.default, control.min, control.max, control.step)
    is_integer = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
    if control.slider_scale == "log":
        slider = widgets.FloatLogSlider(
            value=control.default,
            base=10,
            min=math.log10(control.min),
            max=math.log10(control.max),
            step=control.step,
            description=control.label,
            continuous_update=control.continuous,
            readout=control.readout_formatter is None,
            readout_format=control.readout_format
            or (".0f" if isinstance(control.default, int) else ".3g"),
            style={"description_width": "initial"},
            layout=widgets.Layout(width="auto", min_width="280px"),
        )
        _attach_formatted_slider_description(slider, control)
        return slider
    slider_type = widgets.IntSlider if is_integer else widgets.FloatSlider
    kwargs: dict[str, Any] = {}
    if control.readout_format is not None:
        kwargs["readout_format"] = control.readout_format
    slider = slider_type(
        value=control.default,
        min=control.min,
        max=control.max,
        step=control.step,
        description=control.label,
        continuous_update=control.continuous,
        readout=control.readout_formatter is None,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto", min_width="220px"),
        **kwargs,
    )
    _attach_formatted_slider_description(slider, control)
    return slider


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
    if control.kind == "toggle_button":
        button = widgets.ToggleButton(
            value=bool(control.default),
            description=control.label,
            tooltip=control.description or control.label,
            layout=widgets.Layout(width="auto", min_width="120px", height="30px"),
        )
        button.style.font_size = "0.95rem"
        return button
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


def _trace_structure_signature(trace: go.BaseTraceType) -> tuple[Any, ...]:
    """Trace role for structure checks (exclude data-derived ``name`` labels)."""

    return (
        trace.__class__.__name__,
        getattr(trace, "xaxis", None),
        getattr(trace, "yaxis", None),
        getattr(trace, "legend", None),
        getattr(trace, "legendgroup", None),
    )


def _figures_share_structure(source: go.Figure, target: go.FigureWidget) -> bool:
    if len(source.data) != len(target.data):
        return False
    return all(
        _trace_structure_signature(source_trace) == _trace_structure_signature(target_trace)
        for source_trace, target_trace in zip(source.data, target.data)
    )


def _plotly_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_plotly_values_equal(left[key], right[key]) for key in left)
    left_is_seq = isinstance(left, (list, tuple, np.ndarray))
    right_is_seq = isinstance(right, (list, tuple, np.ndarray))
    if left_is_seq or right_is_seq:
        if not (left_is_seq and right_is_seq):
            return False
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        if left_arr.shape != right_arr.shape:
            return False
        # Numeric arrays compare with equal_nan; string/object arrays (e.g. categorical
        # axis labels) need plain array_equal, which always returns a scalar bool.
        if left_arr.dtype.kind in "biufc" and right_arr.dtype.kind in "biufc":
            return bool(np.array_equal(left_arr, right_arr, equal_nan=True))
        return bool(np.array_equal(left_arr, right_arr))
    # Scalar fallback: coerce any stray array result (e.g. a 0-d array) to a bool.
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(result.all())
    return bool(result)


_SKIP_TRACE_KEYS = frozenset({"uid", "type", "meta", "frame"})


def _is_plotly_compound(obj: Any) -> bool:
    """Whether ``obj`` is a Plotly compound (so we can patch leaf properties on it).

    The recursion guard must NOT use ``hasattr(obj, "keys")``: Plotly compound
    objects (axes, markers, lines, ...) do not implement ``keys``. Replacing a whole
    sub-object instead of patching the changed leaf makes the FigureWidget transmit
    every property of that sub-object — and any property with a ``'plot'`` edit type
    (e.g. ``xaxis.domain``) forces Plotly to replot the entire figure, repainting all
    subplot backgrounds even though only one leaf (e.g. ``xaxis.range``) changed.
    """

    return isinstance(obj, BasePlotlyType)


def _assign_plotly_property(target: Any, key: str, source_value: Any, target_value: Any) -> None:
    if _plotly_values_equal(source_value, target_value):
        return
    if isinstance(source_value, dict):
        child = getattr(target, key, None)
        if _is_plotly_compound(child):
            for subkey, subvalue in source_value.items():
                current = child[subkey] if subkey in child else None
                if not _plotly_values_equal(subvalue, current):
                    _assign_plotly_property(child, subkey, subvalue, current)
            return
    target[key] = source_value


def _apply_trace_diff(target_trace: go.BaseTraceType, source_trace: go.BaseTraceType) -> None:
    source_json = source_trace.to_plotly_json()
    target_json = target_trace.to_plotly_json()
    for key, source_value in source_json.items():
        if key in _SKIP_TRACE_KEYS:
            continue
        _assign_plotly_property(target_trace, key, source_value, target_json.get(key))


def _apply_nested_layout_diff(
    target: Any,
    source_value: Mapping[str, Any],
    last_value: Mapping[str, Any],
) -> None:
    for key in set(source_value) | set(last_value):
        source_item = source_value.get(key)
        last_item = last_value.get(key)
        if isinstance(source_item, dict) and isinstance(last_item, dict):
            if not _plotly_values_equal(source_item, last_item):
                child = getattr(target, key, None)
                if _is_plotly_compound(child):
                    _apply_nested_layout_diff(child, source_item, last_item)
                else:
                    setattr(target, key, copy.deepcopy(source_item))
            continue
        if not _plotly_values_equal(source_item, last_item):
            setattr(target, key, copy.deepcopy(source_item))


def _apply_layout_diff(
    target_layout: go.Layout,
    source_layout: Mapping[str, Any],
    last_factory_layout: Mapping[str, Any],
) -> None:
    for key in set(source_layout) | set(last_factory_layout):
        source_value = source_layout.get(key)
        last_value = last_factory_layout.get(key)
        if isinstance(source_value, dict) and isinstance(last_value, dict):
            if not _plotly_values_equal(source_value, last_value):
                target_attr = getattr(target_layout, key, None)
                if _is_plotly_compound(target_attr):
                    _apply_nested_layout_diff(target_attr, source_value, last_value)
                else:
                    setattr(target_layout, key, copy.deepcopy(source_value))
            continue
        if not _plotly_values_equal(source_value, last_value):
            setattr(target_layout, key, copy.deepcopy(source_value))


def _restore_plotly_ui_state(target: go.FigureWidget, ui_state: Mapping[str, Any]) -> None:
    for key, value in ui_state.items():
        scene_name, _, property_name = key.partition(".")
        scene = getattr(target.layout, scene_name, None)
        if scene is not None and property_name == "camera":
            scene.camera = value


def _replace_figure_widget_full(
    target: go.FigureWidget,
    source: go.Figure,
    *,
    preserve_ui_state: bool = True,
) -> None:
    ui_state = _capture_plotly_ui_state(target) if preserve_ui_state else {}
    with target.batch_update():
        target.data = []
        target.add_traces(list(source.data))
        target.layout = source.layout
        try:
            target.frames = source.frames
        except (AttributeError, ValueError):
            pass
        _restore_plotly_ui_state(target, ui_state)


def _update_figure_widget_in_place(
    target: go.FigureWidget,
    source: go.Figure,
    last_factory_layout: MutableMapping[str, Any],
    *,
    preserve_ui_state: bool = True,
) -> None:
    ui_state = _capture_plotly_ui_state(target) if preserve_ui_state else {}
    source_layout = source.layout.to_plotly_json()
    with target.batch_update():
        for source_trace, target_trace in zip(source.data, target.data):
            _apply_trace_diff(target_trace, source_trace)
        _apply_layout_diff(target.layout, source_layout, last_factory_layout)
        try:
            target.frames = source.frames
        except (AttributeError, ValueError):
            pass
        _restore_plotly_ui_state(target, ui_state)
    last_factory_layout.clear()
    last_factory_layout.update(copy.deepcopy(source_layout))


def replace_figure_widget(
    target: go.FigureWidget,
    source: go.Figure,
    *,
    preserve_ui_state: bool = True,
    last_factory_layout: MutableMapping[str, Any] | None = None,
) -> None:
    """Replace or incrementally update a FigureWidget from a freshly built figure.

    When ``last_factory_layout`` is supplied and trace structure is unchanged,
    only changed trace properties and layout keys are patched in place.
    """

    if last_factory_layout is not None and _figures_share_structure(source, target):
        _update_figure_widget_in_place(
            target,
            source,
            last_factory_layout,
            preserve_ui_state=preserve_ui_state,
        )
        return

    _replace_figure_widget_full(target, source, preserve_ui_state=preserve_ui_state)
    if last_factory_layout is not None:
        last_factory_layout.clear()
        last_factory_layout.update(copy.deepcopy(source.layout.to_plotly_json()))


def _full_width_slider(slider: widgets.Widget) -> widgets.Widget:
    slider.layout = widgets.Layout(width="100%", min_width="0")
    return slider


# Above-figure readouts (scale on the left, privacy on the right) share one font so
# the "Scale = ..." message and the "(ε, δ)-DP" message read as a matched pair. The
# font is kept below the centered distribution title, and the compute-ε toggle button
# is sized to sit on the same line beside them.
_READOUT_STYLE = "font-size:1.05rem;font-weight:600;line-height:1.4;white-space:nowrap;"


def _readout_html(text: str, *, center: bool = False) -> str:
    span = f'<span style="{_READOUT_STYLE}">{escape(text)}</span>'
    if not center:
        return span
    return f'<div style="text-align:center;width:100%;">{span}</div>'


def _subplot_aligned_grid_row(
    *,
    left: widgets.Widget | None,
    right: widgets.Widget | None,
    figure_width: int,
    slider_grid_template: str,
) -> widgets.GridBox:
    empty = widgets.Box(layout=widgets.Layout(width="100%"))
    return widgets.GridBox(
        [
            empty,
            left or empty,
            empty,
            right or empty,
            empty,
        ],
        layout=widgets.Layout(
            width=f"{figure_width}px",
            max_width="100%",
            grid_template_columns=slider_grid_template,
            grid_gap="0px",
        ),
    )


def _column_box(child_widgets: Sequence[widgets.Widget]) -> widgets.Widget:
    if not child_widgets:
        return widgets.Box(layout=widgets.Layout(width="100%"))
    if len(child_widgets) == 1:
        return child_widgets[0]
    return widgets.VBox(
        list(child_widgets),
        layout=widgets.Layout(width="100%", min_width="0", grid_gap="6px"),
    )


def _distribution_title_html(name: str) -> str:
    lowered = name.lower().rstrip()
    if lowered.endswith("distribution") or lowered.endswith("distributions"):
        label = name
    else:
        label = f"{name} distribution"
    return (
        f'<div style="text-align:center;font-size:1.35rem;font-weight:600;'
        f'margin:6px 0 4px;width:100%;">{escape(label)}</div>'
    )


def _set_widget_shown(widget: widgets.Widget, shown: bool) -> None:
    widget.layout.display = None if shown else "none"


def roc_subplot_control_layout(
    *,
    scale_readout: Callable[[Mapping[str, widgets.Widget]], str],
    privacy_readout: Callable[[Mapping[str, widgets.Widget]], str] | None = None,
    compute_epsilon_control: str | None = None,
    below_left: Sequence[str] = ("scale",),
    below_right: Sequence[str] = ("delta",),
    below_left_footer: Sequence[str] = (),
    below_right_footer_actions: Sequence[str] = (),
    toolbar: Sequence[str] = (),
    centered_title: str = "",
    centered_title_control: str | None = None,
    figure_width: int = 1000,
    slider_grid_template: str | None = None,
) -> ControlLayout:
    """Lay out the ROC explorer controls aligned with its PDF and ROC subplot panels.

    The result stacks vertically::

        [ distribution selector ]            (optional toolbar)
        [ centered distribution title ]      (optional)
        [ "Scale = ..."  |  Compute ε  (ε, δ)-DP ]   above-figure readouts
        [               figure              ]
        [ scale slider   |  delta slider ]   below-figure primary controls
        [ samples slider |  Generate button ]   below-figure footer (empirical only)

    The delta slider and the ``(ε, δ)-DP`` readout are revealed only once epsilon
    computation is active -- either because the compute-ε toggle is pressed, or
    because the explorer was created with epsilon mode permanently on (no toggle).
    """

    def layout(
        figure: go.FigureWidget,
        controls: Mapping[str, widgets.Widget],
        actions: Mapping[str, widgets.Button],
        errors: widgets.Output,
    ) -> widgets.Widget:
        grid_template = slider_grid_template or f"{figure_width}px"

        def aligned_row(
            left: widgets.Widget | None,
            right: widgets.Widget | None,
        ) -> widgets.GridBox:
            return _subplot_aligned_grid_row(
                left=left,
                right=right,
                figure_width=figure_width,
                slider_grid_template=grid_template,
            )

        def sliders(names: Sequence[str]) -> list[widgets.Widget]:
            return [_full_width_slider(controls[name]) for name in names if name in controls]

        def epsilon_active() -> bool:
            if compute_epsilon_control is None or compute_epsilon_control not in controls:
                return True
            return bool(controls[compute_epsilon_control].value)

        # --- optional distribution toolbar + centered title -------------------
        toolbar_widgets = [controls[name] for name in toolbar if name in controls]
        toolbar_row = widgets.HBox(
            toolbar_widgets,
            layout=widgets.Layout(
                width=f"{figure_width}px",
                max_width="100%",
                display="flex",
                flex_flow="row wrap",
                align_items="center",
                grid_gap="8px",
            ),
        )
        title_widget = widgets.HTML(
            value=_distribution_title_html(centered_title),
            layout=widgets.Layout(width=f"{figure_width}px", max_width="100%"),
        )
        if centered_title_control and centered_title_control in controls:

            def sync_centered_title(_change: dict[str, Any] | None = None) -> None:
                title_widget.value = _distribution_title_html(
                    str(controls[centered_title_control].value)
                )

            controls[centered_title_control].observe(sync_centered_title, names="value")
            sync_centered_title()

        # --- above-figure readouts: scale message | compute toggle + DP message.
        # Each side is centered under its subplot panel.
        scale_label = widgets.HTML(layout=widgets.Layout(width="100%"))
        privacy_label = widgets.HTML(layout=widgets.Layout(margin="0 0 0 12px"))
        above_right_widgets: list[widgets.Widget] = []
        if compute_epsilon_control and compute_epsilon_control in controls:
            above_right_widgets.append(controls[compute_epsilon_control])
        if privacy_readout is not None:
            above_right_widgets.append(privacy_label)
        above_right = (
            widgets.HBox(
                above_right_widgets,
                layout=widgets.Layout(
                    width="100%",
                    display="flex",
                    flex_flow="row wrap",
                    align_items="center",
                    justify_content="center",
                    grid_gap="10px",
                ),
            )
            if above_right_widgets
            else None
        )
        above_row = aligned_row(left=scale_label, right=above_right)

        # --- below-figure primary controls: scale slider | delta slider -------
        primary_row = aligned_row(
            left=_column_box(sliders(below_left)),
            right=_column_box(sliders(below_right)),
        )

        # --- below-figure footer: samples slider | generate button ------------
        footer_left = sliders(below_left_footer)
        footer_right = [actions[name] for name in below_right_footer_actions if name in actions]
        for button in footer_right:
            button.layout = widgets.Layout(width="100%", min_width="0")
        footer_row = (
            aligned_row(
                left=_column_box(footer_left) if footer_left else None,
                right=_column_box(footer_right) if footer_right else None,
            )
            if footer_left or footer_right
            else None
        )

        def sync_readouts(_change: dict[str, Any] | None = None) -> None:
            active = epsilon_active()
            scale_label.value = _readout_html(scale_readout(controls), center=True)
            for name in below_right:
                if name in controls:
                    _set_widget_shown(controls[name], active)
            if privacy_readout is None:
                return
            if not active:
                privacy_label.value = ""
                _set_widget_shown(privacy_label, False)
                return
            try:
                text = privacy_readout(controls)
            except Exception as error:
                text = f"{type(error).__name__}: {error}"
            privacy_label.value = _readout_html(text)
            _set_widget_shown(privacy_label, True)

        for name in ("compute_epsilon", "scale", "delta", "distribution", "n_samples"):
            if name in controls:
                controls[name].observe(sync_readouts, names="value")
        sync_readouts()

        children: list[widgets.Widget] = []
        if toolbar_widgets:
            children.append(toolbar_row)
        if centered_title or centered_title_control:
            children.append(title_widget)
        children.extend([above_row, figure, primary_row])
        if footer_row is not None:
            children.append(footer_row)
        children.append(errors)
        return widgets.VBox(
            children,
            layout=widgets.Layout(width=f"{figure_width}px", max_width="100%"),
        )

    return layout


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
    last_factory_layout: dict[str, Any] = copy.deepcopy(figure_widget.layout.to_plotly_json())

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
            last_factory_layout=last_factory_layout,
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

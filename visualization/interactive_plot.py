"""Legacy interactive plot wrapper — prefer ``InteractiveSpec`` and ``AbstractInteractivePlot``."""

import plotly.graph_objects as go
from IPython.display import display

from .interactive import ControlSpec, InteractiveSpec
from .interactive_widgets import render_ipywidgets


class PlotMethod:
    def __init__(self, name, function):
        self.name = name
        self.function = function


class PlotSlider:
    def __init__(
        self,
        description,
        initial_value,
        min_value,
        max_value,
        step,
        continuous_update=True,
    ):
        self.description = description
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.continuous_update = continuous_update


class PlotAxisData:
    def __init__(self, title, range, type):
        self.title = title
        self.range = range
        self.type = type


class PlotMetaData:
    def __init__(self, title, x_axis, y_axis, width, height, hyper_params, title_func=None):
        self.title = title
        self.title_func = title_func
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.width = width
        self.height = height
        self.hyper_params = hyper_params


class InteractivePlot:
    """Legacy API — use ``InteractiveSpec`` with ``render_ipywidgets`` or ``AbstractInteractivePlot``."""
    def __init__(
        self,
        sliders_data: list[PlotSlider],
        methods: list[PlotMethod],
        arg_creator,
        meta_data: PlotMetaData,
    ):
        self.arg_creator = arg_creator
        self.methods = methods
        self.hyper_params = meta_data.hyper_params
        self.title_func = meta_data.title_func
        self.x_range = meta_data.x_axis.range
        self.y_range = meta_data.y_axis.range

        controls = tuple(
            ControlSpec(
                name=f"value_{index}",
                kind="slider",
                label=slider.description,
                default=slider.initial_value,
                min=slider.min_value,
                max=slider.max_value,
                step=slider.step,
                continuous=slider.continuous_update,
            )
            for index, slider in enumerate(sliders_data)
        )

        def make_figure(**values):
            positional_values = [values[f"value_{index}"] for index in range(len(controls))]
            arguments = self.arg_creator(self.hyper_params, *positional_values)
            figure = go.Figure()
            for method in self.methods:
                x, y = method.function(**arguments)
                figure.add_trace(go.Scatter(x=x, y=y, name=method.name, line={"width": 2}))
            figure.update_layout(
                title=(
                    self.title_func(**arguments) if self.title_func is not None else meta_data.title
                ),
                xaxis_title=meta_data.x_axis.title,
                xaxis_type=meta_data.x_axis.type,
                yaxis_title=meta_data.y_axis.title,
                yaxis_type=meta_data.y_axis.type,
                showlegend=True,
                width=meta_data.width,
                height=meta_data.height,
            )
            if self.x_range is not None:
                figure.update_xaxes(range=self.x_range)
            if self.y_range is not None:
                figure.update_yaxes(range=self.y_range)
            return figure

        spec = InteractiveSpec(
            name="legacy_interactive_plot",
            artifact_name="legacy-interactive-plot",
            controls=controls,
            preferred_backend="ipywidgets",
            allowed_backends=("ipywidgets",),
            make_figure=make_figure,
            figure_factory=f"{__name__}:InteractivePlot",
        )
        # Compatibility path: legacy callers construct ``InteractivePlot`` directly
        # instead of using ``AbstractInteractivePlot.show()``.
        self._rendered = render_ipywidgets(spec)
        self.fig = self._rendered.figure
        self.sliders = list(self._rendered.controls.values())

    def update_plot(self, *args):
        self._rendered.update()

    def show(self):
        display(self._rendered.root)

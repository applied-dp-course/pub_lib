import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display


class PlotMethod:
    def __init__(self, name, function):
        self.name = name
        self.function = function


class PlotSlider:
    def __init__(self, description, initial_value, min_value, max_value, step):
        self.description = description
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step


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

        self.fig = go.FigureWidget()
        for method in methods:
            self.fig.add_trace(go.Scatter(x=[], y=[], name=method.name, line=dict(width=2)))

        self.sliders = []
        for slider_data in sliders_data:
            self.sliders.append(
                widgets.FloatSlider(
                    value=slider_data.initial_value,
                    min=slider_data.min_value,
                    max=slider_data.max_value,
                    step=slider_data.step,
                    description=slider_data.description,
                    continuous_update=False,
                )
            )
            self.sliders[-1].observe(self.update_plot, 'value')

        self.fig.update_layout(
            title=meta_data.title,
            xaxis_title=meta_data.x_axis.title,
            xaxis_type=meta_data.x_axis.type,
            yaxis_title=meta_data.y_axis.title,
            yaxis_type=meta_data.y_axis.type,
            showlegend=True,
            width=meta_data.width,
            height=meta_data.height,
        )
        if self.x_range is not None:
            self.fig.update_xaxes(range=self.x_range)
        if self.y_range is not None:
            self.fig.update_yaxes(range=self.y_range)
        if self.title_func is not None:
            args = self.arg_creator(self.hyper_params, *[slider.value for slider in self.sliders])
            self.fig.update_layout(title_text=self.title_func(**args))

        self.update_plot()

    def update_plot(self, *args):
        args = self.arg_creator(self.hyper_params, *[slider.value for slider in self.sliders])
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        with self.fig.batch_update():
            for i, method in enumerate(self.methods):
                x, y = method.function(**args)
                self.fig.data[i].x = x
                self.fig.data[i].y = y
                x_min = min(x_min, np.min(x))
                x_max = max(x_max, np.max(x))
                y_min = min(y_min, np.min(y))
                y_max = max(y_max, np.max(y))
        if self.title_func is not None:
            self.fig.update_layout(title_text=self.title_func(**args))
        if self.x_range is not None:
            self.fig.update_xaxes(range=[x_min, x_max])
        if self.y_range is not None:
            self.fig.update_yaxes(range=[y_min, y_max])

    def show(self):
        # Display widgets and plot
        display(widgets.VBox([self.fig, widgets.HBox(*[self.sliders])]))

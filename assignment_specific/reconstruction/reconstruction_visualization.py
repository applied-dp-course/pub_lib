import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


def make_reconstruction_error_figure(errors, noise_range, title=""):
    fig, ax = plt.subplots()
    for err in errors:
        ax.plot(noise_range, err[0] * 100, label=err[1])
    ax.set_xlabel('Noise Range')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title(f"Reconstruction error as a function of noise range{title}")
    ax.legend()
    return fig


def plot_reconstruction_error_as_noise_function(errors, noise_range, title=""):
    fig = make_reconstruction_error_figure(errors, noise_range, title=title)
    display(fig)
    return fig

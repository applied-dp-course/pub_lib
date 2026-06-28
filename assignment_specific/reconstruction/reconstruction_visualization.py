import matplotlib.pyplot as plt
import numpy as np


def make_reconstruction_error_figure(errors, noise_range, title=""):
    fig, ax = plt.subplots()
    for err in errors:
        ax.plot(noise_range, err[0] * 100, label=err[1])
    ax.set_xlabel('Noise Range')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title(f"Reconstruction error as a function of noise range{title}")
    ax.legend()
    return fig

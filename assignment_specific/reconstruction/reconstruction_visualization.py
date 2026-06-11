from matplotlib import pyplot as plt


def plot_reconstruction_error_as_noise_function(errors, noise_range, title=""):
    for err in errors:
        plt.plot(noise_range, err[0] * 100, label=err[1])

    plt.xlabel('Noise Range')
    plt.ylabel('Reconstruction Error')
    plt.title(f"Reconstruction error as a function of noise range" + title)
    plt.legend()
    plt.show()

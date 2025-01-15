import numpy as np
from src.fourrier import decompose
import matplotlib.pyplot as plt


if __name__ == "__main__":

    time_grid = np.linspace(0, 1, 100)

    load_amplitudes = [3, 1]
    load_frequency = 1
    load_angular_frequency = 2 * np.pi * load_frequency
    sigmas = load_amplitudes[0] * np.cos(load_angular_frequency*3 * time_grid) + \
              load_amplitudes[1] * np.sin(load_angular_frequency * time_grid)
    sigmas = sigmas + 10

    A, B, omegas, y = decompose(sigmas, time_grid)

    fig, ax = plt.subplots()
    ax.plot(time_grid, sigmas)
    ax.plot(time_grid, y)
    plt.close()
    fig.savefig(r"results/fourrier_singal.png")

    
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.fft import fft, ifft


def consolidation(cv: float, amps: np.ndarray[Union[float, int], "2"], angular_frequency: Union[int, float],
                  z_grid: np.ndarray[float, "n_depths"], time_grid: np.ndarray[float, "n_times"], n_terms: int = 1_000,
                  eta: float = 1.):

    amps = amps if amps.ndim > 1 else np.expand_dims(amps, axis=0)

    H = z_grid.max()
    A, B = np.hsplit(amps, 2)

    j_grid = np.expand_dims(np.arange(1, n_terms+1), axis=(0, 1))
    time_grid = np.expand_dims(time_grid, axis=(0, -1))
    z_grid = np.expand_dims(z_grid, axis=(1, 2))

    T_v = cv * time_grid / H ** 2

    theta = eta * cv / (angular_frequency * H ** 2)

    chi = (2 * j_grid - 1) * np.pi / 2

    Y_j = (A + B * theta * chi ** 2) * (np.cos(angular_frequency * time_grid) - np.exp(-eta * chi ** 2 * T_v)) - \
          (A * theta * chi ** 2 - B) * np.sin(angular_frequency * time_grid)

    sum_arg = (-1) ** j_grid * Y_j * np.cos(chi * z_grid / H) / (chi + theta ** 2 * chi ** 5)

    u = - 2 * eta * np.sum(sum_arg, axis=-1)

    return u


def DFT(x):
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))

    e = np.exp(-2j * np.pi * k * n / N)
    X_k = np.zeros_like(k, dtype=np.complex128)
    for row in n:
        for col in n:
            X_k[row] += e[row, col] * x[col]
    return X_k


if __name__ == "__main__":

    time_grid = np.linspace(0, 1, 50)

    load_amplitudes = np.c_[np.arange(1, 4), np.arange(11, 14)]
    load_frequency = np.arange(1, 4)
    load_angular_frequency = 2 * np.pi * load_frequency
    sigmas = (load_amplitudes[:, 0] * np.cos(load_angular_frequency * time_grid[:, np.newaxis])+
              load_amplitudes[:, 1] * np.sin(load_angular_frequency * time_grid[:, np.newaxis]))
    sigmas = sigmas.sum(axis=1)

    # X_k = DFT(sigmas)
    # N = X_k.size
    # n = np.arange(N)
    # interval = time_grid[1] - time_grid[0]
    # rate = 1 / interval
    # T = N / rate
    # freq = n / T
    #
    # n_oneside = N // 2
    # freq = freq[:n_oneside]
    # X_k = X_k[:n_oneside] / n_oneside
    #
    # amps = np.abs(X_k).squeeze()
    # idx = np.argsort(amps)[::-1]
    # freq = freq[idx]
    # amps = amps[idx]
    # cum_amps = np.cumsum(amps) / amps.sum()
    # idx = cum_amps <= 0.95
    # amps = amps[idx]
    # freq = freq[idx]

    AA = amps[:, None] * np.sin(freq[:, None]*time_grid)
    AAA = AA.sum(0)


    N = sigmas.size
    interval = time_grid[1] - time_grid[0]
    n_oneside = N // 2
    amps = np.fft.fft(sigmas)
    amps = np.abs(amps)[:n_oneside]
    freq = np.fft.fftfreq(N, d=interval)
    freq = freq[:n_oneside]

    fig = plt.figure()
    plt.scatter(freq, amps)
    plt.close()
    fig.savefig(r"results/fourrier_load_height.png")


    h = 1
    cv = 1.7e-8
    z_grid = np.linspace(0, h, 1_000)

    # u = consolidation(
    #     cv=cv,
    #     amplitudes=np.asarray([A, B]),
    #     angular_frequency=load_angular_frequency,
    #     z_grid=z_grid,
    #     time_grid=time_grid,
    # )

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    axs[0].plot(time_grid, sigmas)
    axs[0].plot(time_grid, AAA)
    # axs[0].set_xlabel("Time [s]", fontsize=12)
    # axs[0].set_ylabel("Load [kPa]", fontsize=12)
    # axs[0].grid()
    # contourf_ = axs[1].contourf(time_grid, z_grid.squeeze()/z_grid.max(), u/load_amplitude)
    # cbar = fig.colorbar(contourf_)
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel("Overpressure/Amplitude [-]", rotation=270)
    # axs[1].set_xlabel("Time [s]", fontsize=12)
    # axs[1].set_ylabel("z/H [-]", fontsize=12)
    # axs[1].grid()
    plt.close()
    fig.savefig(r"results/fourrier_load_timelines.png")

    # fig = plt.figure()
    # plt.plot(np.abs(u).max(axis=1)/load_amplitude, z_grid.squeeze()/z_grid.max())
    # plt.xlabel("Overpressure/Amplitude [-]", fontsize=12)
    # plt.ylabel("z/H [-]", fontsize=12)
    # plt.grid()
    # plt.close()
    # fig.savefig(r"results/fourrier_load_height.png")


    
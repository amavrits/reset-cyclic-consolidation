import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    time_grid = np.linspace(0, 1, 1_000)

    load_amplitude = 1
    load_frequency = 1
    load_angular_frequency = 2 * np.pi * load_frequency
    sigmas = load_amplitude * np.sin(load_angular_frequency * time_grid)
    A, B = 0, load_amplitude

    h = 1
    # gamma_w = 9.81
    # k = 1e-7
    # m_v = 4.2e-4
    # c_v = k / (gamma_w * m_v)
    c_v = 1.7e-8
    eta = 1

    j_grid = np.expand_dims(np.arange(1, 101), axis=(0, 1))
    t_grid = np.expand_dims(time_grid, axis=(0, -1))
    z_grid = np.expand_dims(np.linspace(0, h, 10), axis=(1, 2))

    T_v = c_v * t_grid / h ** 2
    theta = eta * c_v / (load_angular_frequency * h ** 2)
    chi = (2 * j_grid - 1) * np.pi / 2
    Y_j = (A + B * theta * chi ** 2) * (np.cos(load_angular_frequency * t_grid) - np.exp(-eta * chi ** 2 * T_v)) - \
          (A * theta * chi ** 2 - B) * np.sin(load_angular_frequency * t_grid)

    sum_arg = (-1) ** j_grid * Y_j * np.cos(chi * z_grid / h) / (chi + theta ** 2 * chi ** 5)
    u = - 2 * eta * np.sum(sum_arg, axis=-1)


    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    axs[0].plot(time_grid, sigmas)
    axs[0].set_xlabel("Time [s]", fontsize=12)
    axs[0].set_ylabel("Load [kPa]", fontsize=12)
    contourf_ = axs[1].contourf(time_grid, z_grid.squeeze()/z_grid.max(), u/load_amplitude)
    cbar = fig.colorbar(contourf_)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Overpressure/Amplitude [-]', rotation=270)
    axs[1].set_xlabel("Time [s]", fontsize=12)
    axs[1].set_ylabel("z/H [-]", fontsize=12)
    plt.close()
    fig.savefig(r"results/sin_load.png")


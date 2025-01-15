import numpy as np
from src.consolidation import consolidation_fourrier
import matplotlib.pyplot as plt


if __name__ == "__main__":

    h = 1
    cv = 1e-0

    time_grid = np.linspace(0, 2, 500)
    tb = 2.5 * h ** 2 / cv
    ta = 0.05 * tb
    T = 1.05 * tb
    z_grid = np.linspace(0, h, 20)

    load_amplitude = 1
    load_frequency = 10
    fixed_load = 10
    load_angular_frequency = 2 * np.pi * load_frequency
    # sigmas = np.where(time_grid > =0.1, fixed_load, 0)
    sigmas = np.where(
        time_grid >= 0.1,
        fixed_load+load_amplitude * np.sin(load_angular_frequency * time_grid),
        0)

    u = consolidation_fourrier(
        sigmas,
        cv=cv,
        z_grid=z_grid,
        time_grid=time_grid,
    )

    fig = plt.figure()
    plt.plot(time_grid, sigmas, label="Load")
    for i, z in enumerate(z_grid[::2]):
        plt.plot(time_grid, u[i], label=str(z))
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Overpressure [kPa]", fontsize=12)
    plt.legend()
    plt.close()
    fig.savefig(r"results/fourrier_load_timelines.png")

    fig = plt.figure()
    for i in range(0, time_grid.size, 5):
        t = time_grid[i]
        plt.plot(u[:, i], z_grid, label=str(t))
    plt.xlabel("Overpressure [kPa]", fontsize=12)
    plt.ylabel("Depth [m]", fontsize=12)
    plt.legend()
    plt.close()
    fig.savefig(r"results/fourrier_load_height.png")


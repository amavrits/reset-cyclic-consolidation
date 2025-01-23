import numpy as np
from src.consolidation import dynamic_consolidation_fourrier, jax_dynamic_consolidation_fourrier
import matplotlib.pyplot as plt


if __name__ == "__main__":

    h = 1
    cv = 1e-0

    time_grid = np.linspace(0, 2, 200)
    z_grid = np.linspace(0, h, 20)

    fixed_load = 10
    load_amplitude = 1
    load_frequency = 5
    load_angular_frequency = 2 * np.pi * load_frequency
    load = np.where(time_grid >= 0.1, fixed_load + load_amplitude * np.sin(load_angular_frequency * time_grid), 0)
    # load = np.where(time_grid >= 0.1, fixed_load, 0)

    # u = dynamic_consolidation_fourrier(load, cv, z_grid, time_grid)
    u = np.asarray(jax_dynamic_consolidation_fourrier(load, cv, z_grid, time_grid))

    fig = plt.figure()
    plt.plot(time_grid, load, c="k", label="Load")
    for i in range(0, z_grid.size, 5):
        z = z_grid[i]
        plt.plot(time_grid, u[i], label=round(z/h, 1))
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Overpressure [kPa]", fontsize=12)
    plt.legend(title="z/H [-]")
    plt.grid()
    plt.close()
    fig.savefig(r"results/fourrier_load_timelines.png")

    fig = plt.figure()
    for i in range(0, time_grid.size, 40):
        t = time_grid[i]
        plt.plot(u[:, i], z_grid/h, label=str(round(t, 1)))
    plt.xlabel("Overpressure [kPa]", fontsize=12)
    plt.ylabel("z/H [-]", fontsize=12)
    plt.legend(title="Time [s]")
    plt.grid()
    plt.close()
    fig.savefig(r"results/fourrier_load_height.png")


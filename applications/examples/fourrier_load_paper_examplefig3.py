import numpy as np
from src.consolidation import dynamic_consolidation_fourrier
import matplotlib.pyplot as plt


if __name__ == "__main__":

    h = 1
    cv = 1e+2

    tb = 2.5 * h ** 2 / cv
    ta = 0.05 * tb
    T = 1.05 * tb

    time_grid = np.linspace(0, T, 100)
    z_grid = np.linspace(0, h, 20)

    fixed_load = 10
    load_frequency = 5
    load = np.where(np.logical_and(time_grid >= ta, time_grid <= tb), fixed_load, 0)

    u = dynamic_consolidation_fourrier(load, cv, z_grid, time_grid)

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
    fig.savefig(r"results/paper_fig3_timelines.png")

    fig = plt.figure()
    T_v = cv * (time_grid - ta) / h ** 2
    for tv in [0.1, 0.2, 0.4, 0.7, 0.9]:
        i = np.argmin(np.abs(T_v-tv))
        plt.plot(u[:, i]/fixed_load, z_grid/h, label=str(round(tv, 1)))
    plt.xlabel("u/q [-]", fontsize=12)
    plt.ylabel("z/H [-]", fontsize=12)
    plt.legend(title="${T}_{v}$ [-]")
    plt.grid()
    plt.close()
    fig.savefig(r"results/paper_fig3_height.png")


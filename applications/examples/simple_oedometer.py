import numpy as np
import jax
import jax.numpy as jnp
from src.consolidation import dynamic_consolidation, jax_dynamic_consolidation, jax_oedometer
import matplotlib.pyplot as plt
from typing import Union


if __name__ == "__main__":

    time_grid = np.linspace(0, 1, 100)
    h = 1
    cv = 1e-0
    load = 10
    z_grid = np.linspace(0, h, 10)

    u = jax_oedometer(cv, load, z_grid, time_grid)

    fig = plt.figure()
    plt.axhline(load, c="k", label="Load")
    for i, z in enumerate(z_grid[::2]):
        plt.plot(time_grid, u[i], label=str(z))
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Overpressure [kPa]", fontsize=12)
    plt.legend()
    plt.close()
    fig.savefig(r"results/oedometer_timelines.png")

    fig = plt.figure()
    for i, t in enumerate(time_grid):
        plt.plot(u[:, i], z_grid, label=str(t))
    plt.xlabel("Overpressure [kPa]", fontsize=12)
    plt.ylabel("Depth [m]", fontsize=12)
    plt.legend()
    plt.close()
    fig.savefig(r"results/oedometer_height.png")




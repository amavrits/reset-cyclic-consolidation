import numpy as np
import jax
import jax.numpy as jnp
from src.consolidation import consolidation, jax_consolidation
import matplotlib.pyplot as plt
from typing import Union


if __name__ == "__main__":

    time_grid = np.linspace(0, 1, 100)

    load_amplitude = 1
    load_frequency = 1
    load_angular_frequency = 2 * np.pi * load_frequency
    sigmas = load_amplitude * np.sin(load_angular_frequency * time_grid)
    A, B = 0, load_amplitude

    h = 1
    cv = 1.7e-2

    z_grid = np.linspace(0, h, 10)

    u = consolidation(
        cv=cv,
        amps=(A, B),
        angular_frequency=load_angular_frequency,
        z_grid=z_grid,
        time_grid=time_grid,
    )

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    axs[0].plot(time_grid, sigmas, label="Load")
    axs[0].plot(time_grid, np.abs(u).max(0) * np.sign(sigmas), label="Max. overpressure")
    axs[0].set_xlabel("Time [s]", fontsize=12)
    axs[0].set_ylabel("Load [kPa]", fontsize=12)
    axs[0].grid()
    axs[0].legend()
    contourf_ = axs[1].contourf(time_grid, z_grid.squeeze()/z_grid.max(), u/load_amplitude)
    cbar = fig.colorbar(contourf_)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Overpressure/Amplitude [-]", rotation=270)
    axs[1].set_xlabel("Time [s]", fontsize=12)
    axs[1].set_ylabel("z/H [-]", fontsize=12)
    axs[1].grid()
    plt.close()
    fig.savefig(r"results/sin_load_timelines.png")

    fig = plt.figure()
    plt.plot(np.abs(u).max(axis=1)/load_amplitude, z_grid.squeeze()/z_grid.max())
    plt.xlabel("Overpressure/Amplitude [-]", fontsize=12)
    plt.ylabel("z/H [-]", fontsize=12)
    plt.grid()
    plt.close()
    fig.savefig(r"results/sin_load_height.png")


    cvs = 10 ** jnp.linspace(-9, -4, 20)
    frequencies = 10 ** jnp.linspace(-3, -1, 21)
    angular_frequencies = 2 * jnp.pi * frequencies
    cv_mesh, frequency_mesh = jnp.meshgrid(cvs, angular_frequencies)
    cv_mesh, frequency_mesh = cv_mesh.flatten(), frequency_mesh.flatten()
    angular_frequency_mesh = 2 * jnp.pi * frequency_mesh

    f = lambda x, y: jax_consolidation(
        cv=x,
        amps=jnp.asarray([A, B]),
        angular_frequency=y,
        z_grid=jnp.asarray(z_grid),
        time_grid=jnp.asarray(time_grid)
    )
    us = jax.vmap(f)(cv_mesh, angular_frequency_mesh)
    us = us.reshape(angular_frequencies.size, cvs.size, z_grid.size, time_grid.size)
    u_mid_max = us[..., 5, :].max(axis=(-1))
    u_mid_max = np.asarray(u_mid_max)

    fig = plt.figure()
    cv_mesh, frequency_mesh = jnp.meshgrid(cvs, frequencies)
    contourf_ = plt.contourf(cv_mesh, frequency_mesh, u_mid_max / load_amplitude)
    cbar = fig.colorbar(contourf_)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Max. overpressure / Amplitude [-]", rotation=270, fontsize=12)
    plt.xlabel("${c}_{v}$ [${m}^{2}/s$]", fontsize=12)
    plt.ylabel("Frequency [Hz]", fontsize=12)
    plt.grid()
    plt.close()
    fig.savefig(r"results/sin_load_sensitivity.png")


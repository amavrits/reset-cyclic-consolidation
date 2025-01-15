import numpy as np
from src.fourrier import decompose
import jax
import jax.numpy as jnp
from typing import Union
from jaxtyping import Array, Float, Int


def consolidation(cv: float, amps: Union[tuple, list, np.ndarray[Union[float, int], "2"]],
                  angular_frequency: Union[int, float], z_grid: np.ndarray[float, "n_depths"],
                  time_grid: np.ndarray[float, "n_times"], n_terms: int = 1_000, eta: float = 1.):

    angular_frequency = angular_frequency if np.abs(angular_frequency) > 1e-6 else 1e-6  # Failsafe for angular_frequency=0

    H = z_grid.max()
    A, B = amps

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


def consolidation_fourrier(load: np.ndarray[float, "n_times"], cv: float, z_grid: np.ndarray[float, "n_depths"],
                  time_grid: np.ndarray[float, "n_times"], n_terms: int = 1_000, eta: float = 1.):

    A, B, angular_freqs, _ = decompose(load, time_grid)

    n_comps = load.size
    n_depths = z_grid.size
    n_times = time_grid.size
    u_comps = np.zeros((n_comps, n_depths, n_times))

    for i, (a, b, omega) in enumerate(zip(A, B, angular_freqs)):
        u_comps[i] = consolidation(cv, (a, b), omega, z_grid, time_grid, n_terms, eta)

    u = u_comps.sum(axis=0)


    return u


@jax.jit
def jax_consolidation(cv: Float[Array, "n_cvs"], amps: Float[Array, "2"], angular_frequency: Float[Array, "n_freqs"],
                      z_grid: Float[Array, "n_depths"], time_grid: Float[Array, "n_times"],
                      n_terms: int = 1_000, eta: float = 1.):

    H = z_grid.max()
    A, B = jnp.take(amps, 0, axis=-1), jnp.take(amps, 1, axis=-1)

    j_grid = jnp.expand_dims(jnp.arange(1, n_terms+1), axis=(0, 1))
    time_grid = jnp.expand_dims(time_grid, axis=(0, -1))
    z_grid = jnp.expand_dims(z_grid, axis=(1, 2))

    T_v = cv * time_grid / H ** 2

    theta = eta * cv / (angular_frequency * H ** 2)

    chi = (2 * j_grid - 1) * jnp.pi / 2

    Y_j = (A + B * theta * chi ** 2) * (jnp.cos(angular_frequency * time_grid) - jnp.exp(-eta * chi ** 2 * T_v)) - \
          (A * theta * chi ** 2 - B) * jnp.sin(angular_frequency * time_grid)

    sum_arg = (-1) ** j_grid * Y_j * jnp.cos(chi * z_grid / H) / (chi + theta ** 2 * chi ** 5)

    u = - 2 * eta * jnp.sum(sum_arg, axis=-1)

    return u


if __name__ == "__main__":

    pass


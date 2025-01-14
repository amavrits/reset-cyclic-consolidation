import numpy as np
import jax
import jax.numpy as jnp
from typing import Union
from jaxtyping import Array, Float, Int


def consolidation(cv: float, amps: Union[tuple, list, np.ndarray[Union[float, int], "2"]],
                  angular_frequency: Union[int, float], z_grid: np.ndarray[float, "n_depths"],
                  time_grid: np.ndarray[float, "n_times"], n_terms: int = 1_000, eta: float = 1.):

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


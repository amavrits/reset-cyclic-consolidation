import numpy as np
from src.fourrier import decompose, jax_decompose
import jax
import jax.numpy as jnp
from typing import Union
from jaxtyping import Array, Float
from functools import partial


def dynamic_consolidation(cv: float, amps: Union[tuple, list, np.ndarray[Union[float, int], "2"]],
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


def dynamic_consolidation_fourrier(load: np.ndarray[float, "n_times"], cv: float, z_grid: np.ndarray[float, "n_depths"],
                                   time_grid: np.ndarray[float, "n_times"], n_terms: int = 1_000, eta: float = 1.,
                                   keep_positive: bool = False):

    A, B, angular_freqs, _ = decompose(load, time_grid, keep_positive)

    n_comps = load.size
    n_depths = z_grid.size
    n_times = time_grid.size
    u_comps = np.zeros((n_comps, n_depths, n_times))

    for i, (a, b, omega) in enumerate(zip(A, B, angular_freqs)):
        u_comps[i] = dynamic_consolidation(cv, (a, b), omega, z_grid, time_grid, n_terms, eta)

    u = u_comps.sum(axis=0)

    return u


@partial(jax.jit, static_argnums=(5,))
def jax_dynamic_consolidation(cv: Float[Array, "n_cvs"], amps: Float[Array, "2"],
                              angular_frequency: Float[Array, "n_freqs"], z_grid: Float[Array, "n_depths"],
                              time_grid: Float[Array, "n_times"], n_terms: int = 1_000, eta: float = 1.):

    H = z_grid.max()
    A, B = jnp.take(amps, 0, axis=-1), jnp.take(amps, 1, axis=-1)

    angular_frequency = jnp.where(
        jnp.abs(angular_frequency) > 1e-6,
        angular_frequency,
        jnp.ones_like(angular_frequency) * 1e-6
    )  # Failsafe for angular_frequency=0

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


@jax.jit
def jax_dynamic_consolidation_fourrier(load: np.ndarray[float, "n_times"], cv: float,
                                       z_grid: np.ndarray[float, "n_depths"],
                                       time_grid: np.ndarray[float, "n_times"], n_terms: int = 1_000, eta: float = 1.):

    A, B, angular_freqs, _ = jax_decompose(load, time_grid)

    def component_loop(a, b, omega):
        u = jax_dynamic_consolidation(cv, jnp.asarray([a, b]), omega, z_grid, time_grid, n_terms, eta)
        return u

    u_comps = jax.vmap(component_loop)(A, B, angular_freqs)

    return u_comps.sum(axis=0)


# @jax.jit
def jax_oedometer(cv, q, z_grid, time_grid, n_terms = 1_000):


    j_grid = jnp.expand_dims(jnp.arange(n_terms), axis=(0, 1))
    time_grid = jnp.expand_dims(time_grid, axis=(0, -1))
    q = jnp.expand_dims(q, axis=(0, -1))
    z_grid = jnp.expand_dims(z_grid, axis=(1, 2))

    H = z_grid.max()
    T_v = cv * time_grid / H ** 2
    M_j = jnp.pi / 2 * (2 * j_grid + 1)

    sum_args = 2 * q / M_j * jnp.sin(M_j * z_grid / H) * jnp.exp(-M_j ** 2 * T_v)

    u = sum_args.sum(axis=-1)

    u = jnp.flipud(u)  # Because the analytical solution puts z=0 at bottom.

    return u


if __name__ == "__main__":

    pass


import numpy as np
import jax.numpy as jnp


def decompose(signal: np.ndarray[float, "n_times"], times: np.ndarray[float, "n_times"], keep_positive: bool = False):

    N = signal.size
    dt = times[1] - times[0]
    magnitude = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, dt)

    #TODO: Need to change the equation for reconstructing the signal when only keeping the positive frequencies.
    if keep_positive:
        freq = freq[:N//2]
        magnitude = np.abs(magnitude[:N//2]) # Scaling to account for removed negative side
        magnitude[1:-1] *= 2

    A = np.real(magnitude) / N
    B = - np.imag(magnitude) / N
    angular_freqs = 2 * np.pi * freq

    y = np.sum(A[:, None] * np.cos(angular_freqs[:, None] * times) + \
               B[:, None] * np.sin(angular_freqs[:, None] * times), axis=0)

    return A, B, angular_freqs, y


def jax_decompose(signal: np.ndarray[float, "n_times"], times: np.ndarray[float, "n_times"]):

    N = signal.size
    dt = times[1] - times[0]
    magnitude = jnp.fft.fft(signal)
    freq = jnp.fft.fftfreq(N, dt)

    A = jnp.real(magnitude) / N
    B = - jnp.imag(magnitude) / N
    angular_freqs = 2 * jnp.pi * freq

    y = np.sum(A[:, None] * jnp.cos(angular_freqs[:, None] * times) + \
               B[:, None] * jnp.sin(angular_freqs[:, None] * times), axis=0)

    return A, B, angular_freqs, y


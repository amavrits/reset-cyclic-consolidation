import numpy as np


def decompose(sigmas: np.ndarray[float, "n_times"], times: np.ndarray[float, "n_times"]):

    N = sigmas.size
    dt = times[1] - times[0]
    yf = np.fft.fft(sigmas)
    freq = np.fft.fftfreq(N, dt)

    A = np.real(yf) / N
    B = - np.imag(yf) / N
    omegas = 2 * np.pi * freq

    y = np.sum(A[:, None] * np.cos(omegas[:, None] * times) + \
               B[:, None] * np.sin(omegas[:, None] * times), axis=0)

    return A, B, omegas, y


import numpy as np


def decompose(load: np.ndarray[float, "n_times"], times: np.ndarray[float, "n_times"]):

    N = load.size
    dt = times[1] - times[0]
    yf = np.fft.fft(load)
    freq = np.fft.fftfreq(N, dt)

    A = np.real(yf) / N
    B = - np.imag(yf) / N
    angular_freqs = 2 * np.pi * freq

    y = np.sum(A[:, None] * np.cos(angular_freqs[:, None] * times) + \
               B[:, None] * np.sin(angular_freqs[:, None] * times), axis=0)

    return A, B, angular_freqs, y


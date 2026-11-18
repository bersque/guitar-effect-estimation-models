import numpy as np
from scipy.signal import lfilter

def wiener_hammerstein(x, h1, h2, k):
    # Filtr wejściowy
    u = lfilter(h1, [1.0], x)

    # Nieliniowość
    v = np.tanh(k * u)

    # Filtr wyjściowy
    y = lfilter(h2, [1.0], v)

    return y
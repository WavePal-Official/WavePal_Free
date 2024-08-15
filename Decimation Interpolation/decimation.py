import numpy as np
from scipy.io import wavfile


def create_LPF_decimation(M):
    n = np.linspace(-200 * M, 200 * M, 400 * M + 1)
    h_n = np.sinc(n / M) / M
    return h_n


def decimate(signal, M):
    h = create_LPF_decimation(M)
    y = np.convolve(signal, h, mode='same')
    return y[::M]


def decimate_wav(path, M, output_path):
    fs, signal = wavfile.read(path)
    y = decimate(signal, M)
    wavfile.write(output_path, fs // M, y.astype(signal.dtype))
    return y



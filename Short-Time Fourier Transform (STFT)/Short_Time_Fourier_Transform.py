import math
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def transpose_matrix(matrix):
    return np.transpose(matrix)

def nuttall_window(size):
    """
    Creates a nuttall window in the time domain

    Parameters:
    size (int): Size of the wanted window (length)

    Returns:
    window (array): Discrete nuttall window of the wanted size
    """
    # Constants used to create the window
    a0 = 0.355768
    a1 = 0.487396
    a2 = 0.144232
    a3 = 0.012604

    # Nuttall window is defined as nuttall_window[n]=a0-a1*cos(2pi*n/N)+a2*cos(4pi*n/N)-a*cos(6pi*n/N)
    window = \
        [
            a0
            - a1 * math.cos((2 * math.pi * n) / (size - 1))
            + a2 * math.cos((4 * math.pi * n) / (size - 1))
            - a3 * math.cos((6 * math.pi * n) / (size - 1))
            for n in range(size)
        ]
    return window


def fft(x):
    return np.fft.fft(x)

def ifft(X):
    return np.fft.ifft(X)



def rfft(x):
    return np.fft.rfft(x)

def irfft(X):
    return np.fft.irfft(X)


def stft(audio_data, window_size, hop_size, fs):
    """
    Perform Short-Time Fourier Transform (STFT) on a signal.

    Parameters:
    audio_data (array): Input signal to be transformed.
    window_size (int): Size of the window to segment the signal.
    hop_size (int): Step size to move the window (overlap size).

    Returns:
    stft_matrix (array): STFT matrix (complex values).
    time_vector (array): Time axis values.
    freq_vector (array): Frequency axis values.
    """
    window = nuttall_window(window_size)  # Creating the wanted window in the time domain

    # Preparing output matrix
    stft_matrix = []
    # Calculating number of windows/transformations that will occur
    num_windows = (len(audio_data) - window_size) // hop_size + 1

    # Calculating the DFT of each segment
    for start in range(0, len(audio_data) - window_size, hop_size):
        segment = audio_data[start:start + window_size]  # Taking a part of the signal
        windowed_segment = [segment[i] * window[i] for i in range(window_size)]  # Taking the signal*window values
        spectrum = rfft(windowed_segment)  # Taking the DFT of the signal in the specified segment
        stft_matrix.append(spectrum)

    # Frequency and time vectors - For plotting purposes
    # We used built-in functions for this because it isn't a part of the general analysis of the signal
    freq_vector = np.fft.rfftfreq(window_size, d=1 / fs)
    time_vector = np.arange(num_windows) * hop_size / fs

    # Get the transpose of the matrix so now, each row of the matrix is the DFT of one framed-signal
    stft_matrix = transpose_matrix(stft_matrix)
    return np.array(stft_matrix).T, freq_vector, time_vector


def istft(stft_matrix, window_size, hop_size):
    """
    Compute the Inverse Short-Time Fourier Transform (ISTFT) of a signal.

    Parameters:
    stft_matrix (numpy.ndarray): The STFT of the signal.
    window_size (int): The size of the window used in the STFT.
    hop_size (int): The number of samples between successive frames.

    Returns:
    numpy.ndarray: The reconstructed time-domain signal.
    """
    # Generate the window function
    window = nuttall_window(window_size)

    # Calculating the signal length
    signal_length = (len(stft_matrix) - 1) * hop_size + window_size

    # Preparing output signal
    signal = np.zeros(signal_length)
    window_sum = np.zeros(signal_length)

    for i, frame in enumerate(stft_matrix):
        frame_signal = np.array(irfft(frame))  # frame_signal = ifft(fft(signal*window))
        start = i * hop_size
        # To avoid wrong calculations caused by the overlap of different windows we will sum the entire appearances of each time point and then divide by the entire appearance of the window in that time point
        signal[start:start + window_size] += frame_signal
        window_sum[start:start + window_size] += window

    # Avoid division by zero
    window_sum[window_sum == 0] = 1
    return signal / window_sum


def plot_stft(stft_matrix, time_vector, freq_vector):
    magnitude = np.abs(transpose_matrix(stft_matrix))
    magnitude_db = 20 * np.log10(magnitude + 1e-6)  # Adding a small constant to avoid log(0)
    plt.pcolormesh(time_vector, freq_vector, magnitude_db, vmin=0, shading='gouraud')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Short-Time Fourier Transform')
    plt.show()

def plot_signal(signal, fs):
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.title('Waveform')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


# Read the WAV file
def read_wav(file_path):
    try:
        fs, audio = wavfile.read(file_path)
        return fs, audio
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        raise


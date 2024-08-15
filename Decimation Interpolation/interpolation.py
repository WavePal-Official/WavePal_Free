import matplotlib.pyplot as plt

import numpy as np
from scipy.io import wavfile


def up_sample(signal, L):
    y = np.zeros(L * len(signal))
    y[::L] = signal  # Direct assignment for upsampling
    return y


def create_LPF_interpolation(L, type='shanon', cutoff=200):
    # Filter length must be odd to center the filter
    filter_length = L * 10 + 1

    if type == 'shanon':
        n = np.linspace(-filter_length / 2, filter_length / 2, filter_length)
        h_n = np.sinc(n / float(L))  # Ensure floating-point division
        return h_n
    elif type == 'ZOH':
        h_n = np.ones(L)
        return h_n
    elif type == 'FOH':
        r = np.ones(L)
        h_n = np.convolve(r, r, mode='full')
        return 1 / float(L) * h_n  # Ensure floating-point division
    else:
        raise ValueError("Unsupported filter type")


def interpolate(x, L, type='shanon'):
    x_L = up_sample(x, L)
    h = create_LPF_interpolation(L, type)
    y = np.convolve(x_L, h, mode='same')
    return y


def interpolate_wav(path, L, type='shanon', output_path=None):
    fs, signal = wavfile.read(path)
    signal = signal.astype(float)  # Convert to float for processing
    y = interpolate(signal, L, type)
    y = np.clip(y, -32768, 32767)  # Clip values to avoid overflow
    y = y.astype(np.int16)  # Convert back to int16 for WAV format

    if output_path:
        wavfile.write(output_path, fs * L, y)
    return y


def main():
    # Generate a synthetic signal
    fs = 100  # Original sample rate
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second of data
    signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

    # Save synthetic signal to WAV file
    wavfile.write('synthetic_signal.wav', fs, (signal * 32767).astype(np.int16))  # Scale to int16

    # Interpolation factor
    L = 4

    # Perform interpolation on synthetic signal
    y = interpolate(signal, L, type='shanon')

    # Plot the original and interpolated signals
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, 'b-', label='Original Signal')  # Using plot for smoother visualization
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    # Interpolated signal time axis
    t_interpolated = np.linspace(0, 1, fs * L, endpoint=False)
    plt.subplot(2, 1, 2)
    plt.plot(t_interpolated, y, 'r-', label='Interpolated Signal')
    plt.title('Interpolated Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Test with WAV file
    wav_file_path = 'synthetic_signal.wav'
    output_wav_file_path = 'interpolated_signal.wav'
    y_wav = interpolate_wav(wav_file_path, L, type='shanon', output_path=output_wav_file_path)

    # Read and plot the interpolated WAV file
    fs_interpolated, x_interpolated_wav = wavfile.read(output_wav_file_path)
    t_interpolated_wav = np.linspace(0, len(x_interpolated_wav) / fs_interpolated, len(x_interpolated_wav),
                                     endpoint=False)

    plt.figure(figsize=(12, 6))
    plt.plot(t_interpolated_wav, x_interpolated_wav, 'g-', label='Interpolated WAV Signal')
    plt.title('Interpolated WAV Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

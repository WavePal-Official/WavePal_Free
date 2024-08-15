import sounddevice as sd  # For recording in real-time
import numpy as np  # Library for numerical computing in Python
from scipy.io import wavfile  # For reading and writing WAV files
from scipy.fft import fft, ifft  # Functions for FFT and inverse FFT
import matplotlib.pyplot as plt  # For plotting the signal


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def DC_Removal(file_path):
    sample_rate, audio_data = wavfile.read(file_path)
    N = len(audio_data)
    N = next_power_of_2(N)
    fourier_Transform = fft(audio_data, n=N)

    # Zeroing out the first 100 frequency components
    fourier_Transform[:100] = 0

    # Inverse FFT to get back to the time domain
    time_Transform = ifft(fourier_Transform)

    # Convert the result back to the original type
    return np.real(time_Transform).astype(audio_data.dtype)


def record_voice(duration, sample_rate):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    return audio_data.flatten(), sample_rate


def plot_signal(time, data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.show()


def dc_removal_real_time(signal, buffer_size, sample_rate):
    """
    Perform DC removal on a buffer of audio data using FFT.
    :param signal: The input audio buffer.
    :param buffer_size: The size of the buffer.
    :param sample_rate: The sample rate of the audio data.
    :return: The DC-removed audio buffer.
    """
    N = next_power_of_2(buffer_size)
    fourier_transform = fft(signal, n=N)

    # Zeroing out the first few frequency components to remove DC
    fourier_transform[:100] = 0

   # Inverse FFT to get back to the time domain
    time_Transform = ifft(fourier_transform)

    # Convert the result back to the original type
    return np.real(time_Transform[:buffer_size]).astype(signal.dtype)


def main(wav_filename):
    '''
    choice = input("Choose 1 for recording, 2 for file-path: ")
    if choice == '1':
        duration = float(input("How many seconds do you want to record? "))
        sample_rate = int(input("Enter sample rate: (recommended is 44100) "))
        signal, fs = record_voice(duration, sample_rate)
    else:
        # Get the file path from the user
        file_path = input("Enter the path to the WAV file: ")
        fs, signal = wavfile.read(file_path)
    '''
    fs, signal = wavfile.read(wav_filename)

    # Create the time axis for the original signal
    time = np.linspace(0, len(signal) / fs, num=len(signal))

    # Plot the original signal
    plot_signal(time, signal, 'Original WAV signal')

    # Apply DC removal
    new_signal = DC_Removal(wav_filename)
    wav_output_filename = wav_filename[:-4] + "_dc_removed.wav"
    # Plot the signal after DC removal
    time = np.linspace(0, len(new_signal) / fs, num=len(new_signal))

    plot_signal(time, new_signal, 'WAV signal after DC Removal')
    wavfile.write(wav_output_filename, fs, new_signal.astype(np.int16))
    return wav_output_filename, new_signal


if __name__ == "__main__":
    main()

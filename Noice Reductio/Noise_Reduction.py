import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from Voice_Activity_Detection import voice_activity_detection
from Short_Time_Fourier_Transform import stft, istft

# Function to read a WAV file from the given path
def read_wav(file_path):
    fs, audio = wavfile.read(file_path)
    return fs, audio

# Function to save the signal to a WAV file
def write_wav(file_path, signal, fs):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = np.clip(signal, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    wavfile.write(file_path, fs, signal.astype(np.int16))

# Function to detect noise in the audio file using VAD
def NoiseActivityDetection(file_path):
    voice, fs = voice_activity_detection(file_path)
    noise_mask = 1 - voice  # Invert the mask to get noise detection
    return noise_mask, fs

# Main noise reduction function
def noise_reduction(file_path, audio_fft, audio_freq):
    fs, audio = read_wav(file_path)
    noise, _ = NoiseActivityDetection(file_path)

    noise_fft, noise_freq, _ = stft(noise, 1024, 512, fs)
    noise_magnitude = np.abs(noise_fft)
    threshold_mask = np.max(noise_magnitude, axis=1) > 0

    common_freqs = np.intersect1d(audio_freq, noise_freq)
    audio_freq_mask = np.isin(audio_freq, common_freqs)
    noise_freq_mask = np.isin(noise_freq, common_freqs)

    audio_freq_indices = np.where(audio_freq_mask)[0]
    noise_freq_indices = np.where(noise_freq_mask)[0]

    audio_fft_filtered = np.copy(audio_fft)

    for idx in range(len(audio_freq_indices)):
        if idx < len(threshold_mask) and threshold_mask[idx]:
            audio_fft_filtered[audio_freq_indices[idx], :] = 0

    processed_signal = istft(audio_fft_filtered, 1024, 512)
    processed_signal = np.nan_to_num(processed_signal, nan=0.0, posinf=0.0, neginf=0.0)

    return processed_signal

if __name__ == "_main_":
    file_path = '/Users/niraltahan/Downloads/activity_unproductive.wav'
    fs = 44100

    # Read the audio file and perform STFT
    original_fs, original_audio = read_wav(file_path)
    audio_stft, audio_freq, _ = stft(original_audio, 1024, 512, fs)

    # Perform noise reduction
    reduced_signal = noise_reduction(file_path, audio_stft, audio_freq)

    # Plotting
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 1, 1)
    plt.title('Original Audio')
    plt.plot(np.arange(len(original_audio)) / original_fs, original_audio)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.title('Processed Audio')
    plt.plot(np.arange(len(reduced_signal)) / fs, reduced_signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # Save the noise-reduced signal to a new file
    reduced_file_path = file_path[:-4] + "_reduced.wav"
    write_wav(reduced_file_path, reduced_signal, fs)
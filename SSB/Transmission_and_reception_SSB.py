import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import sounddevice as sd

# Parameters
fs = 11025 # Sample rate
fc = 1000  # Carrier frequency in Hz
chunk_size = 2048 # Size of each chunk for real-time processing
delay = 0  # Delay in samples chunk size - to make the module casual (hilbert isn't)

# Initialize empty arrays for accumulating signals
original_signal = np.array([])
modulated_signal = np.array([])
demodulated_signal = np.array([])

#not relevant right now
# Buffer for delaying the original signal
delay_buffer = np.zeros(delay)

#This function converts into SSB according to formula that we saw in wikipedia
#The function uses Hilbert Transformationy

def convertIntoSSB(signal_in, t, isPlus, fzero=1000):
    signal_hil = hilbert(signal_in)
    ImSignal_hil = np.imag(signal_hil)
    if isPlus:
        signal_out = signal_in * np.cos(2 * np.pi * fzero * t) + ImSignal_hil * np.sin(2 * np.pi * fzero * t)
    else:
        signal_out = signal_in * np.cos(2 * np.pi * fzero * t) - ImSignal_hil * np.sin(2 * np.pi * fzero * t)
    return signal_out

"""
def ssb_demodulate(signal, carrier_freq, sample_rate):
    t = np.arange(len(signal)) / sample_rate
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)

    mixed_i = signal * carrier_i
    mixed_q = signal * carrier_q

    demodulated = mixed_i + np.imag(hilbert(mixed_q))
    return demodulated
"""

def audio_callback(indata, outdata, frames, time, status):
    global chunk_size, fc, fs, original_signal, modulated_signal, demodulated_signal, delay_buffer
    if status:
        print(status)


    # Flatten the input buffer
    input_signal = indata[:, 0]

    #not relevant right now
    # Delay the input signal
    delayed_input = np.concatenate((delay_buffer, input_signal))
    delay_buffer = delayed_input[-len(delay_buffer):]
    delayed_input = delayed_input[:frames]

    # Time array for this chunk
    t = np.arange(frames) / fs
    # Modulate the chunk
    ssb_signal = convertIntoSSB(delayed_input, t, 0, fc)
    # Demodulate the chunk
    demodulated_signal_chunk = convertIntoSSB(ssb_signal, t, 1, fc)

    # Accumulate signals
    original_signal = np.concatenate((original_signal,input_signal))
    modulated_signal = np.concatenate((modulated_signal, ssb_signal))
    demodulated_signal = np.concatenate((demodulated_signal, demodulated_signal_chunk))

    # Output the demodulated signal
    outdata[:, 0] = demodulated_signal_chunk[:chunk_size]  # Ensure correct shape and size


def real_time_processing():
    # Initialize input and output streams
    with sd.Stream(callback=audio_callback, samplerate=fs, blocksize=chunk_size, dtype='float32', channels=1):
        print("Recording audio. Press Ctrl+C to stop.")
        print("Here")
        sd.sleep(int(5 * 1000))  # Record for 5 seconds


def play_signal(signal, fs):
    sd.play(signal, fs)
    sd.wait()

def process_audio(input_signal, fs, carrier_freq):
    """
    Process a signal (not a recording)
    :param input_signal:
    :param fs:
    :param carrier_freq:
    :return:
    """
    t = np.arange(len(input_signal)) / fs
    ssb_signal = convertIntoSSB(input_signal, t, 0, carrier_freq)
    demodulated_signal = convertIntoSSB(ssb_signal, t, 1, carrier_freq)
    return ssb_signal, demodulated_signal

'''
# Parameters
fs = 44100  # Sample rate (higher for better audio quality)
fc = 1000  # Carrier frequency in Hz
duration = 30  # Duration of the recording in seconds

# Record the voice message
voice_message = wavfile.read('Counting.wav')

# Process the voice message
ssb_signal, demodulated_signal = process_audio(voice_message, fs, fc)

# Play the original, modulated, and demodulated signals
print("Playing original message...")
play_signal(voice_message, fs)

print("Playing SSB modulated signal...")
play_signal(ssb_signal, fs)

print("Playing demodulated message...")
play_signal(demodulated_signal, fs)

# Plot the results
t = np.arange(len(voice_message)) / fs
plt.figure(figsize=(12, 8))

# Original voice message signal
plt.subplot(3, 1, 1)
plt.plot(t, voice_message)
plt.title('Original Voice Message Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# SSB modulated signal
plt.subplot(3, 1, 2)
plt.plot(t, ssb_signal)
plt.title('SSB Modulated Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Demodulated signal
plt.subplot(3, 1, 3)
plt.plot(t, demodulated_signal)
plt.title('Demodulated Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()










# Time array for plotting
total_samples = len(original_signal)
t = np.arange(total_samples) / fs

# Compute Fourier transforms
freqs = np.fft.fftfreq(total_samples, 1/fs)
original_signal_fft = np.fft.fft(original_signal)
modulated_signal_fft = np.fft.fft(modulated_signal[:total_samples])
demodulated_signal_fft = np.fft.fft(demodulated_signal[:total_samples])

print("Playing original message...")
play_signal(original_signal, fs)

print("Playing SSB modulated signal...")
play_signal(modulated_signal, fs)

print("Playing demodulated message...")
play_signal(demodulated_signal, fs)

# Plot time-domain and frequency-domain signals
plt.figure(figsize=(12, 10))

# Time-domain plots
plt.subplot(3, 2, 1)
plt.plot(t, original_signal)
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 2)
plt.plot(freqs, np.abs(original_signal_fft))
plt.title('Original Signal (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 3)
plt.plot(t, modulated_signal[:total_samples])
plt.title('Modulated Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 4)
plt.plot(freqs, np.abs(modulated_signal_fft))
plt.title('Modulated Signal (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 5)
plt.plot(t, demodulated_signal[:total_samples])
plt.title('Demodulated Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 6)
plt.plot(freqs, np.abs(demodulated_signal_fft))
plt.title('Demodulated Signal (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

'''
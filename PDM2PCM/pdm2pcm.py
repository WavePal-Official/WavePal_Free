"""
Pulse Density Modulation (PDM) to Pulse Code Modulation (PCM) and WAV file (PDM2PCM)
====================================================================================
Converts Pulse Density Modulation (PDM) signal from a text file into a
Pulse Code Modulation (PCM) signal, normalizing it, and then writing it to a WAV file.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile


def read_pdm_file(filename):
    """Reads the PDM signal from a text file."""
    # Open the file - filename in reading mode

    with open(filename, 'r') as file:
        # Reads the content of the file, strips any whitespace in the begging/ending,
        # and splits the content into individual string elements
        pdm_data = file.read().strip().split()
        # The string elements we received, we convert to an integer numpy array of type 'int8'
        pdm_signal = np.array(pdm_data, dtype=np.int8)
        # Return the PDM signal as a numpy array
    return pdm_signal


def pdm_to_pcm(pdm_signal, decimation_factor):
    """Convert PDM signal to PCM signal by decimation."""
    b, a = signal.butter(8, 0.5 / decimation_factor, btype='low')
    filtered_signal = signal.lfilter(b, a, pdm_signal)
    pcm_signal = signal.decimate(filtered_signal, decimation_factor)
    return pcm_signal


def normalize_pcm_signal(pcm_signal):
    """Normalize PCM signal to the range of int16."""
    pcm_signal = pcm_signal - np.mean(pcm_signal)
    pcm_signal = pcm_signal / np.max(np.abs(pcm_signal))
    pcm_signal = np.int16(pcm_signal * 32767)
    return pcm_signal


def write_wav_file(filename, pcm_signal, sample_rate):
    """Write PCM signal to a WAV file."""
    wavfile.write(filename, sample_rate, pcm_signal)


def pdm_to_wav(pdm_filename, wav_filename, decimation_factor, sample_rate):
    """Main function to convert PDM to WAV."""
    pdm_signal = read_pdm_file(pdm_filename)
    pcm_signal = pdm_to_pcm(pdm_signal, decimation_factor)
    pcm_signal = normalize_pcm_signal(pcm_signal)
    write_wav_file(wav_filename, pcm_signal, sample_rate // decimation_factor)

def is_pdm(data):
    """
    Determine if the input data is PDM format.
    A PDM signal should have values of -1 and 1.
    """
    unique_values = np.unique(data)
    return np.array_equal(unique_values, [-1, 1])

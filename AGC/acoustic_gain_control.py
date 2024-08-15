"""
Automatic Gain Control (AGC) for Audio Processing
=================================================
This module implements an Automatic Gain Control (AGC) algorithm for
real-time audio processing. It provides applying AGC to a recorded
signal to maintain a consistent volume level, plotting the input
and output signals, and saving the processed audio into a WAV file.
"""

import numpy as np
from math import log, exp
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

FRAMES_PER_BUFFER = 1024

def get_window_indexes(end_index, sequence_size, big_length):
    """
    Calculates start & end indexes of the sequence with specified
    size and wanted end index in the whole list
    :param end_index: wanted non-negative end index of the sequence
    :type end_index: int
    :param sequence_size: the length of the sequence
    :type sequence_size: int
    :param big_length: the length of the whole list
    :type big_length: int
    :return: the start & end indexes of the sequence
    :rtype: tuple
    """
    start_index = max(end_index - sequence_size, 0)
    end_index = min(end_index, big_length - 1)
    return start_index, end_index

def plot_signals(input_arr, output_arr):
    """
    Plots the input & output signals
    :param input_arr: the input data array
    :type input_arr: np.ndarray
    :param output_arr: the output data array
    :type output_arr: np.ndarray
    :return: None
    """
    fig, (subplot1, subplot2) = plt.subplots(2, 1)
    subplot1.plot(input_arr, label="Input Signal", color='b')
    subplot1.set_title('Before AGC')
    subplot1.set_xlabel('Time')
    subplot1.set_ylabel('Value')

    subplot2.plot(output_arr, label="Output Signal", color='g')
    subplot2.set_title('After AGC')
    subplot2.set_xlabel('Time')
    subplot2.set_ylabel('Value')

    plt.tight_layout()
    plt.show()

def agc_process(data_chunk, output_arr, chunk_idx, ref_level,
                step_size, averaged_amount, gain):
    """
    Applies AGC on a chunk of input data. The difference equation is
    log(a[n]) = log(a[n - 1]) + step_size * e[n], a[0] = 1 where
    e[n] = log(ref_level) - log(moving_average{|a[n - 1] * x[n]|})
    x is the input signal, z is the output signal, a[i] is i-th gain,
    e[i] is the i-th error, step_size is the step size, ref_level is
    the reference level, and i >= 0, n > 0 are integers (indexes).
    :param data_chunk: input signal chunk
    :type data_chunk: np.ndarray
    :param output_arr: output array to save the results after AGC
    :type output_arr: np.ndarray
    :param chunk_idx: index of current input chunk
    :type chunk_idx: int
    :param ref_level: desired reference output level
    :type ref_level: int
    :param step_size: gain adjustment factor
    :type step_size: float
    :param averaged_amount: samples amount in each average calculation
    :type averaged_amount: int
    :param gain: current gain value
    :type gain: float
    :return: the updated gain value
    :rtype: float
    """
    chunk_length = len(data_chunk)
    total_length = len(output_arr)
    for n in range(chunk_length):
        curr_idx = n + FRAMES_PER_BUFFER * chunk_idx
        start_idx, end_idx = get_window_indexes(curr_idx, averaged_amount,
                                                total_length)
        if start_idx == end_idx:
            continue
        moving_avg = np.mean(np.absolute(output_arr[start_idx:end_idx]))
        error = log(ref_level) - log(moving_avg)
        gain *= exp(step_size * error)
        output_arr[curr_idx] = gain * data_chunk[n]
    return gain

def agc(input_arr, sample_rate, ref_level, step_size, averaged_amount):
    """
    Applies AGC to an input audio array.
    :param input_arr: the input audio data array
    :type input_arr: np.ndarray
    :param sample_rate: sample rate of the audio data
    :type sample_rate: int
    :param ref_level: desired reference output level
    :type ref_level: int
    :param step_size: gain adjustment factor
    :type step_size: float
    :param averaged_amount: samples amount in each average calculation
    :type averaged_amount: int
    :return: output array after AGC
    :rtype: np.ndarray
    """
    gain = 1
    processed_arr = np.ones(len(input_arr))
    for chunk_idx in range(len(input_arr) // FRAMES_PER_BUFFER):
        data_chunk = input_arr[chunk_idx * FRAMES_PER_BUFFER:(chunk_idx + 1) * FRAMES_PER_BUFFER]
        gain = agc_process(data_chunk, processed_arr, chunk_idx, ref_level, step_size, averaged_amount, gain)
    return processed_arr

def main():
    # Example usage
    sample_rate, input_arr = wavfile.read("counting.wav")
    ref_level = 1000
    step_size = 0.001
    averaged_amount = 100
    print(sample_rate)
    output_arr = agc(input_arr, sample_rate, ref_level, step_size, averaged_amount)
    plot_signals(input_arr, output_arr)
    wavfile.write("output.wav", sample_rate, output_arr.astype(np.int16))


if __name__ == "__main__":
    main()

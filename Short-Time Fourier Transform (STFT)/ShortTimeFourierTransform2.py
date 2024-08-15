import math
import numpy as np
import wave
import struct
import tempfile
import time
import threading
import queue
import sounddevice as sd


def read_wav(file_path):
    """
    Extracts the values and the sample rate of the wav. file

    Parameters:
    file_path (String): Wav file (path)

    Returns:
    sample_rate (int): Sample rate of the signal
    audio_data (array): Values of the signal
    """
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        audio_data = wf.readframes(n_frames)

        # Determine the format string for struct.unpack based on sample width and number of channels
        fmt = {1: 'B', 2: 'h', 4: 'i'}[sample_width]  # Mapping sample width to format
        fmt = f"{n_frames * n_channels}{fmt}"

        audio_data = np.array(struct.unpack(fmt, audio_data))

        return sample_rate, audio_data


def write_wav(audio_data, sample_rate):
    """
    Writes a wav file with the input data and sample rate.
    Parameters:
    audio_data (array): Values of the signal
    sample_rate (int): Sample rate of the signal

    Returns:
    file_name (wav file)
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    filename = temp_file.name

    with wave.open(filename, mode='wb') as wf:
        wf.setnchannels(1)  # 1 for mono or 2 for stereo
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)  # Set the Frequency Rate
        wf.writeframes(audio_data.astype(np.int16).tobytes())  # Set the audio frames of the signal
        wf.close()
    return filename

def transpose_matrix(matrix):
    """
    The function returns the transpose matrix of the given matrix
    matrix[i][j] = transpose_matrix[j][i]

    Parameters:
    matrix (2D Matrix): the matrix to be transposed

    Returns:
    transposed_matrix (2D Matrix): the transposed matrix
    """
    # Get the number of rows and columns in the input matrix
    n = len(matrix)
    m = len(matrix[0])

    # Initialize the output matrix with zeros of size m x n
    transposed_matrix = []
    for i in range(m):
        transposed_matrix.append([0 for _ in range(n)])

    # Perform the transpose operation
    for i in range(n):
        for j in range(m):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix


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

def rfft(x):
    """
    Compute the Real FFT (RFFT) of a real-valued input signal.

    Parameters:
    x (list or array): Real-valued input signal.

    Returns:
    list: RFFT of the input signal.
    """
    return np.fft.rfft(x)


def irfft(X):
    """
    Compute the Inverse Real FFT (IRFFT) of a real-valued input signal.

    Parameters:
    X (list or array): Frequency-domain representation of the input signal.

    Returns:
    list: Reconstructed real-valued time-domain signal.
    """
    return np.fft.irfft(X)

def istft(stft_matrix, window_size, hop_size, fs):
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
        output_signal = np.array(irfft(frame))  # output_signal = ifft(fft(signal*window))
        start = i * hop_size
        # To avoid wrong calculations caused by the overlap of different windows we will sum the entire appearances of each time point and then divide by the entire appearance of the window in that time point
        signal[start:start + window_size] += output_signal
        window_sum[start:start + window_size] += window

    # Avoid division by zero
    window_sum[window_sum == 0] = 1
    return write_wav(signal / window_sum, fs)

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

    stft_matrix = transpose_matrix(stft_matrix)

    # Frequency and time vectors - For plotting purposes
    # We used built-in functions for this because it isn't a part of the general analysis of the signal
    freq_vector = np.fft.rfftfreq(window_size, d=1 / fs)
    time_vector = np.arange(num_windows) * hop_size / fs

    # Get the transpose of the matrix so now, each row of the matrix is the DFT of one framed-signal
    return np.array(stft_matrix).T, freq_vector, time_vector


stop_recording_flag = False

def real_time_stft(window_size, hop_size, fs):
    global stop_recording_flag
    stop_recording_flag = False

    # Queue to hold audio data chunks
    q = queue.Queue()

    # List to hold STFT results
    stft_results = []

    def analyze_audio(data, rate):
        # Apply a window function
        window = nuttall_window(window_size)
        data = data * window

        # Compute the FFT of the windowed data
        fft_result = np.fft.rfft(data)
        return fft_result

    def callback(indata, frames, time, status):
        if status:
            print(status)
        # Put the audio data into the queue
        q.put(indata[:, 0])

    def start_recording(rate, chunk_size):
        global stop_recording_flag
        with sd.InputStream(samplerate=rate, blocksize=chunk_size, dtype='int16',
                            channels=1, callback=callback):
            print("Recording started...")
            while not stop_recording_flag:
                sd.sleep(100)
            print("Recording stopped.")

    def stop_recording():
        global stop_recording_flag
        input("Press Enter to stop the recording...")
        stop_recording_flag = True

    recording_thread = threading.Thread(target=start_recording, args=(fs, window_size))
    recording_thread.start()

    # Start the thread for stopping the recording
    stop_thread = threading.Thread(target=stop_recording)
    stop_thread.start()

    buffer = np.zeros(window_size)
    buffer_offset = 0

    while not stop_recording_flag or not q.empty():
        if not q.empty():
            new_data = q.get()
            while len(new_data) > 0:
                available_space = window_size - buffer_offset
                data_to_copy = min(len(new_data), available_space)

                buffer[buffer_offset:buffer_offset + data_to_copy] = new_data[:data_to_copy]
                buffer_offset += data_to_copy
                new_data = new_data[data_to_copy:]

                if buffer_offset >= window_size:
                    fft_result = analyze_audio(buffer, fs)
                    stft_results.append(fft_result)
                    buffer[:hop_size] = buffer[window_size - hop_size:]
                    buffer_offset = hop_size

                # Flush remaining data in buffer if any
    if buffer_offset > hop_size:
            buffer = np.pad(buffer[:buffer_offset], (0, window_size - buffer_offset), 'constant')
            fft_result = analyze_audio(buffer, fs)
            stft_results.append(fft_result)

    stop_recording()

    # Wait for the recording and STFT threads to finish
    recording_thread.join()
    stop_thread.join()

    # Frequency and time vectors - For plotting purposes
    # We used built-in functions for this because it isn't a part of the general analysis of the signal
    freq_vector = np.fft.rfftfreq(window_size, d=1 / fs)
    time_vector = np.arange(len(stft_results)) * hop_size / fs

    return stft_results, freq_vector, time_vector

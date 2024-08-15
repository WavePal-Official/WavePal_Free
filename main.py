import Transmission_and_reception_SSB
from decimation import decimate_wav, decimate
from interpolation import interpolate_wav, interpolate
from Short_Time_Fourier_Transform import stft, istft, irfft, plot_stft
from pitch_estimator import PitchEstimator
from pdm2pcm import pdm_to_wav, normalize_pcm_signal, is_pdm
from acoustic_gain_control import agc
import sounddevice as sd
import numpy as np
import sys
from pdm2pcm import pdm_to_pcm, normalize_pcm_signal
import change_speed
import dc_removal
from Voice_Activity_Detection import voice_activity_detection, voice_activity_detection_real_time
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from Noise_Reduction import noise_reduction
import ShortTimeFourierTransform2
import wave
import struct

# import Transmission_and_reception_SSB


# Variables

SAMPLING_RATE = 44100  # Sampling rate in Hz
BUFFER_SIZE = 1024  # Number of samples per buffer
DECIMATION_FACTOR = 64  # Example decimation factor for PDM to PCM conversion
FIRST_BUFFER_SECONDS = 0.02  # Duration to collect samples for PDM detection
FRAME_SIZE = 1024
HOP_SIZE = 512
L = 0.25  # Change speed parameter
M = 2
# Initialize global variables


initial_buffer = []
initial_fill_done = False
DELAY_SAMPLES = int(FIRST_BUFFER_SECONDS * SAMPLING_RATE)

FILTER_TYPE = 'shanon'  # Default type

# State variables
decision_made = False  # pdm2pcm
pdm_detected = False
samples_collected = 0
samples_for_decision = FIRST_BUFFER_SECONDS * SAMPLING_RATE
# Buffer to store the detected voice activity
detected_voice_activity = []  # VAD result array
agc_signal = []
timestamps = []

audio_buffer = np.array([])

# Buffer to store processed audio data
processed_audio_data = []


# Initialize the PitchEstimator
# pitch_estimator_real_time = PitchEstimator(method='NCF', frame_size=FRAME_SIZE, hop_size=HOP_SIZE, sample_rate=SAMPLING_RATE)

def plot_signal(time, data, title):
    """
    Plot the audio signal.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.show()


# the following functions are related to real-time audio processing:
def append_audio_to_file(file_path, sample_rate):
    global processed_audio_data
    processed_audio_array = np.concatenate(processed_audio_data)
    wavfile.write(file_path, sample_rate, processed_audio_array.astype(np.int16))
    processed_audio_data = []  # Clear the buffer after saving


def save_processed_audio(filename, sample_rate):
    """
    Save the processed audio data to a WAV file.
    :param filename: The filename to save the WAV file.
    :param sample_rate: The sample rate of the audio data.
    """
    global processed_audio_data
    wavfile.write(filename, sample_rate, np.array(processed_audio_data, dtype=np.int16))
    print(f"Processed audio saved to {filename}")

    """
    the real-time processing require callback functions, one for each function that we support. 
    the following functions are those "callback" functions, which are supposed to get the information directly from
    the recorder, and call the function on live data that is restored in an array/list.
    """


def callback_pdm2pcm(indata, outdata, frames, time, status):
    """
    Processes incoming audio data in real time and determines
    whether the data is PDM or PCM.
    Then processes accordingly the data and output the processed data

    :param indata: The incoming audio data buffer
    :param outdata: The buffer where processed audio data will be stored
    :param frames: The number of frames in the buffer
    :param time: The time information related to the buffer
    :param status: Status information about the buffer
    :return:
    """

    global decision_made, pdm_detected, samples_collected

    """
    Global Variables :
    decision_made : A boolean flag indication if a decision has been made about the data type (PDM/PCM)
    pdm_detected : A boolean flag indicating whether PDM data has been detected.
    samples_collected : The number of samples collect this far 
    """

    if status:
        print(status, file=sys.stderr)

    # Convert incoming data to PCM if necessary
    incoming_data = indata[:, 0]  # Assuming indata is a single-channel audio

    if not decision_made:
        samples_collected += len(incoming_data)

        if samples_collected >= samples_for_decision:
            # Make a decision based on the collected samples
            if is_pdm(incoming_data):
                pdm_detected = True
            else:
                pdm_detected = False

            decision_made = True

    if decision_made:
        if pdm_detected:
            # Data is PDM
            pcm_signal = pdm_to_pcm(incoming_data, DECIMATION_FACTOR)
            pcm_signal = normalize_pcm_signal(pcm_signal)
            processed_data = pcm_signal
        else:
            # Data is PCM
            processed_data = incoming_data
    else:
        # Before decision, just pass the data through
        processed_data = incoming_data

    # Ensure the length of processed_data matches the BUFFER_SIZE
    if len(processed_data) > BUFFER_SIZE:
        processed_data = processed_data[:BUFFER_SIZE]
    elif len(processed_data) < BUFFER_SIZE:
        processed_data = np.pad(processed_data, (0, BUFFER_SIZE - len(processed_data)), mode='constant')

    # Output the processed data
    outdata[:, 0] = processed_data


def callback_DC_removal(indata, outdata, frames, time, status):
    """
    Processes incoming audio data in real time and performs DC removal.

    :param indata: The incoming audio data buffer
    :param outdata: The buffer where processed audio data will be stored
    :param frames: The number of frames in the buffer
    :param time: The time information related to the buffer
    :param status: Status information about the buffer
    :return:
    """
    global processed_audio_data, SAMPLING_RATE
    if status:
        print(status, file=sys.stderr)

    incoming_data = indata[:, 0]  # Assuming indata is a single-channel audio

    #  Remove DC component from incoming audio data
    dc_removed_audio = dc_removal.dc_removal_real_time(incoming_data, BUFFER_SIZE, SAMPLING_RATE)
    # Append the processed audio to the buffer
    processed_audio_data.append(dc_removed_audio)
    print(processed_audio_data[:50])
    append_audio_to_file("dc_removal_realtime.wav", SAMPLING_RATE)


def callback_voice_activity_detector(indata, outdata, frames, time, status):
    global detected_voice_activity, SAMPLING_RATE

    if status:
        print(status, file=sys.stderr)

    incoming_data = indata[:, 0]  # Assuming indata is a single-channel audio
    voice_activity, _ = voice_activity_detection_real_time(incoming_data, SAMPLING_RATE)
    print(voice_activity)
    # Store the detected voice activity
    detected_voice_activity.extend(voice_activity)

    current_time = len(detected_voice_activity) / SAMPLING_RATE
    timestamps.extend(np.linspace(current_time, current_time + len(voice_activity) / SAMPLING_RATE, len(voice_activity),
                                  endpoint=False))
    # print(detected_voice_activity)
    # Output the incoming data to maintain real-time processing
    outdata[:, 0] = indata[:, 0]


def plot_voice_activity_detector_real_time():
    global detected_voice_activity, timestamps
    # improved_signal = dc_removal.dc_removal_real_time(np.array(detected_voice_activity, dtype=np.int16), BUFFER_SIZE, SAMPLING_RATE)
    plot_signal(timestamps, detected_voice_activity, "Voice Activity Detection")


def callback_agc(indata, outdata, frames, time, status):
    global agc_signal, processed_audio_data, ref_level, step_size, averaged_amount, SAMPLING_RATE
    if status:
        print(status, file=sys.stderr)
    incoming_data = indata[:, 0]  # Assuming indata is a single-channel audio
    processed_data = agc(incoming_data, SAMPLING_RATE, ref_level, step_size, averaged_amount)

    # Store the processed data in the buffer
    processed_audio_data.extend(processed_data)
    agc_signal.extend(processed_data)

    current_time = len(agc_signal) / SAMPLING_RATE
    timestamps.extend(np.linspace(current_time, current_time + len(processed_data) / SAMPLING_RATE, len(processed_data),
                                  endpoint=False))
    # Output the processed data
    outdata[:, 0] = processed_data[:frames]


def plot_agc_real_time():
    global agc_signal, timestamps
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, agc_signal, label='AGC Processed Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('AGC Processed Audio in Real-Time')
    plt.grid(True)
    plt.legend()
    plt.show()


'''
def callback_stft(indata, outdata, frames, time, status):
    global processed_stft_matrix, processed_time_vector, processed_freq_vector, WINDOW_SIZE, HOP_SIZE
    if status:
        print(status, file=sys.stderr)
    incoming_data = indata[:, 0]  # Assuming indata is a single-channel audio
    processed_data_stft_matrix, processed_data_time_vector, processed_data_freq_vector = stft(indata, WINDOW_SIZE,
                                                                                              HOP_SIZE, SAMPLING_RATE)

    # Store the processed data in the buffer
    processed_stft_matrix.extend(processed_data_stft_matrix)
    processed_time_vector.extend(processed_data_time_vector)
    processed_freq_vector.extend(processed_data_freq_vector)
'''


def callback_noise_reduction(indata, outdata, frames, time, status):
    global initial_buffer, initial_fill_done, processed_audio_data

    if status:
        print(status, file=sys.stderr)

    incoming_data = indata[:, 0]  # Assuming indata is a single-channel audio

    # Add incoming data to the initial buffer
    initial_buffer.extend(incoming_data)

    if not initial_fill_done:
        if len(initial_buffer) < DELAY_SAMPLES:
            # If initial buffer is not yet filled, output silence
            outdata[:, 0] = np.zeros(frames)
            return
        else:
            # If initial buffer is filled, set the flag
            initial_fill_done = True

    # Convert initial buffer to numpy array for processing
    initial_buffer_np = np.array(initial_buffer)

    # Perform Short-Time Fourier Transform (STFT) on the incoming audio data
    stft_matrix, freq_vector, time_vector = stft(incoming_data, FRAME_SIZE, HOP_SIZE, SAMPLING_RATE)

    # Call the noise_reduction function to process the incoming audio data
    processed_stft_matrix = noise_reduction("Counting.wav", stft_matrix, freq_vector)

    # Perform Inverse STFT to convert the processed audio back to time domain
    processed_audio = istft(processed_stft_matrix, FRAME_SIZE, HOP_SIZE)

    # Update the initial buffer by removing the used part
    if len(initial_buffer) > frames:
        initial_buffer = initial_buffer[frames:]

    # Output the processed audio
    outdata[:, 0] = processed_audio[:frames]

    # Store the processed audio data
    processed_audio_data.extend(processed_audio[:frames])


'''

def callback_noise_redcution(indata, outdata, frames, time, status):
    # Doesn't work

def callback_pitch_estimator(indata, outdata, frames, time, status):
    global audio_buffer
    if status:
        print(status, file=sys.stderr)

    # Append incoming audio data to the buffer
    audio_buffer = np.append(audio_buffer, indata[:, 0])

    # Process audio in frames
    while len(audio_buffer) >= FRAME_SIZE:
        frame = audio_buffer[:FRAME_SIZE]
        pitch = pitch_estimator_real_time.estimate_pitch(frame)
        time_stamp = len(pitch_estimator_real_time.pitches) * HOP_SIZE / SAMPLING_RATE
        pitch_estimator_real_time.pitches.append(pitch)
        pitch_estimator_real_time.times.append(time_stamp)
        audio_buffer = audio_buffer[HOP_SIZE:]

'''
transformed_fs = SAMPLING_RATE


def callback_change_speed(indata, outdata, frames, time, status):
    global processed_audio_data, L, FILTER_TYPE, transformed_fs

    if status:
        print(status, file=sys.stderr)

    # Change speed of the incoming audio data
    audio_data = indata[:, 0]
    speed_changed_audio, transformed_fs = change_speed.speed_up_slow_down(audio_data, SAMPLING_RATE, L, FILTER_TYPE)

    processed_audio_data.append(speed_changed_audio)
    append_audio_to_file("change_speed_real_time.wav", SAMPLING_RATE)


def change_speed_real_time():
    global processed_audio_data, transformed_fs
    t_transformed = np.arange(len(processed_audio_data)) / transformed_fs
    # play_audio(processed_audio_data)
    plt.subplot(2, 1, 2)
    plt.plot(t_transformed, processed_audio_data, label='Transformed Signal', color='orange')
    plt.title('Transformed Signal (Speed Up and Slow Down)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()


def callback_decimation(indata, outdata, frames, time, status):
    global processed_audio_data, M

    if status:
        print(status, file=sys.stderr)

    audio_data = indata[:, 0]
    decimation_audio = decimate(audio_data, M)
    processed_audio_data.append(decimation_audio)
    append_audio_to_file("decimation_real_time.wav", SAMPLING_RATE)


def callback_interpolation(indata, outdata, frames, time, status, L=2, type='shanon'):
    global processed_audio_data
    if status:
        print(status, file=sys.stderr)

    # Apply the interpolation filter directly to the input signal
    input_signal = indata[:, 0]
    interpolated_signal = interpolate(input_signal, L, type)

    # Normalize and store the processed signal
    interpolated_signal = np.clip(interpolated_signal, -1.0, 1.0)
    outdata[:, 0] = interpolated_signal[:frames]

    processed_audio_data.append(interpolated_signal[:frames])
    save_processed_audio("interpolated_wav.wav", SAMPLING_RATE)


# Just to view a recorded audio without any real time processing
def recordingFunction():
    """
    Records audio for a specified duration
    """
    duration = float(input("Enter how many seconds to record: "))
    # sample_rate = int(input("Enter sample rate: (recommended is 44100) "))

    # Reset recorded data and set recording flag
    recorded_data = []

    def local_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        recorded_data.extend(indata[:, 0])

    # Create a stream object
    with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, dtype='int16', callback=local_callback):
        print("Recording in real-time... Press Ctrl+C to stop.")
        try:
            sd.sleep(int(duration * 1000))  # Keep the stream open for the specified duration
        except KeyboardInterrupt:
            print("Recording interrupted.")

    # Convert recorded data to a NumPy array
    recorded_data_np = np.array(recorded_data, dtype='int16')
    time_axis = np.linspace(0, len(recorded_data_np) / SAMPLING_RATE, num=len(recorded_data_np))
    plot_signal(time_axis, recorded_data_np, "Recorded Signal")

    # Return the recorded data and sample rate
    return recorded_data_np, SAMPLING_RATE


def start_real_time_processing(callbackFunc):
    """
    Starts a real time audio processing stream, and using the argument of
    'callbackFunc' it'll process the incoming audio data in real time.
    :param callbackFunc:
    :return:
    """
    print("Start")
    # Create a stream object
    with sd.Stream(channels=1, samplerate=SAMPLING_RATE, blocksize=BUFFER_SIZE, callback=callbackFunc, dtype='int16'):
        print("Real Time processing initiated..")
        print("Press Ctrl+C to stop the stream")
        try:
            sd.sleep(20000)  # Keep the stream open indefinitely
        except KeyboardInterrupt:
            print("Stream stopped")


"""
Every function has her own "single execution" function.
Most of the "single execution" functions return firstly the processed wav filename (the ouput wav), 
and then the processed signal.  
"""


def pdm2pcm_single_execution():
    pdm_filename = input("Enter the path to the PDM file: ")
    wav_output = pdm_filename[:-4] + "_pdm2pcm_output.wav"
    decimation_factor = int(input("Enter decimation factor: "))
    sample_rate = int(input("Enter sample rate: "))

    pdm_to_wav(pdm_filename, wav_output, decimation_factor, sample_rate)
    print("Done.")
    return wav_output


def dc_removal_single_execution(wav_filename):
    return dc_removal.main(wav_filename)  # returns wav_output_filename, new_signal


def voice_activity_detector_single_execution(wav_filename):
    voice, fs = voice_activity_detection(wav_filename)
    time = np.arange(len(voice)) / fs
    plt.plot(time, voice)
    plt.xlabel('Time [s]')
    plt.ylabel('Speech')
    plt.title('vad of Audio Signal')
    plt.show()

    return voice


def agc_single_execution(wav_filename):
    sample_rate, input_audio_data_arr = wavfile.read(wav_filename)
    ref_level = int(input("Enter the desired reference output level: "))
    step_size = float(input("Enter the desired gain adjustment factor (step size): "))
    avg_amount_samples = int(input("Enter the samples amount in each average calculation: "))
    output_wav_filename = wav_filename[:-4] + "_agc_output.wav"
    agc_arr = agc(input_audio_data_arr, sample_rate, ref_level, step_size, avg_amount_samples)
    wavfile.write(output_wav_filename, sample_rate, agc_arr.astype(np.int16))
    return output_wav_filename, agc_arr


def stft_single_execution(wav_filename):
    window_size = int(input("Enter window size: "))
    hop_size = int(input("Enter hop size: "))
    fs, input_signal = wavfile.read(wav_filename)
    stft_matrix, freq_vector, time_vector = stft(input_signal, window_size, hop_size, fs)

    reconstructed_signal = istft(stft_matrix, window_size, hop_size)

    # Save STFT and ISTFT results to WAV files
    stft_output_file = "stft_signal.wav"
    istft_output_file = "reconstructed_signal.wav"

    # Reconstruct a signal from the magnitude of the STFT matrix for saving as a WAV file
    magnitude_signal = np.zeros(len(input_signal))
    for i, spectrum in enumerate(stft_matrix.T):
        frame_signal = np.abs(irfft(spectrum))
        frame_signal = frame_signal[:window_size]
        start = i * hop_size
        magnitude_signal[start:start + window_size] += frame_signal
    wavfile.write(stft_output_file, fs, magnitude_signal)
    wavfile.write(istft_output_file, fs, reconstructed_signal)

    plot_stft(stft_matrix, time_vector, freq_vector)
    # TODO: Fix it so it will output a wav file
    print("STFT completed")


def noise_reduction_single_execution(wav_filename):
    fs, signal = wavfile.read(wav_filename)
    original_fs, original_audio = wavfile.read(wav_filename)
    audio_stft, audio_freq, _ = stft(original_audio, 1024, 512, fs)

    # Perform noise reduction
    reduced_signal = noise_reduction(wav_filename, audio_stft, audio_freq)

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
    reduced_file_path = wav_filename[:-4] + "_reduced_noise.wav"
    wavfile.write(reduced_file_path, fs, reduced_signal.astype(np.int16))
    return reduced_file_path, reduced_signal


def pitch_estimation_single_execution(wav_filename):
    method_chosen = input("Enter the method to use for pitch estimation: ['NCF'/'CEP'/'LHS'/'PEF'/'SHR']")
    method_arr = ["NCF", "CEP", "LHS", "PEF", "SHR"]
    if method_chosen.upper() not in method_arr:
        print("Invalid method. ")
        return
    method_chosen = method_chosen.upper()
    frame_size = int(input("Enter number of samples per frame for pitch estimation: "))
    hop_size = int(input("Enter number of samples to shift between frames: "))
    # Create a pitchEstimator object
    pitch_estimator = PitchEstimator(wav_filename, method_chosen, frame_size, hop_size)

    time_stamps, pitch_estimates = pitch_estimator.estimate_pitch(method_chosen, frame_size, hop_size)

    # Write the new audio signal to a WAV file
    output_file_path = wav_filename[:-4] + "_output_with_pitch_estimation.wav"
    wavfile.write(output_file_path, pitch_estimator.sample_rate, pitch_estimates.astype(np.int16))

    # Plot pitch estimation results
    pitch_estimator.plot_pitch_estimation()
    return output_file_path, pitch_estimates


def change_speed_single_execution(wav_filename):
    return change_speed.main(wav_filename)


def decimation_single_execution(wav_filename):
    M = int(input("Enter decimation factor: "))
    output_wav_filepath = wav_filename[:-4] + "_decimate_wav.wav"
    decimate_wav(wav_filename, M, output_wav_filepath)
    return output_wav_filepath


def interpolation_single_execution(wav_filename):
    output_wav_file_path = wav_filename[:-4] + 'interpolated_signal.wav'
    L = int(input("Enter Interpolation factor: "))
    method_type = 'shanon'
    changeMethod = input(f"The current method is:{method_type}. Would you like to change it? [Y/N]")
    if changeMethod.upper() == "Y":
        method_arr = ['shanon', 'ZOH', 'FOH']
        method_type = input("Enter the method you'd like to use: [shanon/ZOH/FOH]")
        if method_type not in method_arr:
            print("Invalid method.")
            return

    elif changeMethod.upper() == "N":
        pass

    else:
        print("Invalid input.")
        return
    print(f"Doing Interpolation using {method_type} method, and factor:{L}..")
    y_wav = interpolate_wav(wav_filename, L, type=method_type, output_path=output_wav_file_path)
    return output_wav_file_path


def transmission_and_reception_SSB_single_execution(wav_filename):
    fs, voice_message = wavfile.read(wav_filename)

    # Transmission_and_reception_SSB.process_audio(signal, fs)


def plot_stft(stft_matrix, time_vector, freq_vector):
    magnitude = np.abs(ShortTimeFourierTransform2.transpose_matrix(stft_matrix))
    magnitude_db = 20 * np.log10(magnitude + 1e-6)  # Adding a small constant to avoid log(0)
    plt.pcolormesh(time_vector, freq_vector, magnitude_db, vmin=0, shading='gouraud')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Short-Time Fourier Transform')
    plt.show()


def plot_signal_stft(signal, fs):
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.title('Waveform')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


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

        audio_data = struct.unpack(fmt, audio_data)

        # If stereo, convert to mono by averaging the channels
        if n_channels > 1:
            audio_data = np.mean(np.reshape(audio_data, (-1, n_channels)), axis=1)

        return sample_rate, np.array(audio_data)


def play_audio(file_path):
    """
    Plays a WAV file.

    Parameters:
    file_path (str): Path to the WAV file.
    """
    print("Playing...")

    # Open the WAV file
    wf = wave.open(file_path, 'rb')

    # Define a callback function to stream audio data
    def callback(outdata, frames, time, status):
        data = wf.readframes(frames)
        if len(data) == 0:
            raise sd.CallbackStop()
        outdata[:len(data) // wf.getsampwidth()] = np.frombuffer(data, dtype=np.int16).reshape(-1, wf.getnchannels())

    # Start streaming audio
    with sd.OutputStream(samplerate=wf.getframerate(), channels=wf.getnchannels(), dtype=np.int16, callback=callback):
        sd.sleep(int(wf.getnframes() / wf.getframerate() * 1000))

    print("Playback finished.")


# a list of all the functions we support:
single_functions_existing = ["interpolation", "decimation", "change speed", "pitch estimation",
                             "noise reduction", "stft", "acoustic gain control", "voice activity detector",
                             "dc removal", "pdm2pcm"]

single_functions_existing_real_time = ["stft", "dc removal", "voice activity detector",
                                       "acoustic gain control", "interpolation"]


def userChooseAction():
    action = input(
        "Would you like to turn on the entire system or a single function? (Entire system / single function)")
    if action.lower() == "entire system":
        mode = input("Would you like to execute the code in real time or on existing recorder? (real time / existing)")
        # Run Entire System
        if mode.lower() == "real time":

            print("Starting system in real time mode..")
        elif mode.lower() == "existing":
            print("Starting the entire system on existing file..")
            # Firstly, running PDM2PCM as single execution and receiving the wav file
            wav_pdm2pcm = pdm2pcm_single_execution()  # The wav output
            original_fs, original_signal = wavfile.read(wav_pdm2pcm)
            # Now, DC removal
            print(original_fs)
            print("DC removalll")
            DC_removal_signal = dc_removal_single_execution(
                wav_pdm2pcm)  # Runs the dc removal single execution function on the output wav from pdm2pcm
            # Plot the signal after DC removal
            voice, sample_rate = plot_voice_activity_detector_real_time(DC_removal_signal)
            time = np.arange(len(voice)) / sample_rate

            plot_signal(time, voice, "Voice By Time")
            print("AGCC")
            # Now we'll execute AGC
            ref_level = int(input("Enter desired reference output level (recommended: 1000): "))
            step_size = float(input("Enter gain adjustment factor (recommended: 0.001): "))
            averaged_amount = int(input("Enter averaged amount of samples in each average calculation (recommended "
                                        "100): "))

            output_agc = agc(DC_removal_signal, original_fs, ref_level, step_size, averaged_amount)
            wavfile.write("agc_output.wav", original_fs, output_agc.astype(np.int16))

            # STFT
            audio_stft, audio_freq, time_vector = stft(output_agc, 1024, 512, original_fs)

            '''
            # Becuase stft is a matrix we'll save the result of the stft in a wav file (istft), run
            # noise reduction on it, and then return it to the stft
            # So, save intermediate STFT result as a wav file.
            intermediate_stft_signal = istft(stft_matrix, 1024, 512)
            stft_file_path = "intermediate_stft.wav"
            wavfile.write(stft_file_path, fs, intermediate_stft_signal)
            # Noise reduction
            #threshold = float(input("Enter threshold value (recommended: 0.02):  "))
            # The noise reduction function returns 4 values, but we only care for the new signal
            '''
            # Perform noise reduction

            noise_reduction_signal = noise_reduction("agc_output.wav", audio_stft, audio_freq)
            wavfile.write("noise_reduction_entire_system.wav", original_fs, noise_reduction_signal.astype(np.int16))
            plt.figure(figsize=(15, 15))

            plt.subplot(2, 1, 1)
            plt.title('Original Audio')
            plt.plot(np.arange(len(output_agc)) / original_fs, original_signal)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')

            plt.subplot(2, 1, 2)
            plt.title('Processed Audio')
            plt.plot(np.arange(len(noise_reduction_signal)) / original_fs, noise_reduction_signal)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            pitch_Estimator = PitchEstimator("noise_reduction_entire_system.wav", 'NCF', 1024, 512)
            times, pitches = pitch_Estimator.estimate_pitch('NCF', 1024, 512)
            pitch_Estimator.plot_pitch_estimation()
            changed_speed_wav_file = change_speed_single_execution("noise_reduction_entire_system.wav")
            decimation_entire_system_wav = decimation_single_execution(changed_speed_wav_file)
            interpolation_single_execution(decimation_entire_system_wav)



    # if "single function" is chosen, we check which function is chosen
    elif action.lower() == "single function":
        mode = input("Would you like to execute the code in real time or on existing recorder? (real time / existing)")
        if (mode.lower() != "existing" and mode.lower() != "real time"):
            print("Invalid input.")
            return

        else:
            print("Available functions for existing file:")
            for func in single_functions_existing: print(f"{func}\n")
            print("============Available function for real time:==========")
            chosen_single_function = input("Which function would you like to execute?")
            for func in single_functions_existing_real_time: print(f"{func}\n")
            # Checks if the function exists and valid
            if chosen_single_function in single_functions_existing:
                pass
            else:
                print("Error: Invalid input, the input should be one of the above functions.")
                return

            # if we want single function real time processing, the following chunk of code runs the chosen function:
            if mode.lower() == "real time":

                print(f"Executing single function {chosen_single_function} in real time..")
                if chosen_single_function.lower() == "pdm2pcm":
                    start_real_time_processing(callback_pdm2pcm)
                elif chosen_single_function.lower() == "dc removal":
                    start_real_time_processing(callback_DC_removal)
                    fs, dc_removed_recording = wavfile.read("dc_removal_realtime.wav")
                    time = np.linspace(0, len(dc_removed_recording) / fs, num=len(dc_removed_recording))

                    plot_signal(time, dc_removed_recording, 'Recording signal after DC Removal')
                elif chosen_single_function.lower() == "voice activity detector":
                    start_real_time_processing(callback_voice_activity_detector)
                    plot_voice_activity_detector_real_time()

                elif chosen_single_function.lower() == "acoustic gain cocntrol":
                    start_real_time_processing(callback_agc)
                    save_processed_audio("agc_real_time.wav", SAMPLING_RATE)
                    plot_agc_real_time()

                elif chosen_single_function.lower() == "stft":
                    fs = 41000
                    window_size = int(input("Enter window size: "))
                    hop_size = int(input("Enter hop size: "))
                    stft_matrix, freq_vector, time_vector = ShortTimeFourierTransform2.real_time_stft(window_size,
                                                                                                      hop_size, fs)
                    plot_stft(stft_matrix, time_vector, freq_vector)
                    print("istft: starting..")
                    output_signal = ShortTimeFourierTransform2.istft(stft_matrix, window_size, hop_size, fs)
                    print("istft: Done.")

                    fs, output_data = ShortTimeFourierTransform2.read_wav(output_signal)

                    plot_signal_stft(output_data, fs)
                    play_audio(output_signal)

                elif chosen_single_function.lower() == "noise reduction":
                    start_real_time_processing(callback_noise_reduction)

                elif chosen_single_function.lower() == "change speed":
                    start_real_time_processing(callback_change_speed)
                    change_speed_real_time()


                elif chosen_single_function.lower() == "decimation":
                    start_real_time_processing(callback_decimation)

                elif chosen_single_function.lower() == "interpolation":
                    start_real_time_processing(callback_interpolation)


            # start_real_time_processing()
            # MISSING

            # for existing file:
            elif mode.lower() == "existing":
                print("Available functions: ")
                for func in single_functions_existing: print(f"{func}\n")
                print(f"Executing single function {chosen_single_function} on existing recorder..")
                # In case of running on existing file
                '''
                Since pdm2pcm is the only function that doesn't require to enter wav file,
                we'll split to cases - if it's the pdm2pcm and all the other functions.
                '''
                if chosen_single_function.lower() == "pdm2pcm":
                    wav_output = pdm2pcm_single_execution()
                    print(f"The result is saved in {wav_output}.")
                    return
                else:
                    wav_filename = input("Enter the path to the WAV file: ")
                    if chosen_single_function.lower() == "dc removal":
                        wav_output_filename, _ = dc_removal_single_execution(wav_filename)
                        print(f"The new wav file is saved in: {wav_output_filename}")

                    elif chosen_single_function.lower() == "voice activity detector":
                        voice_activity_detector_single_execution(wav_filename)

                    elif chosen_single_function.lower() == "acoustic gain control":
                        output_wav_filename, _ = agc_single_execution(wav_filename)
                        print(f"The processed wav file is : {output_wav_filename}")

                    elif chosen_single_function.lower() == "stft":
                        stft_single_execution(wav_filename)

                    elif chosen_single_function.lower() == "noise reduction":
                        print(f"The reduced noise wav file is : {noise_reduction_single_execution(wav_filename)[0]}")

                    elif chosen_single_function.lower() == "pitch estimation":
                        print(f"The wav with pitch estimation is: {pitch_estimation_single_execution(wav_filename)[0]}")

                    elif chosen_single_function.lower() == "change speed":
                        print(f"The transformed signal is in: {change_speed.main(wav_filename)[0]}")

                    elif chosen_single_function.lower() == "decimation":
                        print(f"The decimate wav: {decimation_single_execution(wav_filename)}")

                    elif chosen_single_function.lower() == "interpolation":
                        print(f"The interpolate wav: {interpolation_single_execution(wav_filename)}")

                    elif chosen_single_function.lower() == "transmisson and reception ssb":
                        fs, voice_message = wavfile.read(wav_filename)
                        ssb_signal, demodulated_signal = Transmission_and_reception_SSB.process_audio(voice_message, fs,
                                                                                                      fc)
                        # Play the original, modulated, and demodulated signals
                        print("Playing original message...")
                        Transmission_and_reception_SSB.play_signal(voice_message, fs)
                        print("Playing SSB modulated signal...")
                        Transmission_and_reception_SSB.play_signal(ssb_signal, fs)
                        print("Playing demodulated message...")
                        Transmission_and_reception_SSB.play_signal(demodulated_signal, fs)
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
            else:
                print("Error: Invalid input, the input should be 'real time' or 'existing'")
                return

    else:
        print("Error: Invalid input, the input should be 'Entire system' or 'Single function'")
        return


def main():
    while True:
        userChooseAction()
    # recordingFunction()


if __name__ == "__main__":
    main()

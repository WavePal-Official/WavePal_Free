from interpolation import interpolate
from decimation import decimate
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def slow_down_wav(signal, fs, L, filter_type='shanon'):
    signal = signal.astype(float)  # Convert to float for processing

    # Interpolate to slow down the signal
    y = interpolate(signal, L, filter_type)

    # Clip and convert back to int16
    y = np.clip(y, -32768, 32767)
    y = y.astype(np.int16)

    return y, fs


def speed_up_wav(signal, fs, L):
    signal = signal.astype(float)  # Convert to float for processing

    # Decimate to speed up the signal
    y = decimate(signal, L)

    # Clip and convert back to int16
    y = np.clip(y, -32768, 32767)
    y = y.astype(np.int16)

    return y, fs // L  # Return the new sample rate


def speed_up_slow_down(signal, fs, L, filter_type='shanon'):
    # Check if L is an integer
    if L.is_integer():
        # Slow down the signal
        return speed_up_wav(signal, fs, int(L))
    else:
        # Handle non-integer L by adjusting the process
        L_int = int(np.round(L * 4))
        slowed_signal, _ = slow_down_wav(signal, fs, L_int, filter_type)
        speeded_signal, _ = speed_up_wav(slowed_signal, fs, 4)  # Use only the signal for speed_up_wav
        return speeded_signal, fs


def main(input_wav_file):
    L = float(input(
        "Enter your speed/slow factor, use 0.25/0.5/0.75..."))  # Factor to slow down the signal (can be a non-integer for demonstration)
    while L <= 0 or not L % 0.25 == 0:
        print("input not in the form of 0.25 * signal or negative number")
        L = float(input("Enter your speed/slow factor, use 0.25/0.5/0.75..."))

    filter_type = 'shanon'
    changeFilterType = input(f"The current filter type is {filter_type}, would you like to change it? [Y/N]")
    if changeFilterType.upper() == "N":
        print(f"Continuing with filter type: {filter_type}.. ")
    elif changeFilterType.upper() == "Y":
        wanted_type = input("Enter the filter type you would like then: [shanon/ZOH/FOH]")
        if wanted_type != "shanon" and wanted_type != "ZOH" and wanted_type != "FOH":
            print("Invalid filter type was entered.")
            return
        filter_type = wanted_type
    else:
        print("Invalid input entered, canceling")
        return

    # Read the input WAV file
    fs, signal = wavfile.read(input_wav_file)

    # Apply the speed_up_slow_down function
    transformed_signal, transformed_fs = speed_up_slow_down(signal, fs, L, filter_type)

    # Create time arrays for plotting
    t_original = np.arange(len(signal)) / fs
    t_transformed = np.arange(len(transformed_signal)) / transformed_fs

    # Plot the original and transformed signals
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_original, signal, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_transformed, transformed_signal, label='Transformed Signal', color='orange')
    plt.title('Transformed Signal (Speed Up and Slow Down)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()
    output_wav = input_wav_file[:-4] + "_transformed_signal_speed.wav"
    wavfile.write(output_wav, fs, transformed_signal)
    return output_wav, transformed_signal

if __name__ == "__main__":
    main()

import numpy as np
import wave
import matplotlib.pyplot as plt


class PitchEstimator:
    """
    The PitchEstimator class is designed to estimate the pitch of audio signals using various methods.
    It takes an audio file as input, processes it frame by frame, and applies a specified pitch estimation method to compute the pitch over time.
    It also provides functionality to plot the original audio signal alongside the estimated pitch.
    """
    def __init__(self, audio_path, method, frame_size, hop_size):
        """
        Initialize the PitchEstimator with audio file path, method for pitch estimation, frame size, and hop size.

        Parameters:
        audio_path (str): Path to the audio file.
        method (str): Method used for pitch estimation (e.g., 'NCF', 'CEP', 'LHS', 'PEF', 'SHR').
        frame_size (int): Number of samples per frame for pitch estimation.
        hop_size (int): Number of samples to shift between frames.
        """
        self.audio_path = audio_path
        self.method = method
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate, self.audio_data = self.load_audio()  # Load audio data and sample rate
        self.audio_data = self.normalize_audio(self.audio_data)  # Normalize the audio data

    def load_audio(self):
        """
        Load the audio file, gets important parameters of the audio file and return the sample rate and audio data.

        specifically it returns a tuple containing the sample rate (int) and audio data (numpy.ndarray).
        (a tuple is a type of data structure that is very similar to lists. The main difference between the two is that tuples are immutable, meaning they cannot be changed once they are created)
        """
        with wave.open(self.audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()  # Get the sample rate of the audio file
            n_channels = wf.getnchannels()  # Get the number of channels in the audio file
            n_frames = wf.getnframes()  # Get the total number of frames in the audio file
            audio_data = wf.readframes(n_frames)  # Read all the frames from the audio file
            audio_data = np.frombuffer(audio_data, dtype=np.int16)  # Convert the audio data to a NumPy array
            if n_channels == 2:  # If the audio file is in stereo
                audio_data = audio_data[::2]  # Convert to mono by taking only one channel
        return sample_rate, audio_data  # Return the sample rate and the mono audio data

    def normalize_audio(self, audio_data):
        """
        Normalize the audio data to be within the range [-1, 1].

        Gets audio_data (numpy.ndarray): The audio data to normalize and returns numpy.ndarray: The normalized audio data.
        """
        return audio_data / np.max(np.abs(audio_data))

    def normalized_cross_correlation(self, signal):
        """
        Calculate the normalized cross-correlation of a signal(used for the NCF method of pitch estimation).

        Parameters:
        signal (numpy.ndarray): The input signal.

        Returns:
        numpy.ndarray: The normalized cross-correlation result.
        """
        norm_signal = (signal - np.mean(signal)) / np.std(signal)  # Normalize the signal
        result = np.correlate(norm_signal, norm_signal, mode='full')  # Compute the full cross-correlation- correlation at all possible lags, including negative ones
        return result[result.size // 2:]  # Return the second half of the result (positive lags)

    def cepstral_analysis(self, signal):
        """
        Calculate the cepstral analysis of a signal(used for the CEP method of pitch estimation).

        Parameters:
        signal (numpy.ndarray): The input signal.

        Returns:
        numpy.ndarray: The cepstral analysis result.
        """
        spectrum = np.fft.fft(signal)  # Compute the Fourier transform of the signal
        log_spectrum = np.log(np.abs(spectrum))  # Compute the logarithm of the magnitude spectrum(the magnitude spectrum is the absolute value of the Fourier Transform result)
        cepstrum = np.fft.ifft(log_spectrum).real  # Compute the inverse Fourier transform of the log spectrum and gets the real part
        return cepstrum  # Return the real part of the cepstrum(cepstrum is defiend as the result of computing the inverse Fourier transform of the logarithm of the estimated signal spectrum).

    def likelihood_harmonic_summation(self, signal, sample_rate):
        """
        Calculate the likelihood harmonic summation of a signal(used for the LHS method of pitch estimation).

        Parameters:
        signal (numpy.ndarray): The input signal.
        sample_rate (int): The sample rate of the audio.

        Returns:
        numpy.ndarray: The harmonic summation result.
        """
        spectrum = np.abs(np.fft.fft(signal))  # Compute the magnitude spectrum of the signal(the magnitude spectrum is the absolute value of the Fourier Transform result)
        harmonics = np.zeros_like(spectrum)  # Initialize an array for harmonic summation
        for h in range(1, 11):  # loop that iterates over the harmonic numbers from 1 to 10(the n-th harmonic number is the sum of the reciprocals of the first n natural numbers)
            # np.roll(spectrum, shift) shifts the spectrum array by a specified number of elements (given by shift). This operation aligns the harmonic components in the spectrum for summation.
            # h * len(spectrum) // (2 * sample_rate) calculates the shift amount based on the harmonic number h. The formula ensures that each harmonic is aligned correctly with respect to the fundamental frequency.
            harmonics += np.roll(spectrum, h * len(spectrum) // (2 * sample_rate))  # Sum the harmonics
        return harmonics  # Return the harmonic summation

    def peak_enhanced_function(self, signal):
        """
        Calculate the peak enhanced function of a signal(used for the PEF method of pitch estimation).

        Parameters:
        signal (numpy.ndarray): The input signal.

        Returns:
        numpy.ndarray: The peak enhanced function result.
        """

        result = np.correlate(signal, signal, mode='full') # Compute the full cross-correlation- correlation at all possible lags, including negative ones
        result = result[result.size // 2:] # Extract the second half of the result (positive lags)

        # Enhance the peaks by zeroing out local minima
        enhanced = np.copy(result)# Make a copy of the result array
        for i in range(1, len(result) - 1):
            #the condition checks if the current element is a local minimum- an element that is smaller than both its preceding (result[i - 1]) and succeeding (result[i + 1]) elements.
            if result[i] < result[i - 1] and result[i] < result[i + 1]:
                enhanced[i] = 0  # Zero out local minimum
        return enhanced # Return the enhanced array

    def spectral_harmonic_ratio(self, signal, sample_rate):
        """
        Calculate the spectral harmonic ratio of a signal(used for the SHR method of pitch estimation).

        Parameters:
        signal (numpy.ndarray): The input signal.
        sample_rate (int): The sample rate of the audio.

        Returns:
        numpy.ndarray: The spectral harmonic ratio result.
        """
        spectrum = np.abs(np.fft.fft(signal))  # Compute the magnitude spectrum of the signal(the magnitude spectrum is the absolute value of the Fourier Transform result)
        freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)  # Compute the frequency bins(computes the frequencies corresponding to each element in the FFT result).
        positive_freqs = freqs[freqs >= 0]  # Keep only the positive frequencies
        positive_spectrum = spectrum[freqs >= 0]  # Keep only the spectrum values corresponding to positive frequencies
        shr = np.zeros_like(positive_spectrum)  # Initialize an array for the spectral harmonic ratio

        for fundamental_freq in range(20, 1000, 10):  # loop that iterates over possible fundamental frequencies from 20 Hz to 1000 Hz, with a step size of 10 Hz
            harmonic_sum = np.zeros_like(positive_spectrum)  # Initialize an array for harmonic sum
            for harmonic in range(1, 6):  # loop that iterates over the harmonic numbers from 1 to 6(the n-th harmonic number is the sum of the reciprocals of the first n natural numbers)
                harmonic_freq = fundamental_freq * harmonic  # Calculate the harmonic frequency
                if harmonic_freq < sample_rate / 2:  # Check if the harmonic frequency is within Nyquist limit (half the sample rate)- beyond which aliasing would occur.
                    bin_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))  #  minimal difference- finds the index of the frequency bin in positive_freqs that is closest to the current harmonic frequency
                    harmonic_sum[bin_idx] += positive_spectrum[bin_idx]  # Sum the harmonic magnitudes
            shr += harmonic_sum  # Adds the harmonic sum for the current fundamental frequency to the shr array.
        return shr  # Return the spectral harmonic ratio

    def estimate_pitch(self, method, frame_size, hop_size):
        """
        Estimate pitch using the specified method.

        Parameters:
        method (str): Method for pitch estimation (e.g., 'NCF', 'CEP', 'LHS', 'PEF', 'SHR').
        frame_size (int): Number of samples per frame.
        hop_size (int): Number of samples to shift between frames.

        Returns:
        tuple: A tuple containing the time stamps (numpy.ndarray) and pitch estimates (numpy.ndarray).
        (a tuple is a type of data structure that is very similar to lists. The main difference between the two is that tuples are immutable, meaning they cannot be changed once they are created)

        What it does in summary:
        Iterate Over Audio Data- Each iteration extracts a frame of the audio signal. Depending on the specified method (NCF, CEP, LHS, PEF, SHR), the code calculates the pitch for the current frame.
        after it Stores the pitch estimates and corresponding time stamps.in the end it Returns the time stamps and pitch estimates as NumPy arrays.
        """
        pitches = []  # Initialize an empty list for pitch estimates
        times = []  # Initialize an empty list for time stamps

        for i in range(0, len(self.audio_data) - frame_size, hop_size):  #  The loop iterates over the entire audio data, processing it in frames.
            frame = self.audio_data[i:i + frame_size]  # Extract a frame of audio data-Slices the audio data from index i to i + frame_size.

            if method == 'NCF':  # Normalized Cross-Correlation Function
                result = self.normalized_cross_correlation(frame)  # Calculate the normalized cross-correlation of a signal using the Normalized Cross-Correlation function from earlier
                # np.diff computes the nth difference along the given axis of an input array. For example, for an input array: a = [n1, n2, n3, n4, n5] numpy. diff(x) = [n2-n1, n3-n2, n4-n3, n5-n4]
                # in this case we get an array of the differences of the autocorrelations
                d = np.diff(result)  # The d array helps identify where the autocorrelation function changes direction, which can indicate potential peaks.
                # identifies the first point in the d array where the autocorrelation function starts increasing after initially decreasing.
                # This is useful to avoid the dominant peak at zero lag and to find the first meaningful peak
                start = np.where(d > 0)[0][0]
                # the first significant peak in the autocorrelation function (after the zero lag) corresponds to the pitch period of the signal.
                peak = np.argmax(result[start:]) + start# finds the index of the maximum value in the autocorrelation function starting from the start index.
                # The peak value is the lag (in number of samples) where the autocorrelation function has a peak, indicating the period of the fundamental frequency
                # The pitch period is converted to pitch frequency using the sample rate because In terms of sampling rate, the period (in seconds) is given by the number of samples divided by the sample rate
                # If no peak is found (i.e., peak == 0), the pitch is set to zero.
                pitch = self.sample_rate / peak if peak != 0 else 0

            elif method == 'CEP':  # Cepstral Analysis
                cepstrum = self.cepstral_analysis(frame)  #gets cepstrum(the result of computing the inverse Fourier transform of the logarithm of the estimated signal spectrum) using the cepstral_analysis function from earlier
                peak = np.argmax(cepstrum[1:]) + 1  # finds the index of the maximum value in the cepstrum, ignoring the zeroth element to avoid the DC componen
                pitch = self.sample_rate / peak if peak != 0 else 0  # Calculate pitch in similar fashion to the NCF method

            elif method == 'LHS':  # Likelihood Harmonic Summation
                lhs = self.likelihood_harmonic_summation(frame, self.sample_rate)  # Calculate LHS- likelihood harmonic summation of a signal using the likelihood_harmonic_summation function from earlier
                peak = np.argmax(lhs)  # finds the index of the maximum value in lhs
                pitch = self.sample_rate / peak if peak != 0 else 0  # Calculate pitch in similar fashion to the NCF method

            elif method == 'PEF':  # Peak Enhanced Function
                pef = self.peak_enhanced_function(frame) #Computes the peak enhanced function of the frame using the peak_enhanced_function function from earlier.
                d = np.diff(pef)#Computes the difference between consecutive elements in pef. More detailed explanation in NCF method.
                start = np.where(d > 0)[0][0] #Finds the first index where the difference is positive.. More detailed explanation in NCF method.
                peak = np.argmax(pef[start:]) + start #finds the index of the maximum value in the autocorrelation function starting from the start index.
                pitch = self.sample_rate / peak if peak != 0 else 0# Calculate pitch in similar fashion to the NCF method

            elif method == 'SHR':  # Spectral Harmonic Ratio
                shr = self.spectral_harmonic_ratio(frame, self.sample_rate)  #  Computes the spectral harmonic ratio of the frame using the spectral_harmonic_ratio from earlier.
                peak = np.argmax(shr)  # Find the index of the maximum value in shr
                pitch = self.sample_rate / peak if peak != 0 else 0  # Calculate pitch in similar fashion to the NCF method

            else:
                raise ValueError(f"Unknown method: {method}")  # Raise an error if the method is unknown

            pitches.append(pitch)  #Adds the calculated pitch to the pitches list.
            # times.append(i / sample_rate) converts the current frame's starting index (i) into time (in seconds) and adds this value to the times array
            # This list will contain the corresponding time (in seconds) for each pitch estimate.
            times.append(i / self.sample_rate)

        return np.array(times), np.array(pitches)  # Return time stamps and pitch estimates

    def plot_pitch_estimation(self):
        """
        Plot the pitch estimation results alongside the original audio signal.
        """
        times, pitches = self.estimate_pitch(self.method, self.frame_size, self.hop_size)  # Estimate the pitch

        plt.figure(figsize=(12, 6))  # Create a new figure with a specified size

        # Plot the original audio signal
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data)), self.audio_data)
        plt.title('Original Audio Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

        # Plot the pitch estimation results
        plt.subplot(2, 1, 2)
        plt.plot(times, pitches, 'r-',label=f'Estimated Pitch ({self.method})')  # Plot the pitch estimates as a blue line
        plt.title(f'Pitch Estimation using {self.method}')
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch [Hz]')

        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.show()  # Display the plots

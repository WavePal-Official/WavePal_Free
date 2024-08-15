import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def read_wav(file_path):
    fs, audio = wavfile.read(file_path)
    if audio.ndim == 2:  # If stereo, convert to mono
        audio = np.mean(audio, axis=1)
    return fs, audio

def calculate_energy(signal):   # Make an array of the energy
    energy = np.zeros(len(signal),dtype=float)
    for i in range(0, len(signal)):
        frame = signal[i]
        frame_energy = frame**2
        energy[i] = (frame_energy)
    return energy

def calculateMin(signal):   # Calculate voice activity threshold
    mean_signal = np.mean(signal)
    mean_signal*=2
    below_mean_values = signal[signal < mean_signal]
    mean_below_mean = np.mean(below_mean_values)
    min=mean_below_mean*1.5
    return min

def vad(energy, min):   # Make a binary array of speach detection
    voice_activity = np.zeros(len(energy),dtype=float)
    for i in range(0, len(energy)):
        frame = energy[i]
        if frame>=min:
            voice_activity[i]=1
    return voice_activity

def smooth_vad(va, fs): # Smoothen the signal by comparing 0 amount to human talking frequency
    MIN_HUMAN=90
    max = fs/MIN_HUMAN

    for i in range(0, len(va)):
        if(va[i]==0):
            continue
        j=1
        if(i+j>=len(va)-1):
            return va
        while(va[i+j]==0):
            j+=1
            if(i+j>len(va)-1):
                break
        if(j<=max):
            va[i+1:i+j]=1
        i+=j
    return va

def voice_activity_detection(file_path):
    fs, audio_signal = read_wav(file_path)
    energy = calculate_energy(audio_signal)
    min = calculateMin(energy)
    voice = vad(energy, min)
    voice = smooth_vad(voice, fs)
    return voice, fs

def voice_activity_detection_real_time(signal, sample_rate):
    energy = calculate_energy(signal)
    min = calculateMin(energy)
    voice = vad(energy, min)
    voice = smooth_vad(voice, sample_rate)
    return voice, sample_rate

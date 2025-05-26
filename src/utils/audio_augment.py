import numpy as np
import librosa
import scipy.signal

def add_white_noise(y, noise_level=0.005):
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def apply_random_gain(y, min_gain=0.7, max_gain=1.3):
    gain = np.random.uniform(min_gain, max_gain)
    return y * gain

def apply_time_stretch(y, rate_range=(0.9, 1.1)):
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(y, rate)

def apply_pitch_shift(y, sr, semitone_range=(-2, 2)):
    n_steps = np.random.uniform(*semitone_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def apply_random_bandpass(y, sr, low_hz=300, high_hz=3000):
    low = np.random.uniform(low_hz, sr // 4)
    high = np.random.uniform(sr // 4, high_hz)
    b, a = scipy.signal.butter(2, [low / (sr / 2), high / (sr / 2)], btype='band')
    return scipy.signal.lfilter(b, a, y)

def augment_signal(y, sr):
    if np.random.rand() < 0.9:
        y = apply_random_gain(y)
    if np.random.rand() < 0.5:
        y = add_white_noise(y)
    if np.random.rand() < 0.5:
        y = apply_time_stretch(y)
    if np.random.rand() < 0.5:
        y = apply_pitch_shift(y, sr)
    if np.random.rand() < 0.3:
        y = apply_random_bandpass(y, sr)
    return y
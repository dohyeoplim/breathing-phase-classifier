import numpy as np
import librosa
import scipy.signal
import random

def add_white_noise(y, noise_level_range=(0.0005, 0.006)): # 노이즈 범위 약간 확대
    noise_level = np.random.uniform(*noise_level_range)
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def apply_random_gain(y, min_gain=0.65, max_gain=1.35): # 게인 범위 약간 확대
    gain = np.random.uniform(min_gain, max_gain)
    return y * gain

def apply_time_stretch(y, expected_length, rate_range=(0.85, 1.15)): # 기존 유지
    rate = np.random.uniform(*rate_range)
    try:
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        if len(y_stretched) < expected_length:
            y_stretched = np.pad(y_stretched, (0, expected_length - len(y_stretched)), mode='constant')
        else:
            y_stretched = y_stretched[:expected_length]
        return y_stretched
    except Exception:
        if len(y) < expected_length:
            y_padded = np.pad(y, (0, expected_length - len(y)), mode='constant')
            return y_padded
        elif len(y) > expected_length:
            return y[:expected_length]
        return y

def apply_pitch_shift(y, sr, semitone_range=(-2, 2)): # 피치 변경 범위 약간 확대
    try:
        n_steps = np.random.uniform(*semitone_range)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except Exception:
        return y

def apply_random_bandpass(y, sr, low_hz_range=(150, 400), high_hz_range=(1800, 4500)): # 기존 유지
    low_hz_val = np.random.uniform(*low_hz_range)
    high_hz_val = np.random.uniform(low_hz_val + 100, min(np.random.uniform(*high_hz_range), sr / 2.0 - 50.0))
    
    if low_hz_val >= high_hz_val or high_hz_val >= sr / 2.0:
        return y
    
    nyquist = sr / 2.0
    low = low_hz_val / nyquist
    high = high_hz_val / nyquist
    filter_order = 3 
    try:
        b, a = scipy.signal.butter(filter_order, [low, high], btype='band')
        return scipy.signal.lfilter(b, a, y)
    except ValueError:
        return y

def apply_time_shift(y, sr, shift_max_ms=75): # 최대 이동 시간 약간 확대
    shift_max_samples = int(shift_max_ms * sr / 1000)
    if shift_max_samples == 0:
        return y
    shift = random.randint(-shift_max_samples, shift_max_samples)
    y_shifted = np.roll(y, shift)
    if shift > 0:
        y_shifted[:shift] = 0
    elif shift < 0:
        y_shifted[shift:] = 0
    return y_shifted

def augment_signal(y, sr, expected_length):
    # 적용 확률을 이전보다 약간씩 높임
    if np.random.rand() < 0.75: # 게인 적용 확률 (이전 0.6 -> 0.75)
        y = apply_random_gain(y, min_gain=0.65, max_gain=1.35)
    if np.random.rand() < 0.55: # 노이즈 적용 확률 (이전 0.4 -> 0.55)
        y = add_white_noise(y, noise_level_range=(0.0005, 0.006))
    if np.random.rand() < 0.35: # Time stretch 적용 확률 (이전 0.25 -> 0.35)
        y = apply_time_stretch(y, expected_length, rate_range=(0.85, 1.15))
    if np.random.rand() < 0.35: # Pitch shift 적용 확률 (이전 0.25 -> 0.35)
        y = apply_pitch_shift(y, sr, semitone_range=(-2, 2))
    if np.random.rand() < 0.35: # Bandpass 적용 확률 (이전 0.25 -> 0.35)
        y = apply_random_bandpass(y, sr)
    if np.random.rand() < 0.45: # Time shift 적용 확률 (이전 0.3 -> 0.45)
        y = apply_time_shift(y, sr, shift_max_ms=75)

    return np.clip(y, -1.0, 1.0)
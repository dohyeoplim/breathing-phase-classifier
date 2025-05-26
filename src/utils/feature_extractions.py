import numpy as np
import librosa
import scipy.signal

def compute_band_energy(waveform, sampling_rate, low_hz, high_hz):
    b, a = scipy.signal.butter(N=3, Wn=[low_hz / (sampling_rate / 2), high_hz / (sampling_rate / 2)], btype='band')
    filtered = scipy.signal.lfilter(b, a, waveform)
    return np.sqrt(np.mean(filtered ** 2))

def compute_spectral_entropy(spectrogram):
    spectrum = spectrogram / (np.sum(spectrogram) + 1e-10)
    return -np.sum(spectrum * np.log2(spectrum + 1e-10))

def compute_zero_crossing_features(waveform):
    zcr = librosa.feature.zero_crossing_rate(waveform)[0]
    return np.mean(zcr), np.std(zcr)

def extract_aux_features(waveform, sampling_rate):
    band_low = compute_band_energy(waveform, sampling_rate, 100, 400)
    band_mid = compute_band_energy(waveform, sampling_rate, 400, 1000)
    band_high = compute_band_energy(waveform, sampling_rate, 1000, 4000)

    zcr_mean, zcr_std = compute_zero_crossing_features(waveform)

    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate, n_mels=64)
    spectral_entropy = compute_spectral_entropy(np.mean(mel_spec, axis=1))

    return np.array([band_low, band_mid, band_high, zcr_mean, zcr_std, spectral_entropy], dtype=np.float32)
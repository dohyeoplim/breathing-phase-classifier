import numpy as np
import librosa
import scipy.signal
import scipy.stats
from scipy.signal import find_peaks
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')

SR = 16000
DURATION = 1.0
EXPECTED_LEN = int(SR * DURATION)
N_MELS = 128
N_MFCC = 40
HOP_LENGTH = 256
N_FFT = 512
FMAX = 4500
N_WORKERS = 2

DELTA_ORDER = 2
N_GAMMATONE = 64
N_LPC = 12

def pad_or_truncate(waveform: np.ndarray, target_len: int) -> np.ndarray:
    L = len(waveform)
    if L >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - L, dtype=np.float32)])

def pad_time(spec2d: np.ndarray, from_bins: int, T_fixed: int) -> np.ndarray:
    _, T_raw = spec2d.shape
    if T_raw >= T_fixed:
        return spec2d[:, :T_fixed]
    pad_w = T_fixed - T_raw
    minv = spec2d.min()
    pad_block = np.full((from_bins, pad_w), minv, dtype=np.float32)
    return np.concatenate([spec2d, pad_block], axis=1)

def pad_freq(spec2d: np.ndarray, from_bins: int, to_bins: int) -> np.ndarray:
    T_fixed = spec2d.shape[1]
    if from_bins >= to_bins:
        return spec2d[:to_bins, :]
    pad_h = to_bins - from_bins
    minv = spec2d.min()
    pad_rows = np.full((pad_h, T_fixed), minv, dtype=np.float32)
    return np.concatenate([spec2d, pad_rows], axis=0)

def extract_enhanced_scalar_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    hop = HOP_LENGTH
    features = []

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]
    features.extend([
        np.mean(rms), np.std(rms), np.max(rms), np.min(rms),
        np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)
    ])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    features.extend([
        np.mean(centroid)/(sr/2), np.std(centroid)/(sr/2), scipy.stats.skew(centroid),
        np.mean(bandwidth)/(sr/2), np.std(bandwidth)/(sr/2),
        np.mean(rolloff)/(sr/2), np.std(rolloff)/(sr/2),
        np.mean(flatness), np.std(flatness),
        np.mean(contrast), np.std(contrast)
    ])

    envelope = np.abs(scipy.signal.hilbert(y))
    envelope_mean = np.mean(envelope)
    envelope_std = np.std(envelope)
    envelope_snr = envelope_mean / (envelope_std + 1e-8)
    peaks, properties = find_peaks(envelope, height=envelope_mean, distance=sr//10)
    n_peaks = len(peaks)
    peak_heights = properties['peak_heights'] if n_peaks > 0 else [0]
    features.extend([
        envelope_mean, envelope_std, envelope_snr,
        n_peaks, np.mean(peak_heights), np.std(peak_heights) if n_peaks > 1 else 0
    ])

    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop))
    low_freq_bins = int(1000 * N_FFT / sr)
    low_freq_energy = np.sum(stft[:low_freq_bins, :] ** 2)
    total_energy = np.sum(stft ** 2)
    low_freq_ratio = low_freq_energy / (total_energy + 1e-8)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    flux = np.sqrt(np.sum(np.diff(mel_db, axis=1) ** 2, axis=0))
    features.extend([
        low_freq_ratio,
        np.mean(flux), np.std(flux), np.max(flux)
    ])

    features.extend([
        scipy.stats.skew(y),
        scipy.stats.kurtosis(y),
        np.percentile(np.abs(y), 90),
        np.percentile(np.abs(y), 10)
    ])

    autocorr = np.correlate(y, y, mode='full')[len(y)-1:]
    autocorr = autocorr / autocorr[0]
    first_min_idx = np.argmin(autocorr[:sr//20]) if len(autocorr) > sr//20 else len(autocorr)//2
    features.extend([
        autocorr[sr//100] if len(autocorr) > sr//100 else 0,
        autocorr[sr//50] if len(autocorr) > sr//50 else 0,
        first_min_idx / sr
    ])

    return np.array(features, dtype=np.float32)

def extract_lpc_features(y: np.ndarray, order: int = N_LPC) -> np.ndarray:
    pre_emphasis = 0.97
    y_emph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    frame_length = int(0.025 * SR)
    frame_shift = int(0.010 * SR)
    lpc_features = []

    for i in range(0, len(y_emph) - frame_length, frame_shift):
        frame = y_emph[i:i + frame_length] * np.hamming(frame_length)
        try:
            a = librosa.lpc(frame, order=order)
            lpc_features.append(a[1:])
        except:
            lpc_features.append(np.zeros(order))

    if not lpc_features:
        return np.zeros((order, 1), dtype=np.float32)

    return np.array(lpc_features, dtype=np.float32).T

def extract_gammatone_features(y: np.ndarray, sr: int = SR, n_filters: int = N_GAMMATONE) -> np.ndarray:
    mel_filters = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=n_filters)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    gammatone_features = np.dot(mel_filters, stft)
    return np.log1p(gammatone_features)

def extract_spectral_modulation_features(mel_db: np.ndarray) -> np.ndarray:
    return dct(dct(mel_db, axis=0, norm='ortho')[:40, :], axis=1, norm='ortho')
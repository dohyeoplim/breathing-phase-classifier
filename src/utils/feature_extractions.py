import numpy as np
import librosa
import scipy.signal

# from src.utils.normalize import normalize_feature_matrix # dataset.py에서 처리

def compute_band_energy(waveform, sampling_rate, low_hz, high_hz):
    nyquist = sampling_rate / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    
    if high >= 1.0: high = 0.9999 # Nyquist 경계 바로 아래로 설정
    if low <= 0.0: low = 0.0001  # 0 또는 음수 방지
    if low >= high: return 0.0

    try:
        b, a = scipy.signal.butter(N=5, Wn=[low, high], btype='band', analog=False)
        filtered = scipy.signal.lfilter(b, a, waveform)
        return np.sqrt(np.mean(filtered**2))
    except ValueError:
        return 0.0

def compute_spectral_entropy(spectrogram_linear_scale):
    epsilon = 1e-12 # 매우 작은 값으로 유지
    if np.sum(spectrogram_linear_scale) < epsilon or spectrogram_linear_scale.size == 0:
        return 0.0
    
    spectrum_normalized = spectrogram_linear_scale / (np.sum(spectrogram_linear_scale) + epsilon)
    spectral_entropy_val = -np.sum(spectrum_normalized * np.log2(spectrum_normalized + epsilon))
    return spectral_entropy_val if not (np.isnan(spectral_entropy_val) or np.isinf(spectral_entropy_val)) else 0.0

def compute_zero_crossing_features(waveform):
    if waveform.size == 0: return 0.0, 0.0
    zcr = librosa.feature.zero_crossing_rate(y=waveform)[0] # y 파라미터 명시
    return np.mean(zcr), np.std(zcr)

def extract_aux_features(waveform, sampling_rate, n_mels_for_entropy=128, n_fft_for_entropy=2048, hop_length_for_entropy=512):
    # 밴드 에너지 (4개)
    band_low = compute_band_energy(waveform, sampling_rate, 50, 300)
    band_mid_low = compute_band_energy(waveform, sampling_rate, 300, 700)
    band_mid_high = compute_band_energy(waveform, sampling_rate, 700, 1500)
    band_high = compute_band_energy(waveform, sampling_rate, 1500, min(7500, sampling_rate / 2.0 - 100.0)) # 상한 약간 높임

    # ZCR (2개)
    zcr_mean, zcr_std = compute_zero_crossing_features(waveform)

    # 스펙트럼 엔트로피 (1개)
    if waveform.size > n_fft_for_entropy : # 충분한 길이의 웨이브폼일 때만 계산
        mel_spec_linear = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate, 
                                                         n_mels=n_mels_for_entropy, 
                                                         n_fft=n_fft_for_entropy, 
                                                         hop_length=hop_length_for_entropy)
        spectral_entropy = compute_spectral_entropy(np.mean(mel_spec_linear, axis=1))
    else:
        spectral_entropy = 0.0
        
    # 스펙트럼 평탄도 (1개)
    spectral_flatness = librosa.feature.spectral_flatness(y=waveform)[0].mean() if waveform.size > 0 else 0.0
    
    # 스펙트럼 롤오프 (1개)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sampling_rate, roll_percent=0.90)[0].mean() if waveform.size > 0 else 0.0 # roll_percent 약간 조정

    aux_feature_values = np.array([
        band_low, band_mid_low, band_mid_high, band_high,
        zcr_mean, zcr_std, spectral_entropy,
        spectral_flatness, spectral_rolloff
    ], dtype=np.float32)
    
    aux_feature_values = np.nan_to_num(aux_feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    return aux_feature_values # 9개 특징
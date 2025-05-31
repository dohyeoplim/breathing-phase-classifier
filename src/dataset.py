import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import os
import re
import scipy.signal
import scipy.stats
import warnings

warnings.filterwarnings("ignore")

class BreathingAudioDataset(Dataset):
    def __init__(self, data_frame, data_directory, is_training=True, transform=None, sampling_rate=16000, duration_in_seconds=1):
        self.data = data_frame
        self.data_directory = data_directory
        self.is_training = is_training
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.duration_in_seconds = duration_in_seconds
        self.expected_length = sampling_rate * duration_in_seconds
        self.n_mels = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_id = row["ID"]
        file_name = re.sub(r'_[EI]_', '_', file_id) + '.wav' if self.is_training else file_id
        file_path = os.path.join(self.data_directory, file_name)

        waveform = self._load_audio(file_path)
        features = self._extract_features(waveform)

        if self.is_training:
            label = 1.0 if row["Target"] == "E" else 0.0
            return features, torch.tensor(label, dtype=torch.float32)
        else:
            return features, row["ID"]

    def _load_audio(self, file_path):
        waveform, _ = librosa.load(file_path, sr=self.sampling_rate)
        return self._normalize_audio(waveform)

    def _normalize_audio(self, waveform, target_rms=0.1):
        rms = np.sqrt(np.mean(waveform ** 2)) + 1e-8
        return waveform * (target_rms / rms)

    def _extract_features(self, y):
        sr = self.sampling_rate
        hop_length = 512  # Consistent hop length for all features

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

        temporal_features = self._extract_temporal_features(y, rms, centroid, zcr)
        spectral_features = self._extract_spectral_features(y)

        scalars = np.concatenate([temporal_features, spectral_features])

        feature_stack = np.vstack([
            mel_db_norm,
            self._normalize(rms[np.newaxis, :]),
            self._normalize(centroid[np.newaxis, :] / sr),
            self._normalize(zcr[np.newaxis, :])
        ])

        return {
            "features": torch.tensor(feature_stack, dtype=torch.float32).unsqueeze(0),
            "scalars": torch.tensor(scalars, dtype=torch.float32)
        }

    def _extract_temporal_features(self, y, rms, centroid, zcr):
        envelope = np.abs(scipy.signal.hilbert(y))
        peak_idx = np.argmax(envelope)
        peak_time_ratio = peak_idx / len(y)

        rise_time = peak_idx / self.sampling_rate
        fall_time = (len(y) - peak_idx) / self.sampling_rate

        rms_slope = np.polyfit(np.arange(len(rms)), rms, deg=1)[0]
        centroid_slope = np.polyfit(np.arange(len(centroid)), centroid, deg=1)[0]

        return np.array([
            np.mean(rms), np.std(rms), rms_slope,
            np.mean(centroid), np.std(centroid), centroid_slope,
            np.mean(zcr), np.std(zcr),
            peak_time_ratio, rise_time, fall_time,
            scipy.stats.skew(y), scipy.stats.kurtosis(y)
        ], dtype=np.float32)

    def _extract_spectral_features(self, y):
        sr = self.sampling_rate

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        freqs, psd = scipy.signal.welch(y, sr, nperseg=512)
        total_power = np.trapezoid(psd, freqs)

        bands = [(20, 500), (500, 2000), (2000, 8000)]
        band_powers = []
        for fmin, fmax in bands:
            idx = np.logical_and(freqs >= fmin, freqs < fmax)
            power = np.trapezoid(psd[idx], freqs[idx]) / total_power
            band_powers.append(power)

        return np.array([
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
            np.mean(contrast), np.std(contrast),
            *band_powers
        ], dtype=np.float32)

    def _normalize(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

def breathing_collate_fn(batch):
    features, scalars, labels_or_ids = [], [], []

    for x, y in batch:
        features.append(x["features"])
        scalars.append(x["scalars"])
        labels_or_ids.append(y)

    max_len = max(f.shape[-1] for f in features)
    features = [F.pad(f, (0, max_len - f.shape[-1])) for f in features]

    if isinstance(labels_or_ids[0], torch.Tensor):
        return torch.stack(features), torch.stack(scalars), torch.stack(labels_or_ids)
    else:
        return torch.stack(features), torch.stack(scalars), labels_or_ids
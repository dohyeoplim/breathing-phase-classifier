import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import os, re, warnings
import scipy.signal, scipy.stats

warnings.filterwarnings("ignore")

class BreathingAudioDataset(Dataset):
    def __init__(self, data_frame, data_directory, is_training=True, transform=None,
                 sampling_rate=16000, duration_in_seconds=1):
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

        y = self._load_audio(file_path)
        features = self._extract_features(y)

        if self.is_training:
            label = 1.0 if row["Target"] == "E" else 0.0
            return features, torch.tensor(label, dtype=torch.float32)
        else:
            return features, row["ID"]

    def _load_audio(self, file_path):
        y, _ = librosa.load(file_path, sr=self.sampling_rate)
        return self._normalize_audio(y)

    def _normalize_audio(self, y, target_rms=0.1):
        rms = np.sqrt(np.mean(y**2)) + 1e-8
        return y * (target_rms / rms)

    def _extract_features(self, y):
        sr = self.sampling_rate
        hop = 512

        # === Mel + energy features ===
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=hop)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = self._normalize(mel_db)

        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

        # === Scalar features ===
        scalars = np.concatenate([
            self._temporal_scalars(y, rms),
            self._spectral_scalars(y, mel)
        ]).astype(np.float32)

        feature_stack = np.vstack([
            mel_norm,
            self._normalize(rms[np.newaxis, :]),
            self._normalize(zcr[np.newaxis, :])
        ])

        return {
            "features": torch.tensor(feature_stack, dtype=torch.float32).unsqueeze(0),
            "scalars": torch.tensor(scalars, dtype=torch.float32)
        }

    def _temporal_scalars(self, y, rms):
        envelope = np.abs(scipy.signal.hilbert(y))
        peak_idx = np.argmax(envelope)
        peak_time_ratio = peak_idx / len(y)
        rise_time = peak_idx / self.sampling_rate
        fall_time = (len(y) - peak_idx) / self.sampling_rate
        rms_slope = np.polyfit(np.arange(len(rms)), rms, 1)[0]

        return np.array([
            np.mean(rms), np.std(rms), rms_slope,
            peak_time_ratio, rise_time, fall_time,
            scipy.stats.kurtosis(y)
        ])

    def _spectral_scalars(self, y, mel):
        sr = self.sampling_rate
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        entropy = np.mean(-np.sum((mel / (mel.sum(axis=0, keepdims=True) + 1e-8)) *
                                  np.log1p(mel + 1e-8), axis=0))
        flux = self._spectral_flux(mel)

        freqs, psd = scipy.signal.welch(y, sr, nperseg=512)
        total_power = np.trapezoid(psd, freqs) + 1e-8
        low = np.trapezoid(psd[(freqs >= 20) & (freqs < 500)], freqs[(freqs >= 20) & (freqs < 500)]) / total_power
        high = np.trapezoid(psd[(freqs >= 2000) & (freqs < 8000)], freqs[(freqs >= 2000) & (freqs < 8000)]) / total_power

        return np.array([
            np.mean(contrast),
            entropy, flux, low, high
        ])

    def _spectral_flux(self, mel):
        diff = np.diff(mel, axis=1)
        return np.mean(np.sqrt((diff ** 2).sum(axis=0)))

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
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
        self.n_mels = 64

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
        hop_length = 512

        # Spectrogram (simplified)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db_norm = self._normalize(mel_db)

        # Scalars: only high-importance features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]

        # Spectral entropy
        mel_norm = mel / (mel.sum(axis=0, keepdims=True) + 1e-8)
        entropy_per_frame = -np.sum(mel_norm * np.log1p(mel_norm), axis=0)
        entropy_mel = np.mean(entropy_per_frame)

        # PSD band energy (low only)
        freqs, psd = scipy.signal.welch(y, sr, nperseg=512)
        idx_low = np.logical_and(freqs >= 20, freqs < 500)
        psd_low = np.trapezoid(psd[idx_low], freqs[idx_low])

        scalars = np.array([
            np.mean(rms),
            np.std(rms),
            np.mean(centroid),
            np.std(centroid),
            np.std(zcr),
            np.mean(flatness),
            entropy_mel,
            psd_low
        ], dtype=np.float32)

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
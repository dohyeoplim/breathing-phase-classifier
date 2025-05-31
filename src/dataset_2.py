import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import os
import re
import scipy.signal
import warnings

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

        waveform = self._load_audio(file_path)
        if self.is_training:
            waveform = self._augment_audio(waveform)

        features = self._extract_features(waveform)

        if self.is_training:
            label = 1.0 if row["Target"] == "E" else 0.0
            return features, torch.tensor(label, dtype=torch.float32)
        else:
            return features, row["ID"]

    def _load_audio(self, file_path):
        y, _ = librosa.load(file_path, sr=self.sampling_rate)
        y = self._bandpass_filter(y, low=100, high=5000)
        return self._normalize_audio(y)

    def _bandpass_filter(self, y, low=100, high=5000):
        sos = scipy.signal.butter(4, [low, high], btype='bandpass', fs=self.sampling_rate, output='sos')
        return scipy.signal.sosfilt(sos, y)

    def _normalize_audio(self, y, target_rms=0.1):
        rms = np.sqrt(np.mean(y ** 2)) + 1e-8
        return y * (target_rms / rms)

    def _augment_audio(self, y):
        if np.random.rand() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=self.sampling_rate, n_steps=np.random.uniform(-1, 1))
        if np.random.rand() < 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
        return y

    def _extract_features(self, y):
        sr = self.sampling_rate
        hop = 512

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=hop)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self._normalize(mel_db)

        # Deltas (1st and 2nd order)
        delta = librosa.feature.delta(mel_db)
        delta2 = librosa.feature.delta(mel_db, order=2)
        mel_stack = np.stack([mel_db, delta, delta2], axis=0)  # Shape: [3, H, T]

        # Frame-level
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]

        # Spectral entropy
        mel_norm = mel / (mel.sum(axis=0, keepdims=True) + 1e-8)
        entropy_mel = np.mean(-np.sum(mel_norm * np.log1p(mel_norm), axis=0))

        # Spectral flux
        flux = self._spectral_flux(mel)

        # PSD 2.52–4kHz (Stage 3 from paper)
        freqs, psd = scipy.signal.welch(y, sr, nperseg=512)
        band_mask = np.logical_and(freqs >= 2520, freqs <= 4000)
        psd_band = 10 * np.log10(np.mean(psd[band_mask]) + 1e-8)

        # ZCR delta (tail - head)
        zcr_diff = np.mean(zcr[-10:]) - np.mean(zcr[:10])

        # Centroid delta (tail - head)
        centroid_diff = np.mean(centroid[-10:]) - np.mean(centroid[:10])

        # Final scalar features (10)
        scalars = np.array([
            np.mean(rms), np.std(rms),
            np.mean(centroid), np.std(centroid), centroid_diff,
            np.std(zcr), zcr_diff,
            np.mean(flatness),
            entropy_mel,
            psd_band
        ], dtype=np.float32)

        return {
            "features": torch.tensor(mel_stack.copy(), dtype=torch.float32),
            "scalars": torch.tensor(scalars.copy(), dtype=torch.float32)
        }

    def _spectral_flux(self, mel):
        diff = np.diff(mel, axis=1)
        return np.mean(np.sqrt((diff**2).sum(axis=0)))

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
    features = torch.stack(features)

    if isinstance(labels_or_ids[0], torch.Tensor):
        return features, torch.stack(scalars), torch.stack(labels_or_ids)
    else:
        return features, torch.stack(scalars), labels_or_ids
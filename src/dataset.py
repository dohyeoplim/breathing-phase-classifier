import os
import re
import torch
import numpy as np
import librosa
import warnings
from torch.utils.data import Dataset
import torch.nn.functional as F
import scipy.signal

warnings.filterwarnings('ignore')


class BreathingAudioDataset(Dataset):
    def __init__(
            self,
            data_frame,
            data_directory,
            is_training=True,
            transform=None,
            sampling_rate=16000,
            duration_in_seconds=1,
            feature_type='simple'  # 'simple', 'medium', 'complex'
    ):
        self.data = data_frame
        self.data_directory = data_directory
        self.is_training = is_training
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.duration_in_seconds = duration_in_seconds
        self.expected_length = sampling_rate * duration_in_seconds
        self.feature_type = feature_type
        self._setup_feature_config()

    def _setup_feature_config(self):
        if self.feature_type == 'simple':
            self.n_mels = 64
            self.n_mfcc = 13
        elif self.feature_type == 'medium':
            self.n_mels = 128
            self.n_mfcc = 20
        else:
            self.n_mels = 128
            self.n_mfcc = 20  # still useful

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_id = row["ID"]
        file_name = re.sub(r'_[EI]_', '_', file_id) + '.wav' if self.is_training else file_id
        file_path = os.path.join(self.data_directory, file_name)

        waveform = self._load_audio(file_path)
        features = self._extract_features(waveform)

        if self.transform:
            features = self.transform(features)

        if self.is_training:
            label = 1.0 if row["Target"] == "E" else 0.0
            return features, torch.tensor(label, dtype=torch.float32)
        else:
            return features, row["ID"]

    def _load_audio(self, file_path):
        waveform, orig_sr = librosa.load(file_path, sr=None)
        if orig_sr != self.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=self.sampling_rate)

        waveform = self._normalize_audio(waveform)

        if self.is_training:
            waveform = self._augment_audio(waveform)

        return waveform

    def _normalize_audio(self, waveform, target_rms=0.1):
        rms = np.sqrt(np.mean(waveform**2)) + 1e-8
        return waveform * (target_rms / rms)

    def _augment_audio(self, waveform):
        if np.random.rand() < 0.8:
            waveform *= np.random.uniform(0.8, 1.2)
        if np.random.rand() < 0.4:
            noise = np.random.normal(0, 0.002, waveform.shape)
            waveform += noise
        return np.clip(waveform, -1.0, 1.0)

    def _extract_features(self, waveform):
        if self.feature_type == 'simple':
            return self._extract_simple_features(waveform)
        elif self.feature_type == 'medium':
            return self._extract_medium_features(waveform)
        else:
            return self._extract_complex_features(waveform)

    def _extract_simple_features(self, waveform):
        mel = librosa.feature.melspectrogram(y=waveform, sr=self.sampling_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=self.n_mfcc)

        mel_db = self._normalize(mel_db)
        mfcc = self._normalize(mfcc)

        features = np.vstack([mel_db, mfcc])
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _extract_medium_features(self, waveform):
        mel = librosa.feature.melspectrogram(y=waveform, sr=self.sampling_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=self.n_mfcc)

        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.sampling_rate)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]

        mel_db = self._normalize(mel_db)
        mfcc = self._normalize(mfcc)
        spectral_centroid = self._normalize(spectral_centroid)
        zcr = self._normalize(zcr)

        features = np.vstack([
            mel_db, mfcc,
            spectral_centroid[np.newaxis, :],
            zcr[np.newaxis, :]
        ])
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _extract_complex_features(self, waveform):
        base = self._extract_medium_features(waveform)
        breathing_feats = self._extract_fft_psd_features(waveform)

        T = base.shape[-1]
        breathing_feats = breathing_feats.unsqueeze(0).expand(-1, T)
        return torch.cat([base, breathing_feats.unsqueeze(0)], dim=1)

    def _extract_fft_psd_features(self, waveform):
        # FFT magnitude spectrum
        fft = np.abs(np.fft.rfft(waveform))[:128]
        fft = self._normalize(fft)

        # Power Spectral Density
        freqs, psd = scipy.signal.welch(waveform, fs=self.sampling_rate)
        psd = psd[:128]
        psd = self._normalize(psd)

        combined = np.concatenate([fft, psd])
        return torch.tensor(combined, dtype=torch.float32)

    def _normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)


def breathing_collate_fn(batch):
    if isinstance(batch[0][1], torch.Tensor):  # Training mode
        features, labels = zip(*batch)
        max_len = max(f.shape[-1] for f in features)
        features = [F.pad(f, (0, max_len - f.shape[-1])) for f in features]
        return torch.stack(features), torch.stack(labels)
    else:  # Inference mode
        features, ids = zip(*batch)
        max_len = max(f.shape[-1] for f in features)
        features = [F.pad(f, (0, max_len - f.shape[-1])) for f in features]
        return torch.stack(features), ids
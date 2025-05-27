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
            duration_in_seconds=1
    ):
        self.data = data_frame
        self.data_directory = data_directory
        self.is_training = is_training
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.duration_in_seconds = duration_in_seconds
        self.expected_length = sampling_rate * duration_in_seconds
        self.n_mels = 128
        self.n_mfcc = 20
        self.bands = [(30, 723), (723, 883), (883, 1030)] # eda_2.py에서 나옴

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
        if len(waveform) < self.expected_length:
            waveform = np.pad(waveform, (0, self.expected_length - len(waveform)))
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
            waveform += np.random.normal(0, 0.002, waveform.shape)
        return np.clip(waveform, -1.0, 1.0)

    def _extract_features(self, waveform):
        mel = librosa.feature.melspectrogram(y=waveform, sr=self.sampling_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=self.sampling_rate, n_mfcc=self.n_mfcc)
        centroid = self._normalize(librosa.feature.spectral_centroid(y=waveform, sr=self.sampling_rate)[0])
        zcr = self._normalize(librosa.feature.zero_crossing_rate(y=waveform)[0])

        return torch.tensor(np.vstack([
            self._normalize(mel_db),
            self._normalize(mfcc),
            centroid[np.newaxis, :],
            zcr[np.newaxis, :]
        ]), dtype=torch.float32).unsqueeze(0)

    def _normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)

def breathing_collate_fn(batch):
    if isinstance(batch[0][1], torch.Tensor):
        features, labels = zip(*batch)
        max_len = max(f.shape[-1] for f in features)
        features = [F.pad(f, (0, max_len - f.shape[-1])) for f in features]
        return torch.stack(features), torch.stack(labels)
    else:
        features, ids = zip(*batch)
        max_len = max(f.shape[-1] for f in features)
        features = [F.pad(f, (0, max_len - f.shape[-1])) for f in features]
        return torch.stack(features), ids

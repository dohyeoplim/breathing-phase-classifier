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
            feature_type='simple'
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

        self.bands = [(30, 723), (723, 883), (883, 1030)] # eda_2.py에서 나옴

    def _setup_feature_config(self):
        self.n_mels = 64 if self.feature_type == 'simple' else 128
        self.n_mfcc = 13 if self.feature_type == 'simple' else 20

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
        if self.is_training and self.feature_type != 'complex':
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
        return torch.tensor(np.vstack([self._normalize(mel_db), self._normalize(mfcc)]), dtype=torch.float32).unsqueeze(0)

    def _extract_medium_features(self, waveform):
        mel = librosa.feature.melspectrogram(y=waveform, sr=self.sampling_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=self.n_mfcc)
        centroid = self._normalize(librosa.feature.spectral_centroid(y=waveform, sr=self.sampling_rate)[0])
        zcr = self._normalize(librosa.feature.zero_crossing_rate(y=waveform)[0])
        return torch.tensor(np.vstack([
            self._normalize(mel_db),
            self._normalize(mfcc),
            centroid[np.newaxis, :],
            zcr[np.newaxis, :]
        ]), dtype=torch.float32).unsqueeze(0)

    def _extract_complex_features(self, waveform):
        base = self._extract_medium_features(waveform)

        envelope = np.abs(scipy.signal.hilbert(waveform))
        smooth_env = scipy.signal.savgol_filter(envelope, 301, polyorder=3)
        envelope_peak = np.max(smooth_env)
        envelope_slope = np.max(np.gradient(smooth_env))
        active_ratio = np.mean(np.abs(waveform) > 0.02)

        h, p = librosa.effects.hpss(waveform)
        harmonic_energy = np.mean(h**2)
        percussive_energy = np.mean(p**2)

        # SNR estimate
        signal_power = np.mean(waveform ** 2)
        noise_power = np.var(waveform - smooth_env)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))

        band_energies = []
        for low, high in self.bands:
            sos = scipy.signal.butter(3, [low, high], btype='band', fs=self.sampling_rate, output='sos')
            filtered = scipy.signal.sosfilt(sos, waveform)
            band_energies.append(np.sqrt(np.mean(filtered**2)))

        scalar_feats = torch.tensor([
            np.sqrt(np.mean(waveform**2)),
            envelope_peak,
            envelope_slope,
            active_ratio,
            *band_energies,
            harmonic_energy,
            percussive_energy,
            snr
        ], dtype=torch.float32)

        # Add temporal envelope feature
        frame_env = self._normalize(librosa.feature.rms(y=waveform, frame_length=1024, hop_length=512)[0])
        frame_env = torch.tensor(frame_env, dtype=torch.float32).unsqueeze(0)

        base = torch.cat([base, frame_env.unsqueeze(0)], dim=1)
        T = base.shape[-1]
        scalar_feats = scalar_feats.unsqueeze(1).expand(-1, T).unsqueeze(0)  # [1, 10, T]
        return torch.cat([base, scalar_feats], dim=1)

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
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa, os, re, warnings
import scipy.signal, scipy.stats
from scipy.ndimage import gaussian_filter1d

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
        rms = np.sqrt(np.mean(y ** 2)) + 1e-8
        return y * (target_rms / rms)

    def _extract_features(self, y):
        sr = self.sampling_rate
        hop = 128

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=hop)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_smooth = gaussian_filter1d(mel_db, sigma=1.2, axis=1)
        mel_norm = self._normalize(mel_smooth)

        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

        scalars = np.concatenate([
            self._temporal_scalars(y, rms, zcr),
            self._spectral_scalars(y, mel)
        ]).astype(np.float32)

        scalars = np.clip(scalars, -5, 5)

        feature_stack = np.vstack([
            mel_norm,
            self._normalize(rms[np.newaxis, :]),
            self._normalize(centroid[np.newaxis, :] / sr),
            self._normalize(zcr[np.newaxis, :])
        ])

        return {
            "features": torch.tensor(feature_stack, dtype=torch.float32).unsqueeze(0),
            "scalars": torch.tensor(scalars, dtype=torch.float32)
        }

    def _temporal_scalars(self, y, rms, zcr):
        rms_mean = np.mean(rms)
        rms_range = np.max(rms) - np.min(rms)
        zcr_std = np.std(zcr)

        envelope = np.abs(scipy.signal.hilbert(y))
        peak_idx = np.argmax(envelope)
        env_mean = np.mean(envelope)
        env_std = np.std(envelope)
        env_range = np.max(envelope) - np.min(envelope)
        rise_time = peak_idx / self.sampling_rate
        # log_rise = np.log1p(rise_time)

        env_snr = env_mean / (env_std + 1e-8)
        # wheeze_ratio = np.mean(envelope > (env_mean + 0.5 * env_std))

        return np.array([
            # rms_mean,
            rms_range,
            rise_time,
            # log_rise,
            # zcr_std,
            # env_std,
            env_snr,
            env_range,
            # wheeze_ratio,
        ])

    def _spectral_scalars(self, y, mel):
        entropy = np.mean(-np.sum((mel / (mel.sum(axis=0, keepdims=True) + 1e-8)) *
                                  np.log1p(mel + 1e-8), axis=0))
        flux = self._spectral_flux(mel)

        freqs, psd = scipy.signal.welch(y, self.sampling_rate, nperseg=512)
        total_power = np.trapezoid(psd, freqs) + 1e-8

        low_mask = (freqs >= 100) & (freqs < 500)
        mid_mask = (freqs >= 500) & (freqs < 1000)
        high_mask = (freqs >= 2520) & (freqs <= 4000)


        low_power = np.trapezoid(psd[low_mask], freqs[low_mask]) + 1e-8
        high_power = np.trapezoid(psd[high_mask], freqs[high_mask]) + 1e-8
        ratio_high_to_low = high_power / low_power
        log_diff = np.log10(high_power + 1e-8) - np.log10(low_power + 1e-8)


        high_band_psd = psd[high_mask]
        range_high_band = np.max(high_band_psd) - np.min(high_band_psd)
        std_high_band = np.std(high_band_psd)
        # low_mask = (freqs >= 20) & (freqs < 500)
        # low_psd = psd[low_mask]
        # low_power = np.trapezoid(low_psd, freqs[low_mask]) / total_power
        # range_low = np.max(low_psd) - np.min(low_psd)

        # psd_geomean = np.exp(np.mean(np.log(psd + 1e-8)))

        return np.array([
            # entropy,
            flux**2,
            # mean_band,
            std_high_band,
            range_high_band,
            ratio_high_to_low,
            # range_low,
            # psd_geomean,
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
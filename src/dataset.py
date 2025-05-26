import os
import re
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

from src.utils.normalize import normalize_to_target_root_mean_square, normalize_feature_matrix
from src.utils.audio_augment import augment_signal


def reconstruct_audio_filename(identifier: str) -> str:
    return re.sub(r'_(E|I)_', '_', identifier) + '.wav'


class BreathingAudioDataset(Dataset):
    def __init__(
            self,
            data_frame,
            data_directory,
            is_training: bool = True,
            transform=None,
            sampling_rate: int = 16000,
            duration_in_seconds: int = 1,
            feature_type: str = "mel_mfcc",  # "mel_mfcc" or "raw"
            feature_mode: str = "default",   # for future use (ZCR, CWT ...)
    ):
        self.data = data_frame
        self.data_directory = data_directory
        self.is_training = is_training
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.duration_in_seconds = duration_in_seconds
        self.expected_length = sampling_rate * duration_in_seconds
        self.feature_type = feature_type
        self.feature_mode = feature_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_name = reconstruct_audio_filename(row["ID"]) if self.is_training else row["ID"]
        file_path = os.path.join(self.data_directory, file_name)

        waveform = self.load_audio(file_path)

        input_tensor = self.extract_features(waveform)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        if self.is_training:
            label = 1 if row["Target"] == "E" else 0
            return input_tensor, torch.tensor(label, dtype=torch.float32)
        else:
            return input_tensor, row["ID"]

    def load_audio(self, file_path: str) -> np.ndarray:
        waveform, _ = librosa.load(file_path, sr=self.sampling_rate)
        waveform = normalize_to_target_root_mean_square(waveform)

        if self.is_training:
            waveform = augment_signal(waveform, self.sampling_rate)

        return waveform

    def extract_features(self, waveform: np.ndarray) -> torch.Tensor:
        if self.feature_type == "raw":
            padded = librosa.util.fix_length(waveform, self.expected_length)
            reshaped = padded[:4000].reshape(1, 250, 16)
            return torch.tensor(reshaped, dtype=torch.float32)

        elif self.feature_type == "mel_mfcc":
            mel = librosa.feature.melspectrogram(y=waveform, sr=self.sampling_rate, n_mels=64)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = normalize_feature_matrix(mel_db)

            mfcc = librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=20)
            mfcc_norm = normalize_feature_matrix(mfcc)

            mel_tensor = torch.tensor(mel_db).float()
            mfcc_tensor = torch.tensor(mfcc_norm).float()
            combined = torch.cat([mel_tensor, mfcc_tensor], dim=0).unsqueeze(0)
            return combined

        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
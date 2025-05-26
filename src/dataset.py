import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import re

from src.utils.normalize import normalize_to_target_root_mean_square, normalize_feature_matrix
from src.utils.augment import apply_random_volume_gain

def reconstruct_audio_filename(identifier: str) -> str:
    return re.sub(r'_(E|I)_', '_', identifier) + '.wav'

class BreathingAudioDataset(Dataset):
    def __init__(self, data_frame, data_directory, is_training=True, transform=None, sampling_rate=16000, duration_in_seconds=1):
        self.data = data_frame
        self.data_directory = data_directory
        self.is_training = is_training
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.expected_length = sampling_rate * duration_in_seconds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_name = reconstruct_audio_filename(row['ID']) if self.is_training else row['ID']
        file_path = os.path.join(self.data_directory, file_name)
        waveform, actual_sampling_rate = librosa.load(file_path, sr=self.sampling_rate)

        waveform = normalize_to_target_root_mean_square(waveform)

        if self.is_training:
            waveform = apply_random_volume_gain(waveform)

        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=actual_sampling_rate, n_mels=64)
        mel_spectrogram_decibel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_decibel = normalize_feature_matrix(mel_spectrogram_decibel)

        mel_frequency_cepstral_coefficients = librosa.feature.mfcc(y=waveform, sr=actual_sampling_rate, n_mfcc=20)
        mel_frequency_cepstral_coefficients_normalized = normalize_feature_matrix(mel_frequency_cepstral_coefficients)

        mel_tensor = torch.tensor(mel_spectrogram_decibel).float()
        mfcc_tensor = torch.tensor(mel_frequency_cepstral_coefficients_normalized).float()

        combined_input_tensor = torch.cat([mel_tensor, mfcc_tensor], dim=0).unsqueeze(0)  # Shape: [1, Channels, Time]

        if self.is_training:
            label = 1 if row['Target'] == 'E' else 0
            return combined_input_tensor, torch.tensor(label, dtype=torch.float32)
        else:
            return combined_input_tensor, row['ID']
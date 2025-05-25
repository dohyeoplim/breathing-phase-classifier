import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import re

def reconstruct_filename(name: str) -> str:
    return re.sub(r'_(E|I)_', '_', name) + '.wav'

class BreathingDataset(Dataset):
    def __init__(self, csv, data_dir, is_train=True, transform=None, sr=16000, duration=3):
        self.data = csv
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        self.sr = sr
        self.duration = duration
        self.length = sr * duration

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.is_train:
            file_name = reconstruct_filename(row['ID'])
        else:
            file_name = row['ID']

        file_path = os.path.join(self.data_dir, file_name)
        y, sr = librosa.load(file_path, sr=self.sr)

        if len(y) < self.length:
            y = np.pad(y, (0, self.length - len(y)))
        else:
            y = y[:self.length]

        if self.transform:
            y = self.transform(y)

        # Mel + MFCC
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc_norm = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        mel_tensor = torch.tensor(mel_db).float()
        mfcc_tensor = torch.tensor(mfcc_norm).float()

        x = torch.cat([mel_tensor, mfcc_tensor], dim=0).unsqueeze(0)

        if self.is_train:
            label = 1 if row['Target'] == 'E' else 0
            return x, torch.tensor(label, dtype=torch.float32)
        else:
            return x, row['ID']

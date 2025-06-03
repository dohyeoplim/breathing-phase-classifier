# src/dataset.py

import os
import re
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

from src.utils.normalize import normalize_to_target_root_mean_square, normalize_feature_matrix
from src.utils.audio_augment import augment_signal
from src.utils.feature_extractions import extract_aux_features


def reconstruct_audio_filename(identifier: str) -> str:
    return re.sub(r'_(E|I)_', '_', identifier) + '.wav'


class BreathingAudioDataset(Dataset):
    def __init__(
            self,
            data_frame,
            data_directory,
            is_training: bool = True,
            sampling_rate: int = 16000,
            duration_in_seconds: int = 2,
            n_mels: int = 128,
            n_mfcc: int = 40,
            hop_length: int = 512,
            n_fft: int = 2048,
            # aux_features 내 spectral_entropy 계산용 파라미터
            n_mels_for_entropy: int = 128, 
            n_fft_for_entropy: int = 2048, 
            hop_length_for_entropy: int = 512
    ):
        self.data = data_frame
        self.data_directory = data_directory
        self.is_training = is_training
        self.sampling_rate = sampling_rate
        self.duration_in_seconds = duration_in_seconds
        self.expected_length = sampling_rate * duration_in_seconds
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels_for_entropy = n_mels_for_entropy
        self.n_fft_for_entropy = n_fft_for_entropy
        self.hop_length_for_entropy = hop_length_for_entropy


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_name = reconstruct_audio_filename(row["ID"]) if self.is_training else row["ID"]
        file_path = os.path.join(self.data_directory, file_name)

        waveform = self.load_audio(file_path)
        
        # 웨이브폼 최종 NaN/Inf 및 길이 확인
        if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)) or waveform.size != self.expected_length:
            # print(f"Warning: Problematic waveform for {file_name} (NaN/Inf/Incorrect Length after load/augment). Using zeros.")
            waveform = np.zeros(self.expected_length, dtype=np.float32)

        input_tensor = self.extract_features(waveform)

        # 최종 텐서 NaN/Inf 확인
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            # print(f"Warning: NaN/Inf in final input_tensor for {file_name}, replacing with zeros.")
            num_features = self.n_mels + self.n_mfcc + 9 # Aux features 9개
            num_frames = 1 + (self.expected_length - self.n_fft) // self.hop_length
            if num_frames <=0 : num_frames = 1 
            input_tensor = torch.zeros((1, num_features, num_frames), dtype=torch.float32)

        if self.is_training:
            label = 1 if row["Target"] == "E" else 0
            return input_tensor, torch.tensor(label, dtype=torch.float32)
        else:
            return input_tensor, row["ID"]

    def load_audio(self, file_path: str) -> np.ndarray:
        try:
            waveform, _ = librosa.load(file_path, sr=self.sampling_rate, duration=None) # 전체 로드 후 길이 조절
        except Exception as e:
            # print(f"Error loading audio {file_path}: {e}. Returning zeros.")
            return np.zeros(self.expected_length, dtype=np.float32)

        # 1. RMS 정규화
        waveform = normalize_to_target_root_mean_square(waveform)
        
        # 2. 길이 맞추기 (증강 전)
        if len(waveform) < self.expected_length:
            waveform = np.pad(waveform, (0, self.expected_length - len(waveform)), mode='constant')
        elif len(waveform) > self.expected_length:
            waveform = waveform[:self.expected_length]

        # 3. 데이터 증강 (학습 시에만)
        if self.is_training:
            waveform = augment_signal(waveform, self.sampling_rate, self.expected_length)
        
        # 4. 최종 길이 보장 (증강 후 길이가 변경될 수 있으므로 다시 한번)
        #    augment_signal 내부에서도 길이를 맞추려고 시도하지만, 여기서 최종적으로 보장.
        if len(waveform) != self.expected_length:
            if len(waveform) < self.expected_length:
                waveform = np.pad(waveform, (0, self.expected_length - len(waveform)), mode='constant')
            elif len(waveform) > self.expected_length:
                waveform = waveform[:self.expected_length]
            
        return waveform

    def extract_features(self, waveform: np.ndarray) -> torch.Tensor:
        # 1. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sampling_rate, 
                                                  n_mels=self.n_mels, n_fft=self.n_fft, 
                                                  hop_length=self.hop_length, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = np.nan_to_num(mel_db, nan=-80.0, posinf=0.0, neginf=-80.0) # NaN은 매우 작은 dB값으로, Inf는 적절한 값으로
        mel_db_norm = normalize_feature_matrix(mel_db) # 각 Mel 스펙트로그램에 대해 정규화

        # 2. MFCC
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=self.n_mfcc,
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0) # NaN/Inf 처리
        mfcc_norm = normalize_feature_matrix(mfcc) # 각 MFCC 행렬에 대해 정규화

        # 3. Auxiliary Features
        aux_features_vec = extract_aux_features(waveform, self.sampling_rate,
                                                n_mels_for_entropy=self.n_mels_for_entropy,
                                                n_fft_for_entropy=self.n_fft_for_entropy,
                                                hop_length_for_entropy=self.hop_length_for_entropy) # (9,)
        # aux_features는 다양한 스케일을 가질 수 있으므로, 전체 벡터에 대해 정규화
        # NaN/Inf는 extract_aux_features 내부에서 이미 0으로 처리됨
        aux_features_norm_vec = normalize_feature_matrix(aux_features_vec) # (9,)

        # 텐서로 변환
        mel_tensor = torch.tensor(mel_db_norm, dtype=torch.float32)
        mfcc_tensor = torch.tensor(mfcc_norm, dtype=torch.float32)
        
        num_frames = mel_tensor.shape[1] # 시간 프레임 수
        aux_expanded = torch.tensor(aux_features_norm_vec, dtype=torch.float32).unsqueeze(1).expand(-1, num_frames)

        full_tensor = torch.cat([mel_tensor, mfcc_tensor, aux_expanded], dim=0).unsqueeze(0)
        
        return full_tensor
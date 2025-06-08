import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class DS(Dataset):
    EXCLUDED_KEYS = {'scalars', 'sr', 'hop_length', 'n_fft'}

    def __init__(self, data_frame: pd.DataFrame, feature_dir: str, is_training: bool):
        self.df = data_frame.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.is_training = is_training

        self._detect_features()

    def _detect_features(self):
        if len(self.df) == 0:
            raise ValueError

        first_id = self.df.iloc[0]["ID"]
        npz_path = os.path.join(self.feature_dir, first_id + ".npz")

        with np.load(npz_path) as data:
            self.feature_names = [k for k in data.keys() if k not in self.EXCLUDED_KEYS]
            self.feature_names.sort()

            self.n_features = len(self.feature_names)
            self.scalar_dim = data['scalars'].shape[0]

        print(f"#Features: {self.n_features} - {', '.join(self.feature_names)}")
        print(f"#Scalars: {self.scalar_dim}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row["ID"]
        npz_path = os.path.join(self.feature_dir, file_id + ".npz")

        data = np.load(npz_path)

        features_list = []
        for feat_name in self.feature_names:
            features_list.append(data[feat_name])

        features = np.stack(features_list, axis=0).astype(np.float32)
        features = torch.from_numpy(features)

        scalars = torch.from_numpy(data['scalars'].astype(np.float32))

        if self.is_training:
            label = 1.0 if row["Target"] == "E" else 0.0
            return features, scalars, torch.tensor(label, dtype=torch.float32)
        else:
            return features, scalars, file_id

def collate_fn(batch): # 디멘션 에러나면 이거 꼭 확인하ㄱㅣ..
    feats, scals, labs_or_ids = [], [], []
    for f, s, y in batch:
        feats.append(f)
        scals.append(s)
        labs_or_ids.append(y)

    features = torch.stack(feats, dim=0)
    scalars  = torch.stack(scals, dim=0) # 이거 개수대로 모델에 num scalar features 지정하기

    if isinstance(labs_or_ids[0], torch.Tensor):
        labels = torch.stack(labs_or_ids, dim=0)
        return features, scalars, labels
    else:
        return features, scalars, labs_or_ids
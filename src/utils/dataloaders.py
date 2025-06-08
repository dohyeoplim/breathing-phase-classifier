import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import DS, collate_fn

def prepare_dataloaders(train_csv_path: str, test_csv_path: str, precomputed_dir: str, batch_size: int = 256, num_workers: int = 8, prefetch: int = 2):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_df_split, val_df_split = train_test_split(train_df, test_size=0.20, shuffle=True, random_state=42)

    train_dataset = DS(data_frame=train_df_split, feature_dir=precomputed_dir, is_training=True)
    val_dataset = DS(data_frame=val_df_split, feature_dir=precomputed_dir, is_training=True)
    test_dataset = DS(data_frame=test_df, feature_dir=precomputed_dir, is_training=False)

    batch_size = batch_size
    num_workers = num_workers
    prefetch = prefetch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
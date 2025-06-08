import torch
import torch.nn as nn
import numpy as np

def cutmix_data(features, labels, alpha=1.0, device='cuda'):
    batch_size = features.size(0)
    indices = torch.randperm(batch_size).to(device)

    lam = np.random.beta(alpha, alpha)

    W = features.size(3)
    H = features.size(2)

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    features_mixed = features.clone()
    features_mixed[:, :, bby1:bby2, bbx1:bbx2] = features[indices, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    labels_mixed = lam * labels + (1 - lam) * labels[indices]

    return features_mixed, labels_mixed, indices, lam


def mixup_data(features, labels, alpha=1.0, device='cuda'):
    batch_size = features.size(0)
    indices = torch.randperm(batch_size).to(device)

    lam = np.random.beta(alpha, alpha)

    mixed_features = lam * features + (1 - lam) * features[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]

    return mixed_features, mixed_labels, indices, lam
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.model import VGG, CNN8
from src.utils.display import print_info, print_success

def load_model(ckpt_path: str, arch: str, num_scalar_features: int, device: torch.device):
    if arch == 'vgg':
        model = VGG(num_scalar_features=num_scalar_features).to(device)
    elif arch == 'cnn8':
        model = CNN8(num_scalar_features=num_scalar_features).to(device)
    else:
        raise ValueError(f"이게머임? {arch}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    return model

def average_ensemble(ckpt_paths: list[str], archs: list[str], test_loader: DataLoader, device: torch.device, num_scalar_features: int):
    assert len(ckpt_paths) == len(archs), "제대로 하자."

    models = []
    for idx, (path, arch) in enumerate(zip(ckpt_paths, archs), 1):
        m = load_model(path, arch, num_scalar_features=num_scalar_features, device=device)
        models.append(m)
        print_info(f"모델: {idx}/{len(ckpt_paths)} @ '{path}'")

    all_ids = []
    per_batch_probs = []

    with torch.no_grad():
        for batch_idx, (feats, scals, ids) in enumerate(test_loader, 1):
            feats = feats.to(device, non_blocking=True)
            scals = scals.to(device, non_blocking=True)

            logits_stack = torch.stack([m(feats, scals).view(-1) for m in models], dim=0)
            probs = torch.sigmoid(logits_stack)
            avg_probs = probs.mean(dim=0)

            per_batch_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    final_probs = np.concatenate(per_batch_probs, axis=0)
    print_success("완료~")
    return all_ids, final_probs


def weighted_ensemble(ckpt_paths: list[str], archs: list[str], test_loader: torch.utils.data.DataLoader, device: torch.device, num_scalar_features: int, val_scores: list[float], use_softmax_weights: bool = True):
    assert len(ckpt_paths) == len(archs) == len(val_scores), "제대로하자"

    weights = torch.tensor(val_scores, dtype=torch.float32)
    weights = torch.softmax(weights, dim=0) if use_softmax_weights else weights / weights.sum()

    models = []
    for idx, (path, arch) in enumerate(zip(ckpt_paths, archs), 1):
        model = load_model(path, arch, num_scalar_features=num_scalar_features, device=device)
        models.append(model)

    all_ids = []
    all_probs = []

    with torch.no_grad():
        for feats, scalars, ids in test_loader:
            feats, scalars = feats.to(device), scalars.to(device)

            logits_stack = torch.stack([model(feats, scalars).view(-1) for model in models])
            probs_stack = torch.sigmoid(logits_stack)

            weighted_avg = (weights[:, None].to(device) * probs_stack).sum(dim=0)
            all_probs.append(weighted_avg.cpu().numpy())
            all_ids.extend(ids)

    return all_ids, np.concatenate(all_probs)
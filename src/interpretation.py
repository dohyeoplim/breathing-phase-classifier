import torch
import numpy as np
from src.utils.display import print_start

def permutation_feature_importance(model, data_loader, device, feature_names):
    print_start("Permutation Feature Importance")
    model.eval()

    # Step 1: baseline accuracy
    def evaluate(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for feats, scalars, labels in loader:
                feats = feats.to(device)
                scalars = scalars.to(device)
                labels = labels.to(device)
                probs = torch.sigmoid(model(feats, scalars))
                preds = (probs > 0.5).int()
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)
        return correct / total

    baseline_acc = evaluate(data_loader)
    print(f"🔹 Baseline Accuracy: {baseline_acc:.4f}")

    # Step 2: feature-wise permutation
    importances = []
    for i, name in enumerate(feature_names):
        accs = []
        for feats, scalars, labels in data_loader:
            feats = feats.to(device)
            scalars = scalars.clone().to(device)
            labels = labels.to(device)

            # Permute one scalar column
            perm = torch.randperm(scalars.size(0))
            scalars[:, i] = scalars[perm, i]

            with torch.no_grad():
                probs = torch.sigmoid(model(feats, scalars))
                preds = (probs > 0.5).int()
                acc = (preds.squeeze() == labels).float().mean().item()
                accs.append(acc)

        drop = baseline_acc - np.mean(accs)
        importances.append((name, drop))

    # Rank features
    ranked = sorted(importances, key=lambda x: x[1], reverse=True)

    print("\n📊 Feature Importance (Accuracy Drop):")
    for name, drop in ranked:
        print(f" - {name:20s}: {drop:.5f}")

    return ranked
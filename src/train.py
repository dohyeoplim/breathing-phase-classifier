import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import BreathingDataset
from src.model import ResNetMiniSE

def spec_augment(spec, freq_mask_param=8, time_mask_param=12, num_mask=2):
    spec = spec.copy()
    for _ in range(num_mask):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, spec.shape[0] - f)
        spec[f0:f0+f, :] = 0
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, spec.shape[1] - t)
        spec[:, t0:t0+t] = 0
    return spec

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    all_probs = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).int().squeeze()
            all_probs.extend(probs.squeeze().tolist())
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"✅ Validation Accuracy: {acc:.4f} | Prob range: {min(all_probs):.3f}–{max(all_probs):.3f}")
    return acc

def train_model():
    print("🚀 Training started...")

    df = pd.read_csv("input/train.csv")
    y_numeric = df["Target"].map(lambda x: 1 if x == "E" else 0)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_numeric)
    pos_weight = torch.tensor(class_weights[1], dtype=torch.float).to(device)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["Target"], random_state=42)

    train_dataset = BreathingDataset(train_df, "input/train", is_train=True)
    train_dataset.spec_transform = spec_augment
    val_dataset = BreathingDataset(val_df, "input/train", is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = ResNetMiniSE().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    os.makedirs("models", exist_ok=True)
    best_acc = 0.0
    patience, counter = 3, 0

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📘 Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}")
        val_acc = evaluate(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pth")
            print("💾 Best model saved.")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️ Early stopping triggered. Best Val Acc: {best_acc:.4f}")
                break
        torch.save(model.state_dict(), f"models/model_epoch{epoch}.pth")

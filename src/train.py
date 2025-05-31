import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.dataset import BreathingAudioDataset, breathing_collate_fn
from src.model import Model
from src.utils.display import (
    print_start, print_epoch_summary, print_validation_accuracy,
    print_success, print_warning, progress_bar, count_parameters
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class BCEWithLogitsLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

def evaluate_model(model, validation_data_loader, criterion):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    all_predictions = []

    with torch.no_grad():
        for features, scalars, labels in validation_data_loader:
            features = features.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device)
            outputs = model(features, scalars).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            all_predictions.extend(probabilities.cpu().tolist())
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(validation_data_loader)
    min_prob = min(all_predictions) if all_predictions else 0
    max_prob = max(all_predictions) if all_predictions else 1
    print_validation_accuracy(accuracy, min_prob, max_prob)
    return accuracy, avg_loss

def train_model():
    print_start("Training")

    dataframe = pd.read_csv("input/train.csv")
    numeric_labels = dataframe["Target"].map(lambda label: 1 if label == "E" else 0)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=numeric_labels)
    positive_class_weight = torch.tensor(class_weights[1], dtype=torch.float).to(device)

    train_dataframe, validation_dataframe = train_test_split(
        dataframe, test_size=0.2, stratify=dataframe["Target"], random_state=42
    )

    train_dataset = BreathingAudioDataset(train_dataframe, "input/train", is_training=True)
    validation_dataset = BreathingAudioDataset(validation_dataframe, "input/train", is_training=True)

    train_data_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=breathing_collate_fn
    )

    validation_data_loader = DataLoader(
        validation_dataset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=breathing_collate_fn
    )

    model = Model().to(device)
    criterion = BCEWithLogitsLabelSmoothing(smoothing=0.05, pos_weight=positive_class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    os.makedirs("models", exist_ok=True)
    best_validation_accuracy = 0.0
    patience = 10
    patience_counter = 0
    min_delta = 0.001

    count_parameters(model)

    accumulation_steps = 2

    for epoch_index in range(1, 51):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        progress = progress_bar(train_data_loader, description=f"📦 Epoch {epoch_index}")

        for batch_idx, (features, scalars, labels) in enumerate(progress):
            features = features.to(device)
            scalars = scalars.to(device)
            labels = labels.float().to(device)

            outputs = model(features, scalars).squeeze()
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            current_lr = optimizer.param_groups[0]['lr']
            progress.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}", "LR": f"{current_lr:.6f}"})

        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        average_loss = total_loss / len(train_data_loader)
        print_epoch_summary(epoch_index, average_loss)

        validation_accuracy, val_loss = evaluate_model(model, validation_data_loader, criterion)
        scheduler.step(validation_accuracy)

        if validation_accuracy > best_validation_accuracy + min_delta:
            best_validation_accuracy = validation_accuracy
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_accuracy': validation_accuracy,
                'validation_loss': val_loss
            }, "models/best_model.pth")
            print_success(f"Best model saved with accuracy: {best_validation_accuracy:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_warning(f"⏹️ Early stopping. Best Validation Accuracy: {best_validation_accuracy:.4f}")
                break

        if epoch_index % 5 == 0:
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_accuracy': validation_accuracy
            }, f"models/checkpoint_epoch{epoch_index}.pth")
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import BreathingAudioDataset, breathing_collate_fn
from src.model import Model

# import shap
# import matplotlib.pyplot as plt

from src.utils.display import (
    print_start,
    print_epoch_summary,
    print_validation_accuracy,
    print_success,
    print_warning,
    print_error,
    progress_bar,
    count_parameters
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, validation_data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    predicted_probabilities = []

    with torch.no_grad():
        for features, labels in validation_data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).int().squeeze()
            predicted_probabilities.extend(probabilities.squeeze().tolist())
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    min_prob = min(predicted_probabilities)
    max_prob = max(predicted_probabilities)
    print_validation_accuracy(accuracy, min_prob, max_prob)
    return accuracy


def train_model():
    print_start("Training")

    dataframe = pd.read_csv("input/train.csv")
    numeric_labels = dataframe["Target"].map(lambda label: 1 if label == "E" else 0)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=numeric_labels)
    positive_class_weight = torch.tensor(class_weights[1], dtype=torch.float).to(device)

    train_dataframe, validation_dataframe = train_test_split(
        dataframe, test_size=0.2, stratify=dataframe["Target"], random_state=42
    )


    train_dataset = BreathingAudioDataset(
        train_dataframe,
        "input/train",
        is_training=True,
    )

    validation_dataset = BreathingAudioDataset(
        validation_dataframe,
        "input/train",
        is_training=True,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=breathing_collate_fn
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=breathing_collate_fn
    )


    model = Model().to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    os.makedirs("models", exist_ok=True)
    best_validation_accuracy = 0.0
    early_stopping_patience = 3
    early_stopping_counter = 0

    count_parameters(model)

    for epoch_index in range(1, 31):
        model.train()
        total_loss = 0.0
        progress = progress_bar(train_data_loader, description=f"📦 Epoch {epoch_index}")

        for features, labels in progress:
            features, labels = features.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        average_loss = total_loss / len(train_data_loader)
        print_epoch_summary(epoch_index, average_loss)

        validation_accuracy = evaluate_model(model, validation_data_loader)
        scheduler.step(validation_accuracy)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), "models/best_model.pth")
            print_success("Best model saved.")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print_warning(f"⏹️ Early stopping. Best Validation Accuracy: {best_validation_accuracy:.4f}")
                break

        torch.save(model.state_dict(), f"models/model_epoch{epoch_index}.pth")

        # print_start("SHAP Value Analysis")
        #
        # for features, labels in validation_data_loader:
        #     features = features.to(device)
        #     break
        #
        # model.eval()
        # def model_forward(x):
        #     return torch.sigmoid(model(x)).detach().cpu().numpy()
        #
        # background = features[:20].to(device)
        # explainer = shap.DeepExplainer(model, background)
        # shap_values = explainer.shap_values(features[:32])
        #
        # shap_array = np.abs(shap_values[0]).mean(axis=0).mean(axis=1)
        #
        # print("📊 SHAP Values:")
        # for i, val in enumerate(shap_array):
        #     print(f" - Channel {i:2d}: {val:.6f}")
        #
        # plt.figure(figsize=(10, 4))
        # plt.bar(np.arange(len(shap_array)), shap_array)
        # plt.title("Mean SHAP Value per Feature Channel")
        # plt.xlabel("Feature Channel Index")
        # plt.ylabel("Mean |SHAP|")
        # plt.tight_layout()
        # plt.savefig("shap_summary.png")
        # plt.show()
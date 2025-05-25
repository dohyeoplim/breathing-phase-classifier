import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import BreathingDataset
from src.model import ResNetMiniSE

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def predict_model():
    print("🔎 Inference started...")

    test_df = pd.read_csv("input/test.csv")
    test_dataset = BreathingDataset(test_df, "input/test", is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = ResNetMiniSE().to(device)
    model_path = "models/best_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        for x, file_names in test_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.sigmoid(out).cpu().squeeze().numpy()

            if probs.ndim == 0:
                probs = [probs]

            preds = ["E" if p > 0.5 else "I" for p in probs]
            results.extend(zip(file_names, preds))

    os.makedirs("submissions", exist_ok=True)
    submission = pd.DataFrame(results, columns=["ID", "Target"])
    submission.to_csv("submissions/submission.csv", index=False)
    print("✅ submission.csv saved!")

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from datetime import datetime

from src.dataset import BreathingAudioDataset, breathing_collate_fn
from src.model import Model

from src.utils.display import (
    print_start,
    print_success,
    print_error,
    progress_bar
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def predict_model():
    print_start("Inference")

    test_dataframe = pd.read_csv("input/test.csv")
    test_dataset = BreathingAudioDataset(test_dataframe, "input/test", is_training=False)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=breathing_collate_fn
    )

    model = Model().to(device)
    model_path = "models/best_model.pth"

    if not os.path.exists(model_path):
        print_error(f"Model file not found: {model_path}")
        raise FileNotFoundError(model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for features, identifiers in progress_bar(test_data_loader, "🔮 Running inference"):
            features = features.to(device)
            outputs = model(features)
            probabilities = torch.sigmoid(outputs).cpu().squeeze().numpy()

            if probabilities.ndim == 0:
                probabilities = [probabilities]

            predicted_labels = ["E" if probability > 0.5 else "I" for probability in probabilities]
            predictions.extend(zip(identifiers, predicted_labels))

    os.makedirs("submissions", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submissions/submission_{timestamp}.csv"
    submission_dataframe = pd.DataFrame(predictions, columns=["ID", "Target"])
    submission_dataframe.to_csv(submission_path, index=False)
    print_success(f"{submission_path} 저장 완료")
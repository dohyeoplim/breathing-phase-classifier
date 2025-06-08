import pandas as pd
import torch
from src.model import CNN8, VGG
from src.train import train_model
from src.utils.dataloaders import prepare_dataloaders
from src.utils.ensemble import weighted_ensemble

def run_train_and_predict(num_scalars: int = 39, epochs: int = 100, device: torch.device = torch.device('cuda')):
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv_path="./data/train.csv",
        test_csv_path="./data/test.csv",
        precomputed_dir="./data/precomputed_features",
        batch_size=512,
        num_workers=8, # 맥에선 0이 나음
        prefetch=2
    )

    cnn8_model = CNN8(num_scalar_features=num_scalars)
    cnn8_ckpt, cnn8_val_acc = train_model(
        model = cnn8_model,
        train_loader =train_loader,
        val_loader = val_loader,
        device = device,
        save_dir = "./models/cnn8",
        num_epochs = epochs,
        base_lr=4e-4,
        weight_decay=1e-4,
        use_cutmix=True,
        use_mixup=True,
        cutmix_prob=0.6,
        mixup_prob=0.4,
        patience=25,
        warmup_epochs=4,
    )
    print(f"CNN8 best = {cnn8_val_acc:.4f}, {cnn8_ckpt}")

    vgg_model = VGG(num_scalar_features=num_scalars)
    vgg_ckpt, vgg_val_acc = train_model(
        model = vgg_model,
        train_loader=train_loader,
        val_loader = val_loader,
        device = device,
        save_dir = "./models/vgg",
        num_epochs = 140,
        patience=55,
    )
    print(f"VGG best = {vgg_val_acc:.4f}, {vgg_ckpt}")

    ckpt_paths = [cnn8_ckpt, vgg_ckpt]
    raw_scores = [cnn8_val_acc, vgg_val_acc]
    archs = ["cnn8", "vgg"]

    all_ids, avg_probs = weighted_ensemble(
        ckpt_paths=ckpt_paths,
        archs=archs,
        test_loader=test_loader,
        device=device,
        num_scalar_features=num_scalars,
        val_scores=raw_scores
    )

    predictions = (avg_probs > 0.5).astype(int)
    final_labels = ["E" if p == 1 else "I" for p in predictions]

    submission_df = pd.DataFrame({
        "ID": all_ids,
        "Target": final_labels
    })
    submission_df.to_csv("./submissions/submission.csv", index=False)
    print(submission_df.head(10))
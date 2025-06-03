import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import BreathingAudioDataset
from src.model import Model
from src.utils.display import (
    print_start,
    print_epoch_summary,
    print_validation_accuracy,
    print_success,
    print_warning,
    print_error,
    progress_bar
)

# --- 전역 설정 ---
IS_DEBUG_MODE = False 
INITIAL_PRINT_FEATURE_SUMMARY = True

# --- 장치 설정 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")

PRINT_FEATURE_SUMMARY = INITIAL_PRINT_FEATURE_SUMMARY

def evaluate_model(model, validation_data_loader, epoch_index_for_debug=None):
    model.eval()
    total_correct = 0
    total_samples = 0
    predicted_probabilities = []
    all_labels = []
    global PRINT_FEATURE_SUMMARY

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(validation_data_loader):
            features, labels = features.to(device), labels.to(device)

            if PRINT_FEATURE_SUMMARY and epoch_index_for_debug == 1 and batch_idx == 0:
                print("\n--- 검증 데이터 첫 배치 특징 요약 ---")
                print(f"Feature tensor shape: {features.shape}")
                if features.numel() > 0:
                    print(f"Feature tensor mean: {features.mean().item():.4f}, std: {features.std().item():.4f}")
                    print(f"Feature tensor min: {features.min().item():.4f}, max: {features.max().item():.4f}")
                else:
                    print("Feature tensor is empty.")
                print(f"Labels in batch (first 10): {labels.cpu().numpy().flatten()[:10]}")
                print("-------------------------------------\n")

            outputs = model(features)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).int().squeeze()

            if predictions.dim() == 0: predictions = predictions.unsqueeze(0)
            if labels.dim() == 0: labels = labels.unsqueeze(0)
            
            current_probs_list = []
            if probabilities.numel() > 0 :
                if probabilities.dim() == 0: 
                    current_probs_list.append(probabilities.item())
                else: 
                    squeezed_probs = probabilities.cpu().squeeze()
                    if squeezed_probs.dim() == 0: 
                        current_probs_list.append(squeezed_probs.item())
                    elif squeezed_probs.numel() > 0: 
                         current_probs_list.extend(squeezed_probs.tolist())
            
            if current_probs_list: predicted_probabilities.extend(current_probs_list)

            all_labels.extend(labels.cpu().tolist())
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    min_prob = min(predicted_probabilities) if predicted_probabilities else 0.0
    max_prob = max(predicted_probabilities) if predicted_probabilities else 0.0
    return accuracy, predicted_probabilities, all_labels


def train_model():
    print_start("Training")
    global PRINT_FEATURE_SUMMARY
    PRINT_FEATURE_SUMMARY = INITIAL_PRINT_FEATURE_SUMMARY 

    dataframe = pd.read_csv("input/train.csv")

    train_dataframe, validation_dataframe = train_test_split(
        dataframe, test_size=0.2, stratify=dataframe["Target"], random_state=42
    )
    print(f"학습 데이터 수: {len(train_dataframe)}, 검증 데이터 수: {len(validation_dataframe)}")

    numeric_labels_train = train_dataframe["Target"].map(lambda label: 1 if label == "E" else 0)
    if not numeric_labels_train.empty:
        unique_labels_in_train, counts_in_train = np.unique(numeric_labels_train, return_counts=True)
        print(f"학습 데이터 클래스 분포: {dict(zip(unique_labels_in_train, counts_in_train))}")
        if len(unique_labels_in_train) == 2:
            class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels_in_train, y=numeric_labels_train)
            positive_class_index = np.where(unique_labels_in_train == 1)[0]
            positive_class_weight = torch.tensor(class_weights[positive_class_index[0]] if len(positive_class_index) > 0 else 1.0, dtype=torch.float).to(device)
        else:
            print_warning(f"학습 데이터에 단일 클래스만 존재하거나({unique_labels_in_train}) 레이블이 비어있습니다. 클래스 가중치는 1.0으로 설정됩니다.")
            positive_class_weight = torch.tensor(1.0, dtype=torch.float).to(device)
        print(f"계산된 Positive 클래스 가중치 (1에 대한 가중치): {positive_class_weight.item():.4f}")
    else:
        print_error("학습 데이터셋에 레이블이 없습니다! 데이터 로딩을 확인하세요.")
        return

    # --- Dataset 파라미터 ---
    SAMPLING_RATE = 16000
    DURATION_IN_SECONDS = 2
    N_MELS = 128
    N_MFCC = 40
    HOP_LENGTH = 512
    N_FFT = 2048
    N_MELS_FOR_ENTROPY = 128
    N_FFT_FOR_ENTROPY = 2048
    HOP_LENGTH_FOR_ENTROPY = 512

    train_dataset = BreathingAudioDataset(train_dataframe, "input/train", is_training=True,
                                          sampling_rate=SAMPLING_RATE, duration_in_seconds=DURATION_IN_SECONDS,
                                          n_mels=N_MELS, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT,
                                          n_mels_for_entropy=N_MELS_FOR_ENTROPY, n_fft_for_entropy=N_FFT_FOR_ENTROPY, hop_length_for_entropy=HOP_LENGTH_FOR_ENTROPY)
    validation_dataset = BreathingAudioDataset(validation_dataframe, "input/train", is_training=True,
                                               sampling_rate=SAMPLING_RATE, duration_in_seconds=DURATION_IN_SECONDS,
                                               n_mels=N_MELS, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT,
                                               n_mels_for_entropy=N_MELS_FOR_ENTROPY, n_fft_for_entropy=N_FFT_FOR_ENTROPY, hop_length_for_entropy=HOP_LENGTH_FOR_ENTROPY)

    # --- 하이퍼파라미터 설정 ---
    LEARNING_RATE = 1e-4  
    BATCH_SIZE = 32       
    WEIGHT_DECAY = 3e-4   # 규제 강화 (이전 2e-4 -> 3e-4)
    GRU_HIDDEN_SIZE = 128 
    NUM_GRU_LAYERS = 2
    DROPOUT_RATE = 0.4    # Dropout 강화 (이전 0.35 -> 0.4)
    NUM_EPOCHS = 300      
    EARLY_STOPPING_PATIENCE = 25 

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    ACTUAL_NUM_INPUT_FEATURES = N_MELS + N_MFCC + 9 
    model = Model(num_freq_bins=ACTUAL_NUM_INPUT_FEATURES,
                  gru_hidden_size=GRU_HIDDEN_SIZE,
                  num_gru_layers=NUM_GRU_LAYERS,
                  dropout_rate=DROPOUT_RATE).to(device)
    print(f"\n--- 모델 아키텍처 ---")
    print(f"입력 특징 수 (num_freq_bins): {ACTUAL_NUM_INPUT_FEATURES}")
    print(model)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("---------------------\n")

    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 스케줄러 factor를 약간 높여 학습률 감소 폭 완화, patience는 유지 또는 약간 조정
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-7, verbose=False) 


    os.makedirs("models", exist_ok=True)
    best_validation_accuracy = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    print_warning(f"학습 시작: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, WeightDecay={WEIGHT_DECAY}, GRU_Hidden={GRU_HIDDEN_SIZE}, GRU_Layers={NUM_GRU_LAYERS}, Dropout={DROPOUT_RATE}, Epochs={NUM_EPOCHS}, EarlyStopPatience={EARLY_STOPPING_PATIENCE}")

    for epoch_index in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        progress = progress_bar(train_data_loader, description=f"📦 Epoch {epoch_index}/{NUM_EPOCHS}")

        for batch_idx, (features, labels) in enumerate(progress):
            features, labels = features.to(device), labels.float().unsqueeze(1).to(device)

            if PRINT_FEATURE_SUMMARY and epoch_index == 1 and batch_idx == 0:
                print("\n--- 학습 데이터 첫 배치 특징 요약 ---")
                if features.numel() > 0:
                    print(f"Feature tensor shape: {features.shape}")
                    print(f"Feature tensor mean: {features.mean().item():.4f}, std: {features.std().item():.4f}")
                    print(f"Feature tensor min: {features.min().item():.4f}, max: {features.max().item():.4f}")
                else:
                    print("Feature tensor is empty.")
                print(f"Labels in batch (first 10): {labels.cpu().numpy().flatten()[:10]}")
                print("-------------------------------------\n")
                PRINT_FEATURE_SUMMARY = False

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print_error(f"손실 값 오류 (NaN 또는 Inf) 발생! 에포크 {epoch_index}, 배치 {batch_idx}. 학습을 중단합니다.")
                if features.numel() > 0: print_error(f"특징값 (features) mean: {features.mean().item()}, std: {features.std().item()}")
                if outputs.numel() > 0: print_error(f"출력값 (outputs) mean: {outputs.mean().item()}, std: {outputs.std().item()}")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.1e}"})

        average_loss = total_loss / len(train_data_loader) if len(train_data_loader) > 0 else 0.0
        print_epoch_summary(epoch_index, average_loss)

        validation_accuracy, val_probs, _ = evaluate_model(model, validation_data_loader, epoch_index_for_debug=epoch_index)
        min_p = min(val_probs) if val_probs else 0.0
        max_p = max(val_probs) if val_probs else 0.0
        print_validation_accuracy(validation_accuracy, min_p, max_p)

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(validation_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        if abs(prev_lr - current_lr) > 1e-9: 
            print_warning(f"학습률 변경됨: {prev_lr:.2e} -> {current_lr:.2e} (에포크 {epoch_index}, Val Acc: {validation_accuracy:.4f})")
        else:
            print(f"에포크 {epoch_index} 종료. 현재 학습률: {current_lr:.2e}")


        if validation_accuracy > best_validation_accuracy + 1e-5 : # 아주 미미한 개선(1e-5 이상)도 인정
            print_success(f"성능 개선! 이전 최고: {best_validation_accuracy:.4f} -> 현재: {validation_accuracy:.4f}")
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch_index
            torch.save(model.state_dict(), "models/best_model.pth")
            print_success(f"🚀 Best model saved at Epoch {best_epoch} with accuracy: {best_validation_accuracy:.4f} (LR: {current_lr:.2e})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print_warning(f"성능 개선 없음 (또는 미미함). Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}. Best Acc: {best_validation_accuracy:.4f} at Epoch {best_epoch}")
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print_warning(f"⏹️ Early stopping at epoch {epoch_index}. Best Validation Accuracy: {best_validation_accuracy:.4f} (achieved at Epoch {best_epoch})")
                break

        if epoch_index % 50 == 0 and not IS_DEBUG_MODE: 
             print_warning(f"💾 Saving checkpoint model at epoch {epoch_index}...")
             torch.save(model.state_dict(), f"models/model_epoch{epoch_index}.pth")

    print_success(f"🎉 Training completed. Best Validation Accuracy: {best_validation_accuracy:.4f} (achieved at Epoch {best_epoch})")

# if __name__ == '__main__':
#     train_model()
#     pass
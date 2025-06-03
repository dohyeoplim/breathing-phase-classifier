import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import BreathingAudioDataset
from src.model import Model
from src.utils.display import (
    print_start,
    print_success,
    print_error,
    progress_bar
)

# --- 전역 설정 ---
CLIP_FEATURES_FOR_DEBUG = False # 추론 시에는 특징 클리핑 사용 안 함

# --- 장치 설정 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference using device: {device}")

def predict_model():
    print_start("Inference")

    test_dataframe = pd.read_csv("input/test.csv")

    # --- Dataset 파라미터 (train.py와 완벽히 일치시켜야 함) ---
    SAMPLING_RATE = 16000
    DURATION_IN_SECONDS = 2
    N_MELS = 128
    N_MFCC = 40
    HOP_LENGTH = 512
    N_FFT = 2048
    N_MELS_FOR_ENTROPY = 128 
    N_FFT_FOR_ENTROPY = 2048 
    HOP_LENGTH_FOR_ENTROPY = 512


    test_dataset = BreathingAudioDataset(test_dataframe, "input/test", is_training=False,
                                         sampling_rate=SAMPLING_RATE, duration_in_seconds=DURATION_IN_SECONDS,
                                         n_mels=N_MELS, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT,
                                         n_mels_for_entropy=N_MELS_FOR_ENTROPY, n_fft_for_entropy=N_FFT_FOR_ENTROPY, hop_length_for_entropy=HOP_LENGTH_FOR_ENTROPY)
    
    test_data_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, pin_memory=False, shuffle=False)

    # --- 모델 파라미터 (train.py와 완벽히 일치시켜야 함) ---
    ACTUAL_NUM_INPUT_FEATURES = N_MELS + N_MFCC + 9 # Aux 특징 9개 가정
    GRU_HIDDEN_SIZE = 128
    NUM_GRU_LAYERS = 2
    DROPOUT_RATE = 0.3 # 학습 시 사용된 값과 동일하게 (저장된 모델 구조와 맞아야 함)

    model = Model(num_freq_bins=ACTUAL_NUM_INPUT_FEATURES,
                  gru_hidden_size=GRU_HIDDEN_SIZE,
                  num_gru_layers=NUM_GRU_LAYERS,
                  dropout_rate=DROPOUT_RATE).to(device)
    
    model_path = "models/best_model.pth"

    if not os.path.exists(model_path):
        print_error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first using 'python main.py train'.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print_error(f"Error loading model state_dict from {model_path}: {e}")
        print_warning("Please ensure the model architecture defined in 'src/model.py' matches the saved model checkpoint.")
        raise
        
    model.eval()

    predictions = []
    with torch.no_grad():
        for features, identifiers in progress_bar(test_data_loader, "🔮 Running inference"):
            features = features.to(device)

            if CLIP_FEATURES_FOR_DEBUG: # 현재 False
                 # features = torch.clamp(features, min=FEATURE_CLIP_MIN, max=FEATURE_CLIP_MAX)
                 pass

            outputs = model(features)
            probabilities = torch.sigmoid(outputs).cpu().squeeze()

            current_probs_list = []
            if probabilities.numel() > 0:
                if probabilities.dim() == 0:
                    current_probs_list.append(probabilities.item())
                else:
                    current_probs_list.extend(probabilities.tolist())
            
            current_identifiers = [identifiers] if isinstance(identifiers, str) else list(identifiers)

            if len(current_identifiers) != len(current_probs_list):
                print_warning(f"Mismatch between number of identifiers ({len(current_identifiers)}) and "
                              f"probabilities ({len(current_probs_list)}) for a batch. Identifiers: {current_identifiers}, Probs: {current_probs_list}")
                # 길이가 다를 경우, 짧은 쪽에 맞추거나 해당 배치를 건너뛸 수 있음
                # 여기서는 확률이 없는 식별자는 예측 불가로 처리하거나, 식별자가 없는 확률은 무시
                # 가장 안전한 방법은 길이가 같을 때만 처리
                if len(current_identifiers) == 0 or len(current_probs_list) == 0 : # 둘 중 하나라도 비어있으면 다음으로
                    continue
                min_len = min(len(current_identifiers), len(current_probs_list))
                current_identifiers = current_identifiers[:min_len]
                current_probs_list = current_probs_list[:min_len]


            predicted_labels = ["E" if probability > 0.5 else "I" for probability in current_probs_list]
            
            if len(current_identifiers) == len(predicted_labels): # 최종 확인
                 predictions.extend(zip(current_identifiers, predicted_labels))
            else:
                print_error(f"CRITICAL: Length mismatch after attempting to fix for identifiers and predicted_labels. Skipping batch.")


    submission_directory = "submissions"
    os.makedirs(submission_directory, exist_ok=True)
    submission_file_path = os.path.join(submission_directory, "submission.csv")
    
    submission_dataframe = pd.DataFrame(predictions, columns=["ID", "Target"])
    if submission_dataframe.empty and len(test_dataframe) > 0:
        print_error("No predictions were made. Submission file will be empty or not generated as expected.")
    elif not submission_dataframe.empty:
         submission_dataframe.to_csv(submission_file_path, index=False)
         print_success(f"{submission_file_path} saved!")
    else:
        print_warning("Test dataframe was empty, so no submission file was generated.")


# if __name__ == '__main__':
# predict_model()
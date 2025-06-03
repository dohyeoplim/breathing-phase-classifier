import numpy as np

def normalize_to_target_root_mean_square(signal: np.ndarray, target_root_mean_square: float = 0.1) -> np.ndarray:
    # RMS가 0이거나 매우 작은 경우 (무음 오디오)를 대비해 epsilon 추가
    current_root_mean_square = np.sqrt(np.mean(signal**2)) + 1e-9 # epsilon 값 유지
    if current_root_mean_square < 1e-8: # RMS가 거의 0이면 (0으로 나누기 방지)
        return signal # 원본 신호 반환 (또는 np.zeros_like(signal) 고려)
    return signal * (target_root_mean_square / current_root_mean_square)

def normalize_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    # 특징 행렬 전체에 대한 표준화 (Z-score normalization)
    mean = np.mean(matrix)
    std = np.std(matrix)
    if std < 1e-9: # 표준편차가 거의 0이면 (모든 값이 동일한 경우 등)
        # 모든 값이 평균과 같다면, (matrix - mean)은 0이 됨.
        return matrix - mean 
    return (matrix - mean) / std
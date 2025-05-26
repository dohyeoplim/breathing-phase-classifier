import numpy as np

def normalize_to_target_root_mean_square(signal: np.ndarray, target_root_mean_square: float = 0.1) -> np.ndarray:
    current_root_mean_square = np.sqrt(np.mean(signal**2)) + 1e-6
    return signal * (target_root_mean_square / current_root_mean_square)

def normalize_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    return (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-6)
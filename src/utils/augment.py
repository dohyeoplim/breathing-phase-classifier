import numpy as np

def apply_random_volume_gain(signal: np.ndarray, minimum_gain: float = 0.7, maximum_gain: float = 1.3) -> np.ndarray:
    gain = np.random.uniform(minimum_gain, maximum_gain)
    return signal * gain

def apply_spec_augmentation(spectrogram: np.ndarray, maximum_frequency_mask_width: int = 8, maximum_time_mask_width: int = 12, number_of_masks: int = 2) -> np.ndarray:
    augmented = spectrogram.copy()
    for _ in range(number_of_masks):
        frequency_width = np.random.randint(0, maximum_frequency_mask_width)
        frequency_start = np.random.randint(0, spectrogram.shape[0] - frequency_width)
        augmented[frequency_start:frequency_start+frequency_width, :] = 0

        time_width = np.random.randint(0, maximum_time_mask_width)
        time_start = np.random.randint(0, spectrogram.shape[1] - time_width)
        augmented[:, time_start:time_start+time_width] = 0

    return augmented
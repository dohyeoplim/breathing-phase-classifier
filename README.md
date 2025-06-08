# 🫁 A Deep Learning Approach to Respiratory Phase Detection from Audio Spectrograms

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/dohyeoplim/a83fa674e537473f1adb66960db0a32c/kaggle_2_v3.ipynb)

Developed by:

- **Dohyeop Lim** — Dept. of Artificial Intelligence, Seoul National University of Science and Technology  
- **Songwon Won** — Dept. of Artificial Intelligence, Seoul National University of Science and Technology
  
<br/>

## 🚀 Quick Start


**Project Structure:**
```plaintext
├── main.py              # Main entry point for training or precomputing
├── src/
│   ├── augmentation.py  # CutMix and Mixup
│   ├── dataset.py       # Dataset Class from the precomputed features
│   ├── model.py         # Model architectures
│   ├── train.py         # Train logics
│   ├── precompute/      # Feature extraction pipeline: spectrograms, scalars, and preprocessing utilities
│   └── scripts/         # Training and evaluation routines
└── README.md
```

**Train and Predict:**
```bash
python main.py
```

<br/>

## 📊 Dataset Overview

### 📁 Provided Files

- **`train/`**: Labeled `.wav` files of single breathing phases (inhale/exhale).
- **`test/`**: Unlabeled `.wav` files for evaluation.
- **`train.csv`**: Metadata linking `train/` files to ground-truth labels.
- **`test.csv`**: IDs of `test/` files to be predicted.

### 📝 Column Descriptions

#### `train.csv`

| Column     | Type   | Description                                  |
|------------|--------|----------------------------------------------|
| ID  | string | ID matching a `.wav` file in `train/`        |
| Target | string | Breathing phase: `I` = Inhale, `E` = Exhale |

#### `test.csv`

| Column | Type   | Description                            |
|--------|--------|----------------------------------------|
| ID     | string | ID matching a `.wav` file in `test/`   |

<br />

## 🎧 Feature Extraction Pipeline

All audio samples are preprocessed and saved as `.npz` files to reduce computation during training phase.

![Waveform](https://github.com/user-attachments/assets/144ab1b4-b24c-48ef-87e1-61a50d55835c)


### Spectrogram-based features

- `STFT` (Log-Power Spectrogram)
  ![STFT](https://github.com/user-attachments/assets/3bedca86-7ade-4bb8-9f14-a8e3aff220e0)
- `mel`, `mel_delta`, `mel_delta2`: log-scaled Mel-spectrogram (128 bins) and 1st and 2nd temporal derivatives of mel
  ![Mel](https://github.com/user-attachments/assets/cc4803fa-e331-4812-973b-45f5b1eb8dd7)
- `mfcc`: 40 MFCCs + delta + delta², vertically stacked and rescaled to 128 bins
  ![MFC](https://github.com/user-attachments/assets/c8ea0e8f-c6f9-4b47-92fb-cd7cb55b7b8f)
- `chroma`: concatenation of chroma energy (`chroma_stft`) and CENS features
  ![chroma](https://github.com/user-attachments/assets/c4e10ca0-be6c-4248-8929-1bb1276efa59)
- `gammatone`: Mel-filter-applied gammatone approximation (64 bands → 128 bins)
- `lpc`: LPC coefficient matrix padded to `[128 × T]`
- `mod_spec`: 2D DCT of log-mel (spectral modulation)
- `tempogram`: tempogram extracted from onset envelope

### Scalar features
- RMS, ZCR, centroid, bandwidth, rolloff, flatness, contrast
- Envelope mean/variance, peak count and statistics
- Low-frequency energy ratio, spectral flux
- Skewness, kurtosis, amplitude percentiles
- Short-lag autocorrelation values, first minimum index

### Execution

```bash
python main.py --precompute
```

<br />

## 🧠 Model Overview

### 📦 CNN8
A compact 8-layer convolutional neural network.

- **Input**:  
  - Spectrogram input: `shape = [B, 9, H, W]`  
  - Scalar input: 39-dimensional waveform-derived vector

- **Spectrogram Branch**:
  - 4 convolutional blocks:
    - Conv2D → ReLU → BatchNorm
    - MaxPool2D and Dropout2D applied after 2nd and 4th blocks
  - Final: `AdaptiveAvgPool2d((1, 1))` to produce global feature map

- **Scalar Branch**:
  - Linear(39 → 64) → ReLU → BatchNorm → Dropout  
  - Linear(64 → 64) → ReLU → BatchNorm

- **Classifier**:
  - Concatenates pooled CNN feature (256-d) with scalar embedding (64-d)  
  - Linear(320 → 256) → ReLU → BatchNorm → Dropout  
  - Linear(256 → 128) → ReLU → BatchNorm  
  - Linear(128 → 1) → output logit

- **Total Parameters**: **~2.43M**

 
### 📦 VGG-Inspired

Inspired by the VGG family, enhanced with GELU, residual connection, and aggressive regularizations.

- **Input**:  
  - Spectrogram input: `shape = [B, 9, H, W]`  
  - Scalar input: 39-dimensional waveform-derived vector

- **Spectrogram Branch**:
  - **Block 1**: 3 × Conv2D(64) → BatchNorm → GELU → Downsample  
  - **Block 2**: 3 × Conv2D(128) → BatchNorm → GELU → MaxPool  
  - **Block 3**: 3 × Conv2D(256) → BatchNorm → GELU → MaxPool  
  - **Block 4**: 3 × Conv2D(512) → BatchNorm → GELU  
    - **Residual path**: Conv2D(256 → 512) → BatchNorm  
    - Output = Main + Residual  
  - Final: `AdaptiveAvgPool2d((1, 1))`

- **Scalar Branch**:
  - Linear(39 → 64) → BatchNorm → GELU → Dropout  
  - Linear(64 → 64) → BatchNorm → GELU

- **Classifier**:
  - Concatenates pooled CNN feature (512-d) with scalar embedding (64-d)  
  - Linear(576 → 256) → BatchNorm → GELU → Dropout  
  - Linear(256 → 128) → BatchNorm → GELU → Dropout  
  - Linear(128 → 1) → output logit

- **Total Parameters**: **~8.15M**

<br />

## ⚒️ Training Strategy

- **Optimizer**: AdamW with `weight_decay = 1e-4`
- **Loss Function**: `BCEWithLogitsLoss`
- **Learning Rate Scheduling**:
  - Linear warm-up during the early phase
  - Cosine annealing for the remaining steps
- **Epochs**:
  - CNN8: up to 100 epochs
  - VGG: up to 140 epochs
- **Gradient Clipping**: Applied with `max_norm = 1.0`
- **Data Augmentation**:
  - **CutMix** and **MixUp** are applied probabilistically during training (`cutmix_prob = 0.6`, `mixup_prob = 0.4`)
  - Augmentations are activated after a warm-up period (`warmup_epochs = 4`)

<br />

## 🔗 Final Prediction Aggregation

Each model outputs a sigmoid-activated probability. The final prediction is a weighted average:

Let:
- `M = {model₁, model₂, ..., modelₙ}` be the set of models (here: CNN8 and VGG)
- `logitsᵢ` be the raw output of model *i*
- `σ(logitsᵢ)` be the sigmoid activation (i.e., predicted probability)
- `wᵢ` be the validation score of model *i*
- `αᵢ = softmax(wᵢ)` if `use_softmax_weights = True`, else normalized directly

Then the final ensemble output is:

```math
P_\text{final} = \sum_{i=1}^{n} \alpha_i \cdot \sigma(\text{logits}_i)
```


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

**Precompute Features Only:**
```bash
python main.py --precompute
```

<br/>

## 📊 Dataset Description

### Provided Files 📁
- **train/**

  Contains .wav audio files used for model training. Each file represents a single instance of a person’s breathing sound, labeled as either an inhale or an exhale.

- **test/**

  Contains .wav audio files used for model evaluation. These files do not include labels; the model must predict whether each clip is an inhale or exhale.

- **train.csv**

  Metadata file for the training set. Each row corresponds to an audio file in the train/ folder and includes the ground-truth label.

- **test.csv**

  Metadata file for the test set. It lists the IDs of audio files in the test/ folder that require prediction.


### 📝 Column Descriptions

#### `train.csv`

| Column     | Type   | Description                                                   |
|------------|--------|---------------------------------------------------------------|
| file_name  | string | Unique identifier for each audio file in the `train/` folder |
| label      | string | Ground-truth label of the breathing phase (`I` for Inhale, `E` for Exhale) |

#### `test.csv`

| Column | Type   | Description                                                   |
|--------|--------|---------------------------------------------------------------|
| ID     | string | Unique identifier for each audio file in the `test/` folder   |

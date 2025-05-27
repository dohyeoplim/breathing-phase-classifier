import os
import re
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

CSV_PATH = "../input/train.csv"
AUDIO_DIR = "../input/train"
SAMPLE_RATE = 16000
MAX_SAMPLES = 500

df = pd.read_csv(CSV_PATH)
df["Label"] = df["Target"].map({"E": 1, "I": 0})
df["Cleaned_ID"] = df["ID"].apply(lambda x: re.sub(r'_[EI]_', '_', x) + ".wav")

print("✅ Label Counts:")
print(df["Target"].value_counts())
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Target")
plt.title("Label Distribution")
plt.tight_layout()
plt.show()

wave_stats = []

print("\n🔍 Analyzing waveforms...")
for i, row in tqdm(df.iterrows(), total=min(len(df), MAX_SAMPLES)):
    if i >= MAX_SAMPLES:
        break
    path = os.path.join(AUDIO_DIR, re.sub(r'_[EI]_', '_', row["ID"]) + ".wav")
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    rms = np.sqrt(np.mean(y**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    wave_stats.append({
        "ID": row["ID"],
        "Target": row["Target"],
        "Length": len(y),
        "RMS": rms,
        "ZCR": zcr,
        "SpectralCentroid": centroid,
    })

stats_df = pd.DataFrame(wave_stats)

print("\n📊 RMS, ZCR, Spectral Centroid Summary:")
for col in ["RMS", "ZCR", "SpectralCentroid"]:
    print(f"\n--- {col} ---")
    print(stats_df.groupby("Target")[col].describe()[["mean", "std", "min", "max"]])

for feature in ["RMS", "ZCR", "SpectralCentroid"]:
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=stats_df, x="Target", y=feature)
    plt.title(f"{feature} by Class")
    plt.tight_layout()
    plt.show()

for label in ["E", "I"]:
    example_row = df[df["Target"] == label].iloc[0]
    path = os.path.join(AUDIO_DIR, re.sub(r'_[EI]_', '_', example_row["ID"]) + ".wav")
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    print(f"\n🎧 Example: {label} | ID: {example_row['ID']}")
    print(f" - Length: {len(y)}")
    print(f" - RMS: {np.sqrt(np.mean(y**2)):.5f}")
    print(f" - ZCR: {np.mean(librosa.feature.zero_crossing_rate(y)):.5f}")
    print(f" - Centroid: {np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)):.2f}")

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.title(f"Mel Spectrogram ({label})")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.title(f"MFCC ({label})")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
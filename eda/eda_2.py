import os
import re
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal
from tqdm import tqdm

CSV_PATH = "input/train.csv"
AUDIO_DIR = "input/train"
SAMPLE_RATE = 16000
MAX_SAMPLES = 500

df = pd.read_csv(CSV_PATH)
df["Label"] = df["Target"].map({"E": 1, "I": 0})
df["Cleaned_ID"] = df["ID"].apply(lambda x: re.sub(r'_[EI]_', '_', x) + ".wav")

rolloff_values = []

print("📈 Precomputing spectral rolloffs for band edges...")
for i, row in tqdm(df.iterrows(), total=min(len(df), MAX_SAMPLES)):
    if i >= MAX_SAMPLES:
        break
    path = os.path.join(AUDIO_DIR, re.sub(r'_[EI]_', '_', row["ID"]) + ".wav")
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    rolloff_values.append(np.mean(rolloff))

rolloff_25 = np.percentile(rolloff_values, 25)
rolloff_50 = np.percentile(rolloff_values, 50)
rolloff_75 = np.percentile(rolloff_values, 75)

bands = [(30, rolloff_25), (rolloff_25, rolloff_50), (rolloff_50, rolloff_75)]
print(f"\n📊 Data-driven bands (Hz): {[(int(a), int(b)) for a, b in bands]}")

results = []

print("\n🔍 Analyzing waveforms with data-driven band features...")
for i, row in tqdm(df.iterrows(), total=min(len(df), MAX_SAMPLES)):
    if i >= MAX_SAMPLES:
        break
    file_path = os.path.join(AUDIO_DIR, re.sub(r'_[EI]_', '_', row["ID"]) + ".wav")
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    envelope = np.abs(scipy.signal.hilbert(y))
    envelope_smooth = scipy.signal.savgol_filter(envelope, 301, polyorder=3)
    peak_env = np.max(envelope_smooth)
    mean_env = np.mean(envelope_smooth)
    slope_env = np.max(np.gradient(envelope_smooth))

    rms = np.sqrt(np.mean(y**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    active_ratio = np.mean(np.abs(y) > 0.02)

    h, p = librosa.effects.hpss(y)
    h_energy = np.mean(h**2)
    p_energy = np.mean(p**2)

    band_energies = []
    for low, high in bands:
        sos = scipy.signal.butter(3, [low, high], btype='band', fs=sr, output='sos')
        filtered = scipy.signal.sosfilt(sos, y)
        band_energy = np.sqrt(np.mean(filtered**2))
        band_energies.append(band_energy)

    results.append({
        "ID": row["ID"],
        "Target": row["Target"],
        "RMS": rms,
        "ZCR": zcr,
        "Centroid": centroid,
        "EnvelopePeak": peak_env,
        "EnvelopeSlope": slope_env,
        "BandEnergy_Low": band_energies[0],
        "BandEnergy_Mid": band_energies[1],
        "BandEnergy_High": band_energies[2],
        "ActiveRatio": active_ratio,
        "HarmonicEnergy": h_energy,
        "PercussiveEnergy": p_energy
    })

stats_df = pd.DataFrame(results)

print("\n📊 Grouped Feature Means by Class:")
numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
group_means = stats_df.groupby("Target")[numeric_cols].mean()
print(group_means)

feature_cols = [
    "RMS", "ZCR", "Centroid", "EnvelopePeak", "EnvelopeSlope",
    "BandEnergy_Low", "BandEnergy_Mid", "BandEnergy_High",
    "ActiveRatio", "HarmonicEnergy", "PercussiveEnergy"
]

for feature in feature_cols:
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=stats_df, x="Target", y=feature)
    plt.title(f"{feature} by Class")
    plt.tight_layout()
    plt.show()
import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')
from src.precompute.methods import (
    pad_or_truncate, pad_time, pad_freq,
    extract_gammatone_features, extract_lpc_features,
    extract_spectral_modulation_features, extract_enhanced_scalar_features
)

SR = 16000
DURATION = 1.0
EXPECTED_LEN = int(SR * DURATION)
N_MELS = 128
N_MFCC = 40
HOP_LENGTH = 256
N_FFT = 512
FMAX = 4500

DELTA_ORDER = 2
N_GAMMATONE = 64
N_LPC = 12

def process_and_save_npz(args):
    file_id, wav_path, target_dir = args
    try:
        y, _ = librosa.load(wav_path, sr=SR)
        y = pad_or_truncate(y, EXPECTED_LEN)
        T_FIXED = (EXPECTED_LEN // HOP_LENGTH) + 1

        mel_spec = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_delta = librosa.feature.delta(mel_db, order=1)
        mel_delta2 = librosa.feature.delta(mel_db, order=2)
        mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        delta_norm = (mel_delta - mel_delta.mean()) / (mel_delta.std() + 1e-8)
        delta2_norm = (mel_delta2 - mel_delta2.mean()) / (mel_delta2.std() + 1e-8)
        mel_padded = pad_time(mel_norm, N_MELS, T_FIXED)
        delta_padded = pad_time(delta_norm, N_MELS, T_FIXED)
        delta2_padded = pad_time(delta2_norm, N_MELS, T_FIXED)

        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_all = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        mfcc_norm = (mfcc_all - mfcc_all.mean(axis=1, keepdims=True)) / (mfcc_all.std(axis=1, keepdims=True) + 1e-8)
        mfcc_time = pad_time(mfcc_norm, mfcc_all.shape[0], T_FIXED)
        mfcc_padded = pad_freq(mfcc_time, mfcc_all.shape[0], N_MELS)

        stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        chroma = librosa.feature.chroma_stft(S=stft, sr=SR, hop_length=HOP_LENGTH)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP_LENGTH)
        chroma_all = np.vstack([chroma, chroma_cens])
        chroma_norm = (chroma_all - chroma_all.mean(axis=1, keepdims=True)) / (chroma_all.std(axis=1, keepdims=True) + 1e-8)
        chroma_time = pad_time(chroma_norm, 24, T_FIXED)
        chroma_padded = pad_freq(chroma_time, 24, N_MELS)

        gammatone = extract_gammatone_features(y, SR, N_GAMMATONE)
        gammatone_norm = (gammatone - gammatone.mean()) / (gammatone.std() + 1e-8)
        gammatone_time = pad_time(gammatone_norm, N_GAMMATONE, T_FIXED)
        gammatone_padded = pad_freq(gammatone_time, N_GAMMATONE, N_MELS)

        lpc = extract_lpc_features(y, N_LPC)
        lpc_norm = (lpc - lpc.mean()) / (lpc.std() + 1e-8)
        lpc_time = pad_time(lpc_norm, N_LPC, T_FIXED)
        lpc_padded = pad_freq(lpc_time, N_LPC, N_MELS)

        mod_spec = extract_spectral_modulation_features(mel_db)
        mod_spec_norm = (mod_spec - mod_spec.mean()) / (mod_spec.std() + 1e-8)
        mod_spec_time = pad_time(mod_spec_norm, 40, T_FIXED)
        mod_spec_padded = pad_freq(mod_spec_time, 40, N_MELS)

        onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_LENGTH)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=SR, hop_length=HOP_LENGTH)
        tempo_norm = (tempogram - tempogram.mean()) / (tempogram.std() + 1e-8)
        tempo_time = pad_time(tempo_norm, tempogram.shape[0], T_FIXED)
        tempo_padded = pad_freq(tempo_time, tempogram.shape[0], N_MELS)

        scalars_arr = extract_enhanced_scalar_features(y, SR)

        mel_arr = mel_padded.astype(np.float32)
        delta_arr = delta_padded.astype(np.float32)
        delta2_arr = delta2_padded.astype(np.float32)
        mfcc_arr = mfcc_padded.astype(np.float32)
        chroma_arr = chroma_padded.astype(np.float32)
        gammatone_arr = gammatone_padded.astype(np.float32)
        lpc_arr = lpc_padded.astype(np.float32)
        mod_spec_arr = mod_spec_padded.astype(np.float32)
        tempo_arr = tempo_padded.astype(np.float32)

        save_path = os.path.join(target_dir, file_id + ".npz")
        np.savez(save_path,
                 mel=mel_arr,
                 mfcc=mfcc_arr,
                 chroma=chroma_arr,
                 mel_delta=delta_arr,
                 mel_delta2=delta2_arr,
                 gammatone=gammatone_arr,
                 lpc=lpc_arr,
                 mod_spec=mod_spec_arr,
                 tempogram=tempo_arr,
                 scalars=scalars_arr)

        return file_id, True, None

    except Exception as e:
        return file_id, False, str(e)
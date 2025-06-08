import os
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.precompute.process import process_and_save_npz
from src.utils.display import print_error, print_success, print_info

SR = 16000
DURATION = 1.0
EXPECTED_LEN = int(SR * DURATION)
N_WORKERS = 2
TRAIN_CSV_PATH = "input/train.csv"
TEST_CSV_PATH = "input/test.csv"
TRAIN_AUDIO_DIR = "input/train"
TEST_AUDIO_DIR = "input/test"
PRECOMP_DIR = "input/precomputed/"

def process_dataset_threaded(df, audio_dir, target_dir, dataset_name):
    args_list = []
    for _, row in df.iterrows():
        file_id = row["ID"]
        if dataset_name == "train":
            wav_name = re.sub(r'_[EI]_', '_', file_id) + ".wav"
        else:
            wav_name = file_id if file_id.endswith(".wav") else (file_id + ".wav")
        wav_path = os.path.join(audio_dir, wav_name)
        args_list.append((file_id, wav_path, target_dir))

    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_and_save_npz, args): args for args in args_list}
        with tqdm(total=len(args_list), desc=f"{dataset_name} 처리 중") as pbar:
            for future in as_completed(futures):
                file_id, success, error = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    print_error(f"{file_id}: {error}")
                pbar.update(1)

    print_success(f"{successful} 성공, {failed} 실패")

def precompute():
    os.makedirs(PRECOMP_DIR, exist_ok=True)
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df  = pd.read_csv(TEST_CSV_PATH)

    process_dataset_threaded(train_df, TRAIN_AUDIO_DIR, PRECOMP_DIR, "train")

    process_dataset_threaded(test_df, TEST_AUDIO_DIR, PRECOMP_DIR, "test")

    print_success("완료")

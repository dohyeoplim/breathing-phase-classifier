import argparse
import torch
from src.scripts import run_train_and_predict
from src.precompute.core import precompute

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if args.precompute:
        precompute()
    else:
        run_train_and_predict(device=device)

if __name__ == "__main__":
    main()
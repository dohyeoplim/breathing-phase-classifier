import argparse
from src.train import train_model
from src.infer import predict_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"], help="train | predict")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        predict_model()

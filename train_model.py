
from pathlib import Path

from classifier import train_model


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


if __name__ == "__main__":

    train_model(
        DATASET_PATH,
        MODEL_PATH,
        ENCODER_PATH
    )

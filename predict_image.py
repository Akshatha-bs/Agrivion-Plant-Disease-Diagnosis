
from pathlib import Path
import argparse

from classifier import load_model, analyze_image
from recommendations import REMEDIES


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


def main():

    parser = argparse.ArgumentParser(
        description="Agrivion Plant Disease Diagnosis"
    )

    parser.add_argument(
        "image",
        help="Path to the plant leaf image"
    )

    args = parser.parse_args()

    model, label_encoder = load_model(
        MODEL_PATH,
        ENCODER_PATH
    )

    disease, confidence = analyze_image(
        args.image,
        model,
        label_encoder
    )

    print("\n--- Agrivion Diagnosis Result ---")

    print(f"Predicted disease: {disease}")

    print(f"Confidence: {confidence:.2%}")

    remedy = REMEDIES.get(
        disease,
        "Please consult an agricultural expert for treatment advice."
    )

    print(f"Recommendation: {remedy}")


if __name__ == "__main__":
    main()

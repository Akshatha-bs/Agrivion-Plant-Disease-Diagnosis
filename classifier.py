
"""
Agrivion image classification module.

Reconstructed from the Python implementation excerpts
available in the printed project report.
"""

import cv2
import numpy as np
import pickle

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_image(image_path):
    """Resize an image and extract visual features."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (224, 224))

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(hsv_image[:, :, 0])
    s_mean = np.mean(hsv_image[:, :, 1])
    v_mean = np.mean(hsv_image[:, :, 2])

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(
        gray_image, cv2.CV_64F, 1, 0, ksize=3
    )

    sobel_y = cv2.Sobel(
        gray_image, cv2.CV_64F, 0, 1, ksize=3
    )

    canny_edges = cv2.Canny(gray_image, 100, 200)

    features = [
        h_mean,
        s_mean,
        v_mean,
        np.mean(np.abs(sobel_x)),
        np.mean(np.abs(sobel_y)),
        np.mean(canny_edges > 0)
    ]

    return np.array(features)


def train_model(dataset_path, model_path, encoder_path):

    dataset_path = Path(dataset_path)

    features = []
    labels = []

    for plant_folder in dataset_path.iterdir():

        if not plant_folder.is_dir():
            continue

        for disease_folder in plant_folder.iterdir():

            if not disease_folder.is_dir():
                continue

            label = disease_folder.name

            for image_path in disease_folder.iterdir():

                if image_path.suffix.lower() in [
                    ".jpg", ".jpeg", ".png"
                ]:

                    try:
                        image_features = preprocess_image(image_path)
                        features.append(image_features)
                        labels.append(label)

                    except ValueError:
                        print(f"Skipping unreadable image: {image_path}")

    if len(features) < 2:
        raise ValueError(
            "Not enough valid training images. "
            "Please add more images to the dataset."
        )

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"Model accuracy: {accuracy:.2f}")

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    with open(encoder_path, "wb") as file:
        pickle.dump(label_encoder, file)

    return model, label_encoder


def load_model(model_path, encoder_path):

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(encoder_path, "rb") as file:
        label_encoder = pickle.load(file)

    return model, label_encoder


def analyze_image(image_path, model, label_encoder):

    features = preprocess_image(image_path).reshape(1, -1)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    disease = label_encoder.inverse_transform([prediction])[0]

    confidence = float(np.max(probabilities))

    return disease, confidence

import numpy as np
import joblib
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Artifacts directory
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Model files
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
MLB_PATH = ARTIFACTS_DIR / "mlb.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

# Load trained objects
model = joblib.load(MODEL_PATH)
mlb = joblib.load(MLB_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def predict_disease(user_symptoms):
    """
    Predict disease from a list of symptoms.

    Example:
        ["fever", "cough", "headache"]
    """

    # Clean input symptoms
    user_symptoms = [
        symptom.strip().lower()
        for symptom in user_symptoms
        if symptom.strip()
    ]

    # Create empty feature vector
    features = np.zeros(len(mlb.classes_), dtype=int)

    # Convert symptoms to the same binary representation
    for i, symptom in enumerate(mlb.classes_):
        if symptom.lower() in user_symptoms:
            features[i] = 1

    # Make prediction
    prediction = model.predict([features])[0]

    # Get confidence
    probabilities = model.predict_proba([features])[0]
    confidence = float(np.max(probabilities) * 100)

    # Decode disease
    disease = label_encoder.inverse_transform([prediction])[0]

    return disease, round(confidence, 2)
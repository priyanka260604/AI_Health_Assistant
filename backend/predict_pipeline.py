import os
from pyexpat import features
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "artifacts", "label_encoder.pkl")
SYMPTOM_COLUMNS_PATH = os.path.join(BASE_DIR, "artifacts", "symptom_columns.pkl")

# Load model
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
symptom_columns = joblib.load(SYMPTOM_COLUMNS_PATH)


def predict_disease(user_symptoms):
    """
    user_symptoms = ["fever", "cough", "headache"]
    """

    # Create feature vector
    features = np.zeros(len(symptom_columns))

    # Convert input to lowercase
    user_symptoms = [symptom.strip().lower() for symptom in user_symptoms]

    for i, symptom in enumerate(symptom_columns):
        if symptom.lower() in user_symptoms:
            features[i] = 1

# Predict disease
    prediction = model.predict([features])[0]

    # Get confidence
    probabilities = model.predict_proba([features])[0]
    confidence = float(max(probabilities) * 100)

    # Decode disease name
    disease = encoder.inverse_transform([prediction])[0]

    return disease, round(confidence, 2)

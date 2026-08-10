import numpy as np
import joblib
from pathlib import Path


# -----------------------------------------
# PATHS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"


# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


# -----------------------------------------
# GET EXACT FEATURES USED BY MODEL
# -----------------------------------------

MODEL_FEATURES = list(model.feature_names_in_)

# Normalize feature names for matching
NORMALIZED_FEATURES = {
    feature.strip().lower(): feature
    for feature in MODEL_FEATURES
}


# -----------------------------------------
# PREDICT DISEASE
# -----------------------------------------

def predict_disease(user_symptoms):

    # Clean user input
    user_symptoms = {
        symptom.strip().lower()
        for symptom in user_symptoms
        if symptom and symptom.strip()
    }

    # Create feature vector using EXACT model columns
    features = np.zeros(
        len(MODEL_FEATURES),
        dtype=int
    )

    # Match symptoms
    for i, feature in enumerate(MODEL_FEATURES):

        normalized_feature = feature.strip().lower()

        if normalized_feature in user_symptoms:
            features[i] = 1

    # Create DataFrame with correct feature names
    import pandas as pd

    feature_df = pd.DataFrame(
        [features],
        columns=MODEL_FEATURES
    )

    # Prediction
    prediction = model.predict(feature_df)[0]

    # Probability
    probabilities = model.predict_proba(feature_df)[0]

    # Top 3 predictions
    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        class_number = model.classes_[index]

        disease = label_encoder.inverse_transform(
            [class_number]
        )[0]

        confidence = float(
            probabilities[index] * 100
        )

        top_predictions.append({
            "disease": disease,
            "confidence": round(confidence, 2)
        })

    # Main prediction
    disease = label_encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "top_predictions": top_predictions
    }
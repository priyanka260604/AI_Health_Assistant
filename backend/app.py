print("Running Updated app.py")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import os

from predict_pipeline import predict_disease


app = FastAPI()


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request Model
# =========================

class Symptoms(BaseModel):
    symptoms: list[str]


# =========================
# Base Directory
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# =========================
# Load Disease Information
# =========================

INFO_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "disease_info.csv"
)

disease_info = pd.read_csv(INFO_PATH)

disease_info["Disease"] = (
    disease_info["Disease"]
    .astype(str)
    .str.strip()
)


# =========================
# Load Doctor Information
# =========================

DOCTOR_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "doctor_info.csv"
)

doctor_info = pd.read_csv(DOCTOR_PATH)

doctor_info["Disease"] = (
    doctor_info["Disease"]
    .astype(str)
    .str.strip()
)


# =========================
# Home
# =========================

@app.get("/")
def home():

    return {
        "message": "AI Health Assistant Backend",
        "version": "2.0",
        "status": "running"
    }


# =========================
# Prediction
# =========================

@app.post("/predict")
def predict(data: Symptoms):

    print("\n==============================")
    print("NEW PREDICTION REQUEST")
    print("==============================")

    print("Received symptoms:")
    print(data.symptoms)

    # Predict disease
    disease, confidence = predict_disease(
        data.symptoms
    )

    disease = str(disease).strip()

    print("Predicted Disease:", repr(disease))
    print("Confidence:", confidence)

    # =========================
    # Find Disease Information
    # =========================

    info = disease_info[
        disease_info["Disease"]
        .str.lower()
        == disease.lower()
    ]

    # =========================
    # Find Doctor
    # =========================

    doctor = doctor_info[
        doctor_info["Disease"]
        .str.lower()
        == disease.lower()
    ]

    print("Rows Found:", len(info))

    # =========================
    # If Disease Information
    # Not Found
    # =========================

    if info.empty:

        print("⚠️ Disease information not found")

        return {
            "predicted_disease": disease,
            "confidence": confidence,

            "description":
                "No description available.",

            "precautions": [],
            "diet": [],
            "workout": [],
            "medication": [],

            "doctor": {
                "name": "General Physician",
                "specialization": "General Medicine"
            }
        }

    # =========================
    # Doctor Defaults
    # =========================

    doctor_name = "General Physician"
    specialization = "General Medicine"

    if not doctor.empty:

        doctor_name = str(
            doctor.iloc[0]["Doctor"]
        )

        specialization = str(
            doctor.iloc[0]["Specialization"]
        )

    # =========================
    # Disease Row
    # =========================

    row = info.iloc[0]

    # =========================
    # Google Maps
    # =========================

    doctor_query = (
        doctor_name
        .replace(" ", "+")
    )

    google_maps = (
        f"https://www.google.com/maps/search/"
        f"{doctor_query}+near+me"
    )

    # =========================
    # Response
    # =========================

    return {

        "predicted_disease": disease,

        "confidence": confidence,

        "description":
            str(row["Description"]),

        "precautions":
            str(row["Precautions"]).split("|"),

        "diet":
            str(row["Diet"]).split("|"),

        "workout":
            str(row["Workout"]).split("|"),

        "medication":
            str(row["Medication"]).split("|"),

        "doctor": {

            "name":
                doctor_name,

            "specialization":
                specialization
        },

        "google_maps":
            google_maps
    }
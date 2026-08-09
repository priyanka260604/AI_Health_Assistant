print("Running Updated app.py")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import os

from predict_pipeline import predict_disease

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Symptoms(BaseModel):
    symptoms: list[str]


# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load disease information
INFO_PATH = os.path.join(BASE_DIR, "dataset", "disease_info.csv")
disease_info = pd.read_csv(INFO_PATH)
DOCTOR_PATH = os.path.join(BASE_DIR, "dataset", "doctor_info.csv")
doctor_info = pd.read_csv(DOCTOR_PATH)

@app.get("/")
def home():
    return {
        "message": "THIS IS THE UPDATED BACKEND",
        "version": "2.0"
    }


@app.post("/predict")
def predict(data: Symptoms):
    disease, confidence = predict_disease(data.symptoms)
    print("Predicted:", repr(disease))
    print("Confidence:", confidence)
    print(disease_info["Disease"].tolist())

    # Remove leading/trailing spaces from Disease column
    disease_info["Disease"] = disease_info["Disease"].astype(str).str.strip()
    

# Remove spaces from predicted disease
    disease = disease.strip()

    info = disease_info[
        disease_info["Disease"].str.lower() == disease.lower()
    ]
    doctor = doctor_info[
    doctor_info["Disease"].str.lower() == disease.lower()
    ]
    print(info)
    print("Predicted Disease:", repr(disease))
    print("CSV Diseases:", disease_info["Disease"].head().tolist())
    print("Rows Found:", len(info))

    if info.empty:
        return {
            "predicted_disease": disease,
            "confidence": confidence,
            "description": "No description available.",
            "precautions": [],
            "diet": [],
            "workout": [],
            "medication": [],
            "doctor": {
                "name": "General Physician",
                "specialization": "General Medicine"
            }
        }
    doctor_name = "General Physician"
    specialization = "General Medicine"

    if not doctor.empty:
        doctor_name = doctor.iloc[0]["Doctor"]
        specialization = doctor.iloc[0]["Specialization"]

    row = info.iloc[0]
    doctor_query = doctor_name.replace(" ", "+")
    google_maps = (
    f"https://www.google.com/maps/search/{doctor_query}+near+me"
    )

    return {
        "predicted_disease": disease,
        "confidence": confidence,
        "description": row["Description"],
        "precautions": str(row["Precautions"]).split("|"),
        "diet": str(row["Diet"]).split("|"),
        "workout": str(row["Workout"]).split("|"),
        "medication": str(row["Medication"]).split("|"),
        "doctor": {
            "name": doctor_name,
            "specialization": specialization
        },
        "google_maps": google_maps
    }
    
import pandas as pd
import joblib
import os

# Load dataset
df = pd.read_csv(r"C:\Users\priya\OneDrive\Desktop\AI Health Assistant\dataset\disease_dataset.csv")

# Get all symptom columns
symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]

# Collect all unique symptoms
all_symptoms = set()

for col in symptom_columns:
    values = df[col].dropna().astype(str).str.strip()

    for symptom in values:
        if symptom != "":
            all_symptoms.add(symptom)

all_symptoms = sorted(list(all_symptoms))

print(f"Total Symptoms = {len(all_symptoms)}")

# Create new dataframe
new_df = pd.DataFrame()

new_df["Disease"] = df["Disease"]

# Create binary columns
for symptom in all_symptoms:
    new_df[symptom] = 0

# Fill binary values
for index, row in df.iterrows():

    for col in symptom_columns:

        symptom = str(row[col]).strip()

        if symptom != "nan" and symptom != "":
            new_df.at[index, symptom] = 1

# Save processed dataset
os.makedirs(r"C:\Users\priya\OneDrive\Desktop\AI Health Assistant\artifacts", exist_ok=True)

new_df.to_csv(
    r"C:\Users\priya\OneDrive\Desktop\AI Health Assistant\artifacts\processed_dataset.csv",
    index=False
)

joblib.dump(
    all_symptoms,
    r"C:\Users\priya\OneDrive\Desktop\AI Health Assistant\artifacts\symptom_list.pkl"
)

print("✅ Dataset Processed Successfully")
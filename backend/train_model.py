import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "artifacts", "processed_dataset.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("Disease", axis=1)
y = df["Disease"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

joblib.dump(
    model,
    os.path.join(BASE_DIR, "artifacts", "model.pkl")
)

joblib.dump(
    encoder,
    os.path.join(BASE_DIR, "artifacts", "label_encoder.pkl")
)

joblib.dump(
    list(X.columns),
    os.path.join(BASE_DIR, "artifacts", "symptom_columns.pkl")
)

print("✅ Model Trained Successfully")
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# XGBoost
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

DATA_PATH = os.path.join(
    ARTIFACTS_DIR,
    "processed_dataset.csv"
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original rows:", len(df))


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

df = df.drop_duplicates().reset_index(drop=True)

print("Rows after removing duplicates:", len(df))


# ============================================================
# CHECK TARGET
# ============================================================

if "disease" not in df.columns:
    raise ValueError(
        "Disease column not found in dataset."
    )


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop(
    "disease",
    axis=1
)

y = df["disease"]


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(0)


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nNumber of diseases:", len(label_encoder.classes_))
print("Number of features:", X.shape[1])


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        C=1.0
    ),

    "SVM": SVC(
        probability=True,
        C=1.0,
        kernel="rbf",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# MODEL COMPARISON
# ============================================================

results = []

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


for name, model in models.items():

    print(f"\nTraining {name}...")

    scores = cross_val_score(
        model,
        X,
        y_encoded,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    mean_score = scores.mean()
    std_score = scores.std()

    print(
        f"{name}: "
        f"{mean_score * 100:.2f}% "
        f"(± {std_score * 100:.2f}%)"
    )

    results.append({
        "model": name,
        "mean_accuracy": mean_score,
        "std_accuracy": std_score
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_accuracy",
    ascending=False
)

print("\n")
print("=" * 60)
print("FINAL MODEL RANKING")
print("=" * 60)

print(results_df)


results_df.to_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["model"]

best_model = models[
    best_model_name
]

print(
    f"\nBest model: {best_model_name}"
)


# ============================================================
# TRAIN BEST MODEL
# ============================================================

print("\nTraining final model...")

best_model.fit(
    X,
    y_encoded
)


# ============================================================
# CALIBRATION
# ============================================================

print("\nCalibrating probabilities...")

calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(
    X,
    y_encoded
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    calibrated_model,
    os.path.join(
        ARTIFACTS_DIR,
        "model.pkl"
    )
)


# ============================================================
# SAVE LABEL ENCODER
# ============================================================

joblib.dump(
    label_encoder,
    os.path.join(
        ARTIFACTS_DIR,
        "label_encoder.pkl"
    )
)


# ============================================================
# SAVE SYMPTOM COLUMNS
# ============================================================

joblib.dump(
    list(X.columns),
    os.path.join(
        ARTIFACTS_DIR,
        "symptom_columns.pkl"
    )
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

clean_dataset_path = os.path.join(
    ARTIFACTS_DIR,
    "clean_dataset.csv"
)

clean_df = X.copy()

clean_df["Disease"] = y.values

clean_df.to_csv(
    clean_dataset_path,
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Model: {best_model_name}"
)

print(
    f"Cross-validation accuracy: "
    f"{results_df.iloc[0]['mean_accuracy'] * 100:.2f}%"
)

print(
    "Probability calibration: SIGMOID"
)

print(
    "\nSaved files:"
)

print("✓ model.pkl")
print("✓ label_encoder.pkl")
print("✓ symptom_columns.pkl")
print("✓ model_comparison.csv")
print("✓ clean_dataset.csv")

print("\n✅ DONE")
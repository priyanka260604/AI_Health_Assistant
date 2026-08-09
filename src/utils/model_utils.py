import time
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test
):

    results = []

    for name, model in models.items():

        start = time.time()

        model.fit(
            X_train,
            y_train
        )

        end = time.time()

        predictions = model.predict(
            X_test
        )

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "Precision": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "Training Time": end - start,

            "Model Object": model

        })

    results = pd.DataFrame(results)

    results = results.sort_values(

        by="Accuracy",

        ascending=False

    )

    return results
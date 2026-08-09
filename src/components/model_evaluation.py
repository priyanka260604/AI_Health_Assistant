import os
import sys
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.logger import logger
from src.exception import CustomException


@dataclass
class ModelEvaluationConfig:

    metrics_path = os.path.join(
        "artifacts",
        "metrics.json"
    )

    confusion_matrix_path = os.path.join(
        "artifacts",
        "confusion_matrix.png"
    )


class ModelEvaluation:

    def __init__(self):

        self.config = ModelEvaluationConfig()

    def evaluate_model(
        self,
        model_path,
        test_path
    ):

        try:

            logger.info("Starting Model Evaluation")

            model = joblib.load(model_path)

            test_df = pd.read_csv('C:\\Users\\priya\\OneDrive\\Desktop\\AI Health Assistant\\artifacts\\test_transformed.csv')

            X_test = test_df.drop(
                "Disease",
                axis=1
            )

            y_test = test_df["Disease"]

            predictions = model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            report = classification_report(
                y_test,
                predictions,
                zero_division=0
            )

            print(report)

            metrics = {

                "accuracy": float(accuracy),

                "precision": float(precision),

                "recall": float(recall),

                "f1_score": float(f1)

            }

            with open(
                self.config.metrics_path,
                "w"
            ) as f:

                json.dump(
                    metrics,
                    f,
                    indent=4
                )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            plt.figure(figsize=(12,10))

            sns.heatmap(
                cm,
                cmap="Blues"
            )

            plt.title(
                "Confusion Matrix"
            )

            plt.xlabel(
                "Predicted"
            )

            plt.ylabel(
                "Actual"
            )

            plt.savefig(
                self.config.confusion_matrix_path
            )

            plt.close()

            logger.info(
                "Evaluation Completed"
            )

            return metrics

        except Exception as e:

            raise CustomException(e, sys)
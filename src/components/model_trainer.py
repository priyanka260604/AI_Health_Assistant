import os
import sys
import joblib
import pandas as pd

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.logger import logger
from src.exception import CustomException
from src.utils.model_utils import evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_path = os.path.join(
        "artifacts",
        "model.pkl"
    )

    comparison_path = os.path.join(
        "artifacts",
        "model_comparison.csv"
    )


class ModelTrainer:

    def __init__(self):

        self.config = ModelTrainerConfig()

    def train_model(
        self,
        train_path,
        test_path
    ):

        try:

            logger.info("Starting Model Training")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop(
                "Disease",
                axis=1
            )

            y_train = train_df["Disease"]

            X_test = test_df.drop(
                "Disease",
                axis=1
            )

            y_test = test_df["Disease"]

            models = {

                "Random Forest":
                RandomForestClassifier(
                    random_state=42
                ),

                "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

                "Naive Bayes":
                GaussianNB(),

                "KNN":
                KNeighborsClassifier(),

                "SVM":
                SVC(),

                "XGBoost":
                XGBClassifier(
                    random_state=42,
                    eval_metric="mlogloss"
                )

            }

            results = evaluate_models(

                models,

                X_train,

                y_train,

                X_test,

                y_test

            )

            comparison = results.drop(
                columns=["Model Object"]
            )

            comparison.to_csv(

                self.config.comparison_path,

                index=False

            )

            best_model = results.iloc[0]["Model Object"]

            best_model_name = results.iloc[0]["Model"]

            best_accuracy = results.iloc[0]["Accuracy"]

            joblib.dump(

                best_model,

                self.config.trained_model_path

            )

            logger.info(
                f"Best Model : {best_model_name}"
            )

            logger.info(
                f"Accuracy : {best_accuracy:.4f}"
            )

            return {

                "Best Model": best_model_name,

                "Accuracy": best_accuracy

            }

        except Exception as e:

            raise CustomException(e, sys)
import os
import sys
import joblib
import pandas as pd

from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataTransformationConfig:

    transformed_train_path = os.path.join(
        "artifacts",
        "train_transformed.csv"
    )

    transformed_test_path = os.path.join(
        "artifacts",
        "test_transformed.csv"
    )

    symptom_list_path = os.path.join(
        "artifacts",
        "symptom_list.pkl"
    )

    label_encoder_path = os.path.join(
        "artifacts",
        "label_encoder.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    def transform_data(self, train_path, test_path):

        try:

            logger.info("Starting Data Transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            symptom_columns = [
                col for col in train_df.columns
                if col.startswith("Symptom_")
            ]

            # Collect all unique symptoms from both train and test
            all_symptoms = set()

            for df in [train_df, test_df]:

                for col in symptom_columns:

                    symptoms = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    all_symptoms.update(symptoms)

            all_symptoms = sorted(list(all_symptoms))

            logger.info(f"Total Unique Symptoms : {len(all_symptoms)}")

            # Save symptom list
            joblib.dump(
                all_symptoms,
                self.config.symptom_list_path
            )

            logger.info("Symptom List Saved")

            def encode_dataset(df):

                transformed_df = pd.DataFrame(
                    0,
                    index=df.index,
                    columns=all_symptoms
                )

                for index, row in df.iterrows():

                    patient_symptoms = (
                        row[symptom_columns]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .tolist()
                    )

                    for symptom in patient_symptoms:

                        if symptom in transformed_df.columns:

                            transformed_df.loc[index, symptom] = 1

                return transformed_df

            X_train = encode_dataset(train_df)
            X_test = encode_dataset(test_df)

            # Encode Disease Labels
            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(
                train_df["Disease"]
            )

            y_test = label_encoder.transform(
                test_df["Disease"]
            )

            # Save Label Encoder
            joblib.dump(
                label_encoder,
                self.config.label_encoder_path
            )

            logger.info("Label Encoder Saved")

            X_train["Disease"] = y_train
            X_test["Disease"] = y_test

            X_train.to_csv(
                self.config.transformed_train_path,
                index=False
            )

            X_test.to_csv(
                self.config.transformed_test_path,
                index=False
            )

            logger.info("Data Transformation Completed Successfully")

            return (
                self.config.transformed_train_path,
                self.config.transformed_test_path,
                self.config.symptom_list_path,
                self.config.label_encoder_path
            )

        except Exception as e:

            raise CustomException(e, sys)
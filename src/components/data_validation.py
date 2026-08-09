import os
import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException


class DataValidation:

    def validate_dataset(self, file_path):
        try:

            logger.info("Starting Data Validation")

            # 1. File Exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")

            # 2. Read Dataset
            df = pd.read_csv(file_path)

            logger.info("Dataset Loaded Successfully")

            # 3. Dataset Empty
            if df.empty:
                raise ValueError("Dataset is empty.")

            # 4. Required Columns
            required_columns = [
                "Disease",
                "Symptom_1",
                "Symptom_2",
                "Symptom_3",
                "Symptom_4",
                "Symptom_5",
                "Symptom_6",
                "Symptom_7",
                "Symptom_8",
                "Symptom_9",
                "Symptom_10",
                "Symptom_11",
                "Symptom_12",
                "Symptom_13",
                "Symptom_14",
                "Symptom_15",
                "Symptom_16",
                "Symptom_17"
            ]

            missing_columns = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing Columns: {missing_columns}"
                )

            logger.info("All Required Columns Found")

            # 5. Duplicate Rows
            duplicate_rows = df.duplicated().sum()

            logger.info(f"Duplicate Rows: {duplicate_rows}")

            # 6. Missing Disease Names
            if df["Disease"].isnull().sum() > 0:
                raise ValueError(
                    "Disease column contains missing values."
                )

            # 7. Count Missing Symptoms
            symptom_columns = [
                col for col in df.columns
                if "Symptom_" in col
            ]

            missing_symptoms = df[symptom_columns].isnull().sum().sum()

            logger.info(
                f"Total Missing Symptom Cells: {missing_symptoms}"
            )

            # 8. Number of Diseases
            total_diseases = df["Disease"].nunique()

            logger.info(
                f"Total Diseases: {total_diseases}"
            )

            # 9. Total Records
            logger.info(
                f"Total Records: {len(df)}"
            )

            logger.info("Data Validation Completed Successfully")

            return True

        except Exception as e:
            raise CustomException(e, sys)
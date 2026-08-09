import os   # to create folders
import sys  #exception handling

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logger.info("Entered Data Ingestion Method")

        try:

            df = pd.read_csv("dataset/disease_dataset.csv")

            logger.info("Dataset Loaded Successfully")

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logger.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
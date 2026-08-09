import joblib
import pandas as pd

from src.logger import logger
from src.exception import CustomException


class PredictionPipeline:

    def __init__(self):

        self.model = joblib.load("artifacts/model.pkl")

        self.symptom_list = joblib.load("artifacts/symptom_list.pkl")

        self.label_encoder = joblib.load("artifacts/label_encoder.pkl")

    def predict(self, symptoms):

        try:

            feature_vector = [0] * len(self.symptom_list)

            for symptom in symptoms:

                symptom = symptom.strip()

                if symptom in self.symptom_list:

                    index = self.symptom_list.index(symptom)

                    feature_vector[index] = 1

            input_df = pd.DataFrame(

                [feature_vector],

                columns=self.symptom_list

            )

            prediction = self.model.predict(input_df)

            disease = self.label_encoder.inverse_transform(

                prediction

            )[0]

            logger.info(f"Predicted Disease : {disease}")

            return disease

        except Exception as e:

            raise CustomException(e)
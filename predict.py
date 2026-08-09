from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

symptoms = [

    "itching",

    "skin_rash",

    "nodal_skin_eruptions"

]

prediction = pipeline.predict(symptoms)

print(prediction)
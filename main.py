from src.components.model_trainer import ModelTrainer
trainer = ModelTrainer()

train_transformed = "artifacts/train_transformed.csv"
test_transformed = "artifacts/test_transformed.csv"

result = trainer.train_model(
    train_transformed,
    test_transformed
)

print(result)
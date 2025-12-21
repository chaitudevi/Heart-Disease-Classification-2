
from fastapi import FastAPI
from src.models.predict import predict


app = FastAPI()

@app.post("/predict")
def predict_endpoint(input_data: dict):
    return predict(input_data)

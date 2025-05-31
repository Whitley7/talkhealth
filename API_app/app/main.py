from fastapi import FastAPI
from .schemas import TextRequest, PredictionResponse
from .model import predict

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: TextRequest):
    result = predict(request.text)
    return result
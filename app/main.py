from fastapi import FastAPI
from pydantic import BaseModel
from app.services.model_service import ModelService

app = FastAPI()

class TextRequest(BaseModel):
    text: str

model_service = ModelService()

@app.on_event("startup")
def startup_event():
    """Загружаем модель перед стартом сервиса"""
    model_service.load_model()

@app.post("/train/")
def train_model():
    """API для запуска обучения модели"""
    model_service.train_model()
    return {"message": "Model training started."}

@app.post("/predict/")
def predict(request: TextRequest):
    """API для предсказания токсичности"""
    prediction = model_service.predict(request.text)
    return {"text": request.text, "prediction": prediction}
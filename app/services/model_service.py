import torch
from fastapi import FastAPI
from pydantic import BaseModel
from app.models.toxicity_model import ToxicityModel  # Импортируем обновленный класс модели
from app.data.data_loader import prepare_data

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class ModelService:
    def __init__(self):
        self.model = ToxicityModel()
    
    def load_model(self, path: str = "saved_model"):
        """Загрузка сохраненной модели"""
        self.model.load_model(path)

    def train_model(self):
        """Обучение модели"""
        train_loader, val_loader, _ = prepare_data()
        self.model.train(train_loader, val_loader)

    def predict(self, text: str):
        """Предсказание токсичности текста"""
        return self.model.predict(text)

model_service = ModelService()

@app.on_event("startup")
def startup_event():
    """Загружаем модель перед запуском"""
    try:
        model_service.load_model()  # Пытаемся загрузить сохраненную модель
    except Exception as e:
        print(f"Failed to load model: {e}. Please train the model first.")

@app.post("/train/")
def train_model():
    """API для запуска обучения модели"""
    model_service.train_model()
    model_service.model.save_model("saved_model")
    return {"message": "Model training completed and saved."}

@app.post("/predict/")
def predict(request: TextRequest):
    """API для предсказания токсичности"""
    prediction = model_service.predict(request.text)
    return {"text": request.text, "prediction": prediction}
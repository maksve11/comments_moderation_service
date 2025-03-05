from fastapi import APIRouter
from app.services.model_service import ModelService

router = APIRouter()

model_service = ModelService()

@router.post("/predict/")
async def predict_comment(request: dict):
    text = request.get('text')
    if not text:
        return {"error": "Text is required"}
    prediction = model_service.predict(text)
    return {"prediction": prediction}
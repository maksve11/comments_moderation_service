import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from app.config import Config
import numpy as np
from scipy import stats

class ToxicityModel:
    def __init__(self):
        self.model_name = "unitary/toxic-bert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def load_model(self, path):
        """Загрузка сохраненной модели"""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        print(f"Model loaded from {path}")
        
        
    def train(self, train_loader, val_loader):
        """Функция обучения модели"""
        self.model.train()
        for epoch in range(Config.EPOCHS):
            train_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)

                labels = labels.view(-1)  

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()

                train_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)

            print(f"Epoch {epoch+1}/{Config.EPOCHS}, Loss: {train_loss/len(train_loader)}, Accuracy: {correct_predictions/total_predictions}")
            
            self.evaluate(val_loader)

    def evaluate(self, val_loader):
        """Функция для оценки на валидационном наборе"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)

                labels = labels.view(-1)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)

        print(f"Validation Accuracy: {correct_predictions / total_predictions}")

    def save_model(self, path):
        """Сохранение модели"""
        self.model.save_pretrained(path)
        
    def predict(self, text: str, threshold=0.6):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to("cpu") 

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  

            probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        prob_class_0 = probabilities[0]
        prob_class_1 = probabilities[1] 
        
        threshold = 0.51
        if prob_class_1 > threshold:
            label = "toxic"
        else:
            label = "non-toxic"

        return {
            "label": label,
        }
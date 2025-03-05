import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import random

class ToxicClassifier:
    def __init__(self, model_name="bert-base-uncased"):  
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()  

    def classify(self, text: str) -> int:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        return predicted_class 

class TextGenerator:
    def __init__(self, model_name="gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval() 

    def generate_text(self, prompt: str, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(inputs['input_ids'], max_new_tokens=max_length, num_return_sequences=1, 
                                      no_repeat_ngram_size=2, temperature=0.7)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

def generate_and_classify_comments(num_comments=1000):
    classifier = ToxicClassifier()
    text_generator = TextGenerator()

    comments = []

    for _ in range(num_comments // 2):
        toxic_prompt = "Напиши агрессивный, оскорбительный комментарий"
        toxic_comment = text_generator.generate_text(toxic_prompt) 
        toxic_class = classifier.classify(toxic_comment) 

        neutral_prompt = "Напиши конструктивный, нейтральный комментарий"
        neutral_comment = text_generator.generate_text(neutral_prompt) 
        neutral_class = classifier.classify(neutral_comment) 

        comments.append({"comment": toxic_comment, "toxic": toxic_class})
        comments.append({"comment": neutral_comment, "toxic": neutral_class})

    df = pd.DataFrame(comments)
    return df

df = generate_and_classify_comments()

df.to_csv('generated_comments.csv', index=False)
print(df.head())
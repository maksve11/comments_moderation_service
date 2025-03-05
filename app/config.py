# config.py

import os
import torch

class Config:
    BERT_MODEL_NAME = 'unitary/toxic-bert' 
    MAX_LEN = 512 
    BATCH_SIZE = 8 
    EPOCHS = 3 
    LEARNING_RATE = 2e-5  
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/aclImdb') 
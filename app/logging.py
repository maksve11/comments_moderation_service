import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = 'data/logs'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def setup_logger():
    logger = logging.getLogger('comments-moderation')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = RotatingFileHandler(f"{LOG_DIR}/app.log", maxBytes=10**6, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
import time
from app.logging import logger

class MetricsService:
    def __init__(self):
        self.start_time = None

    def start_timer(self):
        """Начинаем замер времени."""
        self.start_time = time.time()

    def log_duration(self, action_name):
        """Логируем продолжительность выполнения действия."""
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Action '{action_name}' took {duration:.4f} seconds.")
        else:
            logger.warning("Timer was not started.")
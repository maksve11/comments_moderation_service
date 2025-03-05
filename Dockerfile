# Dockerfile

FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install torch==2.2.0 transformers==4.30.0
RUN pip install tokenizers==0.13.3
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порт
EXPOSE 8000

# Запускаем FastAPI через Uvicorn
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
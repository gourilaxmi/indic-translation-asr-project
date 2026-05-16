FROM python:3.10-slim

# System dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Gradio port
EXPOSE 7860

# Set PYTHONPATH so app/ can import models/
ENV PYTHONPATH="/app/app:/app/models"

CMD ["python", "app/main.py"]

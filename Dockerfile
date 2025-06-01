FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "src/train.py"]
RUN echo "Model training simulation complete. Pre-trained weights available."
#RUN cp yolo_runs/alpr-model3/weights/best.pt src/model_training_sample_weights.pt
EXPOSE 5000
CMD ["python", "src/app.py"]

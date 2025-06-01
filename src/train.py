# train.py - YOLOv8 Training Script for ALPR

from ultralytics import YOLO
import os


# Load base YOLOv8 model (n = nano, s = small, m = medium)
model = YOLO("yolov8n.pt")  # You can replace with yolov8s.pt or a custom model

# Start training
model.train(
    data="data.yaml",        # Path to dataset config
    epochs=1,
    imgsz=640,
    batch=8,
    name="alpr-model",
    project="yolo_runs"
)

# Optional: Evaluate the model after training
metrics = model.val()
print("Validation results:", metrics)



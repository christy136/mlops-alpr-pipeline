import os
import cv2
import easyocr
from ultralytics import YOLO

def test_license_plate_detection():
    try:
        model = YOLO("src/pre-trained_weights.pt")
        reader = easyocr.Reader(['en'], gpu=False)

        img_path = 'src/CE5803BH.png'
        img = cv2.imread(img_path)
        results = model(img_path)
        boxes = results[0].boxes

        assert boxes is not None
        assert len(boxes) > 0

    except Exception as e:
        assert False, f"Test failed due to: {e}"

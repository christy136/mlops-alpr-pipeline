# prediction.py - Inference Script with EasyOCR for ALPR

from ultralytics import YOLO
import easyocr
import os
import cv2

def detect_license_plate(img_path='', model_path="pre-trained_weights.pt"):
    """
    Detect license plate from an image using YOLO and EasyOCR.

    Args:
        img_path (str): Path to the input image.
        model_path (str): Path to the trained YOLOv8 model weights.

    Returns:
        list of dict: Each dict contains cropped plate image (as numpy array) and OCR result.
    """
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    results = model(img_path)
    boxes = results[0].boxes
    img = cv2.imread(img_path)

    output = []

    for i, box in enumerate(boxes):
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cropped = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

        ocr_result = reader.readtext(cropped)
        plate_text = ocr_result[0][1] if ocr_result else "No text detected"

        output.append({
            "plate_index": i + 1,
            "cropped_image": cropped,
            "ocr_text": plate_text
        })

        # Save the cropped image with the detected text
        save_path = f"outputs/plate_{i+1}_{plate_text.replace(' ', '_')}.jpg"
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(save_path, cropped)

    return output

detect_license_plate(img_path='src/CE5803BH.png', model_path="src/pre-trained_weights.pt")

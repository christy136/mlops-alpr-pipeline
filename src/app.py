from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from ultralytics import YOLO
import easyocr
import cv2
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')

# Load model and OCR
model = YOLO("src/pre-trained_weights.pt")
reader = easyocr.Reader(['en'], gpu=False)

# Ensure folders exist
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return redirect(url_for('index'))

    # Save uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Run YOLO + OCR
    results = model(img_path)
    img = cv2.imread(img_path)
    predictions = []

    for i, box in enumerate(results[0].boxes):
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cropped = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        ocr_result = reader.readtext(cropped)
        text = ocr_result[-1][1] if ocr_result else "No text"

        # Save to static/outputs
        filename = f"plate_{i+1}_{text.replace(' ', '_')}.jpg"
        crop_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(crop_path, cropped)

        # Return web-safe path
        web_path = f"outputs/{filename}"
        predictions.append({'image': web_path, 'text': text})

    return render_template('result.html', predictions=predictions)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, render_template, request, jsonify
import cv2
import time
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

# --- Configuration and Constants ---
# Model files (Ensure these files are present in the directory)
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
K = 5000  # Distance calibration constant (focal length * avg_face_width / known_distance)
PADDING = 20 # Padding around the face crop

# --- Load Networks ---
try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    print("DNN models loaded successfully.")
except cv2.error as e:
    print(f"Error loading DNN models: {e}. Check if the model files are correctly named and located.")
    # In a production setting, you might raise an error or shut down gracefully.

# --- Core Prediction Logic (Updated for Multi-Face) ---

def get_prediction(frame):
    start_total_time = time.time()
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    # 1. Face Detection
    blob_face = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob_face)
    detections = faceNet.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])

    if not faceBoxes:
        return []  # Return empty list if no faces found

    all_results = []
    
    # 2. Iterate through ALL detected faces
    for faceBox in faceBoxes:
        start_analysis_time = time.time()
        
        # Crop face region with padding
        face = frame[max(0, faceBox[1]-PADDING):min(faceBox[3]+PADDING, frame.shape[0]), 
                     max(0, faceBox[0]-PADDING):min(faceBox[2]+PADDING, frame.shape[1])]
        
        if face.size == 0:
            continue
            
        # Distance calculation
        face_width_pixels = faceBox[2] - faceBox[0]
        distance_cm = K / face_width_pixels if face_width_pixels > 0 else 0

        # Create blob for Age/Gender classification
        blob_classify = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender Prediction
        genderNet.setInput(blob_classify)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age Prediction
        ageNet.setInput(blob_classify)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        end_analysis_time = time.time()
        
        face_result = {
            'gender': gender,
            'age': age,
            'distance': round(distance_cm, 2),
            'analysis_time_s': round(end_analysis_time - start_analysis_time, 3),
            'box': faceBox
        }
        all_results.append(face_result)
        
    end_total_time = time.time()
    
    # Add metadata about total processing time
    if all_results:
        metadata = {'total_processing_time_s': round(end_total_time - start_total_time, 3), 
                    'face_count': len(all_results)}
        # Prepend metadata to the list of results
        all_results.insert(0, metadata)

    return all_results

# --- Flask Routes ---

@app.route('/')
def index():
    # Requires an index.html template file in a 'templates' folder
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    
    if image is None:
        return jsonify({"error": "Failed to read image file"}), 500

    results = get_prediction(image)

    if results:
        # Returns a list of dictionaries (one for metadata, one for each face)
        return jsonify(results) 
    else:
        return jsonify({"error": "No faces detected in the image"}), 400

@app.route('/webcam', methods=['POST'])
def webcam_image():
    # Uses the same logic as upload_image to process the submitted file
    return upload_image()

if __name__ == '__main__':
    app.run(debug=True)
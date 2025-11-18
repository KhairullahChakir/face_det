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

# Model files (make sure to change to your correct model paths)
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def get_prediction(frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
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
        return None  # No faces detected

    # Take first face
    faceBox = faceBoxes[0]
    face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0]), max(0, faceBox[0]):min(faceBox[2], frame.shape[1])]
    face_width_pixels = faceBox[2] - faceBox[0]
    distance_cm = 5000 / face_width_pixels

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()

    genderNet.setInput(blob)
    genderPreds = genderNet.forward()

    age = ageList[agePreds[0].argmax()]
    gender = genderList[genderPreds[0].argmax()]

    return {
        'gender': gender,
        'age': age,
        'distance': round(distance_cm, 2),
        'time': round(time.time(), 3)
    }

@app.route('/')
def index():
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
    result = get_prediction(image)

    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No face detected"}), 400

@app.route('/webcam', methods=['POST'])
def webcam_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    result = get_prediction(image)

    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No face detected"}), 400

if __name__ == '__main__':
    app.run(debug=True)

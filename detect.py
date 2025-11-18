# Face Detection with Age, Gender, Distance, Time & Accuracy

import cv2
import os
import time
import argparse

# ---------------------------
# Functions
# ---------------------------
def highlightFace(net, frame, conf_threshold=0.7):
    # Convert frame to 3-channel BGR
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300,300), [104,117,123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1),(x2,y2),(0,255,0),2)
    return frameOpencvDnn, faceBoxes

# ---------------------------
# Argument parsing
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Path to image or folder")
args = parser.parse_args()

# ---------------------------
# Model files
# ---------------------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderList = ['Male','Female']

# ---------------------------
# Load networks
# ---------------------------
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20
K = 5000  # Distance calibration

# ---------------------------
# Counters
# ---------------------------
total_faces = 0
correct_gender = 0
sum_distance = 0
sum_time = 0

# ---------------------------
# Images
# ---------------------------
if args.image and os.path.isdir(args.image):
    image_files = [os.path.join(args.image,f) for f in os.listdir(args.image)
                   if f.lower().endswith(('.jpg','.png','.jpeg'))]
elif args.image:
    image_files = [args.image]
else:
    image_files = []

# ---------------------------
# Main loop
# ---------------------------
for img_path in image_files:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Skipped {img_path} (cannot read)")
        continue

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print(f"Skipped {img_path} (empty face region)")
        continue

    for faceBox in faceBoxes:
        start_time = time.time()

        face = frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),
                     max(0,faceBox[0]-padding):min(faceBox[2]+padding,frame.shape[1]-1)]
        if face.size == 0:
            print(f"Skipped {img_path} (empty face region)")
            continue

        # Distance
        face_width_pixels = faceBox[2]-faceBox[0]
        distance_cm = K / face_width_pixels

        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        end_time = time.time()
        face_time = end_time - start_time

        print(f"{img_path}: Gender={gender}, Age={age[1:-1]} yrs, Distance={distance_cm:.1f} cm, Time={face_time:.3f}s")

        # Accuracy based on filename
        filename_lower = os.path.basename(img_path).lower()
        true_gender = None
        if any(x in filename_lower for x in ['male','man','boy']):
            true_gender = 'Male'
        elif any(x in filename_lower for x in ['female','woman','girl']):
            true_gender = 'Female'
        if true_gender:
            total_faces += 1
            if gender == true_gender:
                correct_gender += 1

        sum_distance += distance_cm
        sum_time += face_time

        # Display
        cv2.putText(resultImg, f"{gender}, {age}, {distance_cm:.1f}cm",
                    (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Face Detection", resultImg)
    key = cv2.waitKey(1000) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# ---------------------------
# Final metrics
# ---------------------------
num_detected = total_faces
avg_distance = sum_distance / num_detected if num_detected > 0 else 0
avg_time = sum_time / num_detected if num_detected > 0 else 0
gender_acc = correct_gender / total_faces*100 if total_faces > 0 else 0

print("\n---------------------------")
print(f"Total Faces Detected: {num_detected}")
print(f"Average Distance: {avg_distance:.2f} cm")
print(f"Average Detection Time per Face: {avg_time:.3f} s")
print(f"Gender Prediction Accuracy: {gender_acc:.2f}%")
print("---------------------------\n")

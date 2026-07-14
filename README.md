# Deep Learning Age & Gender Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green.svg)](https://opencv.org/)

## 📖 Overview
The **Age and Gender Detection** project uses Deep Learning and Computer Vision to accurately identify the gender and approximate age of a person from a single image or a live webcam feed. 

It leverages pre-trained models by [Tal Hassner and Gil Levi](https://talhassner.github.io/home/projects/Adience/Adience-data.html) that were trained on the Adience dataset. Because estimating exact age from a single photo is highly subjective (due to lighting, makeup, and expression), this model treats age estimation as a **classification problem** predicting an age range bucket (e.g., `25-32 years`).

## ✨ Features
* **Image Inference**: Pass any image file to quickly get predictions.
* **Live Webcam Inference**: Process real-time video frames directly from your local webcam.
* **OpenCV DNN Module**: Fast and lightweight execution using OpenCV's deep neural network module.
* **Pre-trained Caffe Models**: Ready-to-use weights included.

## 🚀 Technologies Used
* **Python 3**
* **OpenCV** (cv2)
* **Caffe Models** (for Age & Gender classification)
* **TensorFlow pb Models** (for Face detection)
* **Flask** (via `app.py` for web-based inference)

## 📁 Project Structure
```text
face_det/
├── samples/                       # Sample images for testing
├── detect.py                      # CLI script for image and webcam detection
├── app.py                         # Flask web application for browser-based detection
├── opencv_face_detector.pbtxt     # Face detection network configuration
├── opencv_face_detector_uint8.pb  # Face detection weights
├── age_deploy.prototxt            # Age prediction network configuration
├── age_net.caffemodel             # Age prediction weights
├── gender_deploy.prototxt         # Gender prediction network configuration
├── gender_net.caffemodel          # Gender prediction weights
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

## 📸 Screenshots
*(Coming soon - Placeholder for sample face detection outputs)*

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KhairullahChakir/face_det.git
   cd face_det
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python argparse flask
   ```
   *(Note: A virtual environment is recommended)*

## 💻 Usage

### 1. Run on a Static Image
Place your image inside the project folder (or in the `samples/` directory) and run:
```bash
python detect.py --image samples/female10.jpg
```

### 2. Run on Live Webcam
Simply execute the script without arguments to automatically open your webcam feed:
```bash
python detect.py
```
*Press **Ctrl + C** in the terminal to stop execution.*

### 3. Run Web Interface (Flask)
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000` in your web browser to upload images via the web interface.

## 🔮 Future Improvements
* Improve accuracy by upgrading from Caffe models to modern PyTorch/TensorFlow architectures (e.g., EfficientNet).
* Dockerize the Flask application for easier deployment.
* Add bounding box tracking across video frames to reduce inference jitter on live video.

## 🤝 Contributing
Contributions are welcome! Please fork this repository and submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

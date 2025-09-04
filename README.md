# ML Object Detection: Pothole Detection (TensorFlow 1)

## Overview
This project was developed as part of my **Bachelor of Engineering final year project**. The system implements a **pothole detection pipeline** using **TensorFlow 1** and **OpenCV**, leveraging object detection techniques for real-time analysis of road surfaces from video footage. It detects potholes, calculates approximate distance, and provides a `WARNING!!!` alert if a pothole is in the vehicle’s path.

> **Note:** This repository is provided for **reference purposes only**. 
> It contains scripts and configurations from a 2019 TensorFlow 1 pothole detection project.
> The code may require outdated dependencies (Python 3.5, TensorFlow 1.x) and is not intended for direct use.

---

## Key Features
- **Video Input**: Supports webcam stream or pre-recorded video.
- **Object Detection**: Uses SSD MobileNet v1 pretrained model with custom pothole detection modifications.
- **Distance Estimation**: Calculates relative distance of potholes to the vehicle path.
- **Alerts**: Displays `WARNING!!!` when a pothole is directly in vehicle path.

---

## Folder Structure

    ml-object-detection-potholes-tf1/
    │
    ├── scripts/ # Pothole detection Python scripts
    │ └── object_detection_video.py
    ├── utils/ # Helper modules from TF Object Detection API
    │ ├── label_map_util.py
    │ └── visualization_utils.py
    ├── data/ # Label map and optional demo images
    │ ├── labelmap.pbtxt
    │ └── test_images/
    ├── requirements.txt # Required Python packages
    └── README.md

---

## Requirements
```cmd
tensorflow==1.15.0
opencv-python==3.4.3.18
numpy==1.16.4
pillow==6.2.1
matplotlib==2.2.3
Cython==0.29.10
contextlib2==0.5.5
lxml==4.2.5
jupyter==1.0.0
protobuf==3.6.1  # optional for .proto compilation
```
---

## Setup & Installation

> These steps are included for documentation and reference only. Running them may require legacy Python and TensorFlow versions.

1. **Install Python 3.5 (amd64)**  
2. **Create TensorFlow folder**: `C:\TensorFlow`  
3. **Install TensorFlow 1.x**:  
```cmd
pip install tensorflow==1.15.0
pip install numpy==1.16.4 pillow==6.2.1 matplotlib==2.2.3
pip install Cython==0.29.10 contextlib2==0.5.5 lxml==4.2.5
pip install jupyter==1.0.0
pip install opencv-python==3.4.3.18
```

4. **Download SSD MobileNet v1 COCO model** from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and extract it into:


5. **Clone this repository:**
```cmd
git clone https://github.com/siddhant-savant/ml-object-detection-potholes-tf1.git
cd ml-object-detection-potholes-tf1
```
in C:\TensorFlow\models\research\object_detection

6. **Run the object detection script (webcam or local video):**
```cmd
python scripts\object_detection_video.py
```
---

## Usage
**Webcam Input**
```cmd
cap = cv2.VideoCapture(0)
```

**Local Video Input**
```cmd
cap = cv2.VideoCapture('C:/path/to/video.mp4')
```

**Detection**

- Bounding boxes highlight potholes.

- Approximate distance displayed above each detected pothole.

- Displays WARNING!!! if pothole is directly in vehicle path.

## References

TensorFlow Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection

SSD MobileNet v1 Model Zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

LabelImg annotation tool: https://tzutalin.github.io/labelImg/Usage

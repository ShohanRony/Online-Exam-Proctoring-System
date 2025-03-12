# Online Exam Proctoring System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

An AI-powered automated proctoring system that monitors users via webcam and microphone during online examinations. This system uses computer vision and audio analysis to detect potential cheating behaviors, ensuring exam integrity in remote settings.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

### Vision-Based Monitoring

1. **Eye Tracking**
   - Detects left, right, or upward eye movements
   - Monitors user's gaze direction in real-time
   - Flags suspicious eye movements that may indicate cheating

2. **Mouth Opening Detection**
   - Records initial lip distances as baseline
   - Identifies instances of mouth opening during examinations
   - Helps detect verbal communication attempts

3. **Person Detection**
   - Counts individuals in the frame
   - Alerts when no one or more than one person is detected
   - Ensures only the authorized test-taker is present

4. **Head Pose Estimation**
   - Analyzes head orientation in 3D space
   - Determines if the user is looking away from the screen
   - Provides accurate assessment of attention focus

5. **Face Spoofing Detection**
   - Distinguishes between real faces and photos/videos
   - Prevents identity fraud during examinations
   - Uses liveness detection algorithms for authentication

### Audio-Based Monitoring (Planned)

- Voice activity detection
- Background noise analysis
- Conversation detection

## System Architecture

The system employs a modular architecture with the following components:

- **Face Detection Module**: Uses OpenCV's DNN for efficient and accurate face detection
- **Facial Landmark Module**: Implements TensorFlow-based models for precise facial feature tracking
- **Monitoring Modules**: Specialized components for eye tracking, mouth detection, head pose estimation, etc.
- **Alert System**: Generates warnings when suspicious activities are detected

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam
- Microphone (for audio features)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ShohanRony/Online-Exam-Proctoring-System.git
   cd Online-Exam-Proctoring-System
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Download the required models:
   ```bash
   # Create models directory if it doesn't exist
   mkdir -p models

   # Download models (links to be added)
   # ...
   ```

## Usage

### Running Individual Modules

Each module can be run independently for testing or specific monitoring needs:

```bash
# Eye tracking
python eye_tracker.py

# Mouth opening detection
python mouth_opening_detector.py

# Head pose estimation
python head_pose_estimation.py

# Face spoofing detection
python face_spoofing.py
```

### Integration with Exam Platforms

Documentation for integrating with common exam platforms will be provided in future updates.

## Technical Implementation

### Face Detection

The system implements OpenCV's DNN module for face detection, which provides better performance than traditional methods:

```python
# Example from face_detector.py
def get_face_detector(modelFile=None, configFile=None, quantized=False):
    if quantized:
        if modelFile == None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile == None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    else:
        if modelFile == None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model
```

### Facial Landmarks

A TensorFlow-based model replaces the traditional Dlib model for improved accuracy in facial landmark detection:

```python
# Example from face_landmarks.py (simplified)
def detect_marks(img, model, rects):
    # Implementation details
    # ...
    return marks
```

### Eye Tracking

The system uses contour detection on eye regions to locate eyeballs and determine gaze direction:

```python
# Example from eye_tracker.py (simplified)
def find_eyeball_position(end_points, cx, cy):
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1  # Looking left
    elif x_ratio < 0.33:
        return 2  # Looking right
    elif y_ratio < 0.33:
        return 3  # Looking up
    else:
        return 0  # Looking center
```

## Future Enhancements

1. **Integrated Dashboard**
   - Real-time monitoring interface for proctors
   - Statistical analysis of user behavior
   - Recording capabilities for review

2. **Audio Analysis**
   - Voice activity detection
   - Background noise classification
   - Multiple speaker detection

3. **Advanced Cheating Detection**
   - Object detection for unauthorized materials
   - Screen sharing detection
   - Behavioral pattern analysis

4. **Integration APIs**
   - LMS integration (Canvas, Moodle, etc.)
   - Custom webhooks for alerts
   - REST API for third-party applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Shohinur Pervez Shohan - [GitHub](https://github.com/ShohanRony)

Project Link: [https://github.com/ShohanRony/Online-Exam-Proctoring-System](https://github.com/ShohanRony/Online-Exam-Proctoring-System)
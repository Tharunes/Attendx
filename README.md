# ATTENDX

An intelligent face recognition-based attendance system that automates attendance tracking using computer vision and machine learning.

## 🧠 System Workflow

![System Workflow](https://github.com/Tharunes/Attendx/blob/main/Workflow/process_flow.png)

## Overview

ATTENDX is a comprehensive attendance management system that uses facial recognition technology to automatically track and record attendance. The system consists of three main modules: face capture for enrollment, model training, and real-time attendance recognition.

## Features

- **Face Capture & Enrollment**: Register new individuals by capturing and storing facial images
- **Machine Learning Model Training**: Train custom face recognition models using captured data
- **Real-time Attendance Tracking**: Automatic attendance marking with confidence threshold validation
- **Web-based Interface**: User-friendly web UI for system control and management
- **Data Export**: Attendance records saved to CSV with timestamps
- **Duplicate Prevention**: Ensures individuals aren't marked present multiple times per day

## System Architecture

The system follows a modular workflow:

1. **Web UI Dashboard** - Central control panel with options for:
   - Capture Faces
   - Train Model  
   - Mark Attendance
   - Stop Camera

2. **Face Capture Module**
   - Enter individual's name
   - Start webcam for live capture
   - Haar Cascade-based face detection
   - Save 5 face images per person to dataset folder
   - Error handling for capture failures

3. **Model Training Module**
   - Load existing face dataset
   - Train recognition model using captured images
   - Face detection and model optimization

4. **Attendance Recognition Module**
   - Load trained model
   - Real-time face detection via webcam
   - Confidence threshold validation (<70%)
   - Person identification and verification
   - Duplicate attendance prevention
   - CSV export with timestamps

## Technical Requirements

### Dependencies
- OpenCV (cv2) - Computer vision and webcam handling
- Haar Cascade Classifier - Face detection
- Machine Learning Framework - Model training and recognition
- CSV library - Data export functionality

### Hardware Requirements
- Webcam or camera device
- Sufficient storage for face dataset
- Processing power for real-time recognition

## Installation & Setup

1. Clone the repository
2. Install required dependencies
3. Ensure webcam is connected and functional
4. Run the main application to access Web UI

## Usage

### 1. Face Registration
- Select "Capture Faces" from Web UI
- Enter the person's name
- Position face in front of camera
- Press 's' to save 5 face images
- Images are stored in dataset folder

### 2. Model Training
- Select "Train Model" after capturing faces
- System will process all captured images
- Train recognition model with face data
- Model is saved for attendance recognition

### 3. Attendance Marking
- Select "Mark Attendance" from Web UI
- System starts webcam for live recognition
- Faces are detected and compared with trained model
- Only recognitions with >70% confidence are accepted
- Attendance is recorded with timestamp in CSV
- Prevents duplicate entries for same day

## Data Storage

- **Face Images**: Stored in organized dataset folders by person name
- **Attendance Records**: Exported to CSV files with columns:
  - Name
  - Date
  - Time
  - Confidence Score

## Error Handling

The system includes robust error handling for:
- Webcam connection failures
- Face detection errors
- Model training issues
- File I/O operations
- Duplicate attendance attempts

## Security & Privacy

- Face data is stored locally
- Confidence threshold prevents false positives
- Attendance verification prevents gaming the system
- Secure model training pipeline

## Future Enhancements

- Multiple camera support
- Advanced anti-spoofing measures
- Integration with existing HR systems
- Mobile application support
- Real-time notifications
- Advanced analytics and reporting

## Contributing

Feel free to contribute to ATTENDX by submitting issues, feature requests, or pull requests.

## License

This project is open source. Please refer to the LICENSE file for details.

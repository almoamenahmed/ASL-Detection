# ASL Hand Recognition

This project recognizes American Sign Language (ASL) letters in real-time using computer vision and machine learning. It features a user-friendly GUI built with Tkinter for performing tasks such as data collection, model training, and live inference.

---

## Features
- **Data Collection**: Capture images of hand gestures for each letter of the ASL alphabet.
- **Dataset Creation**: Automatically process collected data into a machine learning-ready dataset.
- **Model Training**: Train a Random Forest Classifier to recognize ASL letters.
- **Live Inference**: Perform real-time ASL recognition using webcam video.

---

## How It Works
1. **Hand Tracking**: Uses [Mediapipe](https://google.github.io/mediapipe/) to detect and track hand landmarks.
2. **Feature Extraction**: Captures the relative positions of the hand landmarks.
3. **Machine Learning**: Predicts the ASL letter using a trained Random Forest Classifier.
4. **GUI Interface**: Streamlines data collection, training, and live detection.

---

## Usage Instructions
1. **Start the Application**: Run gui.py to launch the Tkinter-based GUI.
2. **Data Collection**: Click Start Data Collection to capture images for each ASL letter. Follow the on-screen instructions to collect data for all letters.
3. **Create Dataset**: Click Create Dataset to process the collected images into a usable dataset.
4. **Train the Model**: Click Train Classifier to train the Random Forest Classifier on the dataset. Training accuracy will be displayed in the status log.
5. **Live ASL Detection**: Click Start Detection to begin real-time ASL recognition using your webcam.

---

## Setup and Installation

### **Requirements**
The project requires the following Python libraries:
- `opencv-python`
- `mediapipe`
- `numpy`
- `scikit-learn`
- `pillow`
- `tkinter` (comes pre-installed with Python on most platforms)

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/almoamenahmed/ASL-Detection.git
   cd ASL-Detection

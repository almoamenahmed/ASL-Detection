import os
import cv2
import mediapipe as mp
import pickle
import tkinter as tk
import threading

# Function to start dataset creation in a separate thread
def create_dataset(status_log):
    update_status(status_log, "Creating dataset...")
    threading.Thread(target=process_dataset, args=(status_log,), daemon=True).start()

# Main function to process the dataset
def process_dataset(status_log):
    try:
        # Initialize Mediapipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        DATA_DIR = './data'
        data = []
        labels = []

        for dir_ in os.listdir(DATA_DIR):
            label = ord(dir_) - 65  # Convert 'A' -> 0, 'B' -> 1, ...
            update_status(status_log, f"Processing folder: {dir_}")
            folder_path = os.path.join(DATA_DIR, dir_)

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    update_status(status_log, f"Failed to load image: {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                # Extract hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_min = min(lm.x for lm in hand_landmarks.landmark)
                        y_min = min(lm.y for lm in hand_landmarks.landmark)

                        features = []
                        for lm in hand_landmarks.landmark:
                            features.append(lm.x - x_min)
                            features.append(lm.y - y_min)

                        if len(features) == 42:  # Ensure 21 landmarks (x, y)
                            data.append(features)
                            labels.append(label)

                update_status(status_log, f"Processed image: {img_name}")

        # Save dataset to a file
        dataset = {'data': data, 'labels': labels}
        with open('data.pickle', 'wb') as f:
            pickle.dump(dataset, f)

        update_status(status_log, "Dataset creation complete!")

    except Exception as e:
        update_status(status_log, f"Error creating dataset: {str(e)}")

# Helper function to update the status log in the GUI
def update_status(status_log, message):
    status_log.insert(tk.END, message + "\n")
    status_log.see(tk.END)

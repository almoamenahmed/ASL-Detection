import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import threading

# Function to start the ASL detection process in a separate thread
def start_inference(cap, status_log, webcam_label):
    """
    Start the ASL detection process in a separate thread.
    """
    update_status(status_log, "Starting ASL detection...")
    threading.Thread(target=process_inference, args=(cap, status_log, webcam_label), daemon=True).start()

# Main function to handle inference
def process_inference(cap, status_log, webcam_label):
    try:
        # Load the trained model
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
        update_status(status_log, "Model loaded successfully!")

        # Initialize Mediapipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Label mapping
        labels_dict = {i: chr(65 + i) for i in range(26)}  # 0 -> 'A', 1 -> 'B', ...

        # Function to process webcam frames
        def capture_and_process():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip the frame horizontally for a mirrored view
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                        # Extract features for prediction
                        x_min = min(lm.x for lm in hand_landmarks.landmark)
                        y_min = min(lm.y for lm in hand_landmarks.landmark)

                        features = []
                        for lm in hand_landmarks.landmark:
                            features.append(lm.x - x_min)
                            features.append(lm.y - y_min)

                        if len(features) == 42:  # Ensure valid landmarks
                            prediction = model.predict([np.array(features)])
                            predicted_character = labels_dict[int(prediction[0])]

                            # Draw prediction on frame
                            h, w, _ = frame.shape
                            x1, y1 = int(x_min * w) - 10, int(y_min * h) - 60  # Move box 50 pixels higher
                            x2, y2 = x1 + 175, y1 + 40
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), -1)
                            cv2.putText(
                                frame,
                                f'Predicted: {predicted_character}',
                                (x1 + 5, y1 + 30),  # Keep text aligned with the box
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 255),
                                2
                            )

                # Update the GUI with the processed frame
                update_gui_with_frame(frame, webcam_label)

        # Start processing the frames
        capture_and_process()

    except FileNotFoundError:
        update_status(status_log, "Model file not found. Please train the model first.")
    except Exception as e:
        update_status(status_log, f"Error during ASL detection: {str(e)}")

# Function to update the GUI with the current webcam frame
def update_gui_with_frame(frame, webcam_label):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(img)

    webcam_label.img_tk = img_tk
    webcam_label.config(image=img_tk)

# Helper function to update the status log in the GUI
def update_status(status_log, message):
    status_log.insert(tk.END, message + "\n")
    status_log.see(tk.END)

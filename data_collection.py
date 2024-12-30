import os
import cv2
import tkinter as tk

# Global variables for data collection
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
current_letter_index = 0
collecting = False
capture_count = 0
capture_interval_ms = 50  # Interval between captures in milliseconds


def start_sequential_collection(cap, status_log, webcam_label):
    """
    Start the sequential data collection process for all letters.
    """
    global current_letter_index, collecting
    current_letter_index = 0
    collecting = True
    update_status(status_log, "Starting data collection...")
    collect_next_letter(cap, status_log, webcam_label)


def collect_next_letter(cap, status_log, webcam_label):
    """
    Prepare to collect data for the next letter in the alphabet.
    """
    global current_letter_index, collecting, capture_count

    if current_letter_index >= len(alphabet):
        update_status(status_log, "Data collection complete!")
        collecting = False
        return

    current_letter = alphabet[current_letter_index]
    capture_count = 0  # Reset the capture count for the new letter
    update_status(status_log, f"Prepare to gesture for letter: {current_letter}. Press SPACE to start.")

    # Bind the spacebar to start capturing images
    def spacebar_trigger(event):
        print(f"Spacebar triggered for letter {current_letter}")  # Debugging
        start_capturing_images(cap, current_letter, status_log, webcam_label)

    # Bind spacebar to the root window for reliability
    root = webcam_label.master.master  # Root window
    root.bind("<space>", spacebar_trigger)
    print("Spacebar binding set up.")  # Debugging


def start_capturing_images(cap, letter, status_log, webcam_label):
    """
    Start capturing images for the specified letter.
    """
    global capture_count
    print(f"Spacebar pressed for letter {letter}")  # Debugging
    root = webcam_label.master.master  # Root window
    root.unbind("<space>")  # Unbind the spacebar to avoid re-triggering
    update_status(status_log, f"Capturing images for letter: {letter}")
    capture_images(cap, letter, status_log, webcam_label)


def capture_images(cap, letter, status_log, webcam_label):
    """
    Capture 100 images for the given letter, saving them to the correct directory.
    """
    global capture_count, current_letter_index

    if capture_count >= 100:
        update_status(status_log, f"Finished capturing for letter: {letter}")
        current_letter_index += 1
        collect_next_letter(cap, status_log, webcam_label)
        return

    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Webcam read status: {ret}")  # Debugging
        if ret:
            frame = cv2.flip(frame, 1)
            data_dir = "./data"
            letter_dir = os.path.join(data_dir, letter)
            os.makedirs(letter_dir, exist_ok=True)
            img_path = os.path.join(letter_dir, f"{capture_count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Image saved: {img_path}")  # Debugging
            capture_count += 1
            update_status(status_log, f"{letter} {capture_count}/100")
        else:
            update_status(status_log, "Failed to capture frame. Retrying...")
    else:
        update_status(status_log, "Webcam is not open. Please check your webcam.")

    # Schedule the next frame capture after the defined interval
    webcam_label.after(capture_interval_ms, lambda: capture_images(cap, letter, status_log, webcam_label))


def update_status(status_log, message):
    """
    Update the status log in the GUI with the given message.
    """
    status_log.insert(tk.END, message + "\n")
    status_log.see(tk.END)

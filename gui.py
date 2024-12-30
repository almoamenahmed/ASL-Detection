import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import cv2
from data_collection import start_sequential_collection
from dataset_processing import create_dataset
from model_training import train_classifier
from inference import start_inference

# Function to stop the webcam and safely exit
def stop_webcam_and_exit(webcam_label, root):
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    root.destroy()

# Main GUI setup
def main():
    root = tk.Tk()
    root.title("ASL Hand Sign Recognition")
    root.geometry("1200x800")
    root.protocol("WM_DELETE_WINDOW", lambda: stop_webcam_and_exit(webcam_label, root))

    # Frames
    reference_frame = tk.Frame(root, width=300, height=800, bg="white")
    reference_frame.pack(side=tk.LEFT, fill=tk.Y)

    left_frame = tk.Frame(root, width=600, height=800, bg="black")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = tk.Frame(root, width=300, bg="white")
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Load reference images
    load_reference_images(reference_frame)

    # Webcam feed label
    global webcam_label
    webcam_label = tk.Label(left_frame, bg="black")
    webcam_label.pack(fill=tk.BOTH, expand=True)

    # Status log
    status_label = tk.Label(right_frame, text="Status Log", font=("Helvetica", 16, "bold"), bg="white")
    status_label.pack(pady=10)

    status_log = tk.Text(right_frame, height=10, width=40, state="normal", wrap=tk.WORD)
    status_log.pack()

    # Buttons
    create_buttons(right_frame, root, webcam_label, status_log)

    # Debugging: Bind spacebar directly to root
    root.bind("<space>", lambda event: print("Spacebar pressed (root binding)"))  # Debugging

    # Start webcam
    start_webcam(webcam_label)

    # Run the Tkinter main loop
    root.mainloop()


def load_reference_images(parent_frame):
    reference_label = tk.Label(parent_frame, text="ASL Reference Images", font=("Helvetica", 16, "bold"), bg="white")
    reference_label.pack(pady=10)

    reference_canvas = tk.Canvas(parent_frame, bg="white")
    scrollbar = tk.Scrollbar(parent_frame, orient="vertical", command=reference_canvas.yview)
    scrollable_frame = tk.Frame(reference_canvas, bg="white")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: reference_canvas.configure(scrollregion=reference_canvas.bbox("all"))
    )

    reference_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    reference_canvas.configure(yscrollcommand=scrollbar.set)

    reference_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    ref_folder = "./asl_reference_photos"
    if os.path.exists(ref_folder):
        for i, filename in enumerate(sorted(os.listdir(ref_folder))):
            img_path = os.path.join(ref_folder, filename)
            img = Image.open(img_path).resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)

            img_label = tk.Label(scrollable_frame, image=img_tk, bg="white")
            img_label.image = img_tk  # Prevent garbage collection
            img_label.grid(row=i // 2, column=i % 2, padx=10, pady=10)

def create_buttons(parent_frame, root, webcam_label, status_log):
    btn_collect = tk.Button(
        parent_frame,
        text="Start Data Collection",
        command=lambda: start_sequential_collection(cap, status_log, webcam_label),  # Pass cap explicitly
        width=20,
        height=2,
        bg="lightblue"
    )
    btn_collect.pack(pady=10)

    btn_create_dataset = tk.Button(
        parent_frame,
        text="Create Dataset",
        command=lambda: create_dataset(status_log),  # Pass status_log
        width=20,
        height=2,
        bg="lightyellow"
    )
    btn_create_dataset.pack(pady=10)

    btn_train = tk.Button(
        parent_frame,
        text="Train Classifier",
        command=lambda: train_classifier(status_log),  # Pass status_log
        width=20,
        height=2,
        bg="lightgreen"
    )
    btn_train.pack(pady=10)

    btn_infer = tk.Button(
        parent_frame,
        text="Start Detection",
        command=lambda: start_inference(cap, status_log, webcam_label),  # Pass cap explicitly
        width=20,
        height=2,
        bg="lightcoral"
    )
    btn_infer.pack(pady=10)

def create_status_log(parent_frame):
    status_label = tk.Label(parent_frame, text="Status Log", font=("Helvetica", 16, "bold"), bg="white")
    status_label.pack(pady=10)

    status_log = tk.Text(parent_frame, height=10, width=40, state="normal", wrap=tk.WORD)
    status_log.pack()

# Webcam feed management
# Webcam feed management
def start_webcam(webcam_label):
    global cap  # Declare cap as global
    cap = cv2.VideoCapture(0)

    def update_frame():
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)

                webcam_label.img_tk = img_tk
                webcam_label.config(image=img_tk)
            webcam_label.after(10, update_frame)

    update_frame()

# Entry point
if __name__ == "__main__":
    main()

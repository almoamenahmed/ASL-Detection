import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
import threading

# Function to start training in a separate thread
def train_classifier(status_log):
    update_status(status_log, "Training classifier...")
    threading.Thread(target=process_training, args=(status_log,), daemon=True).start()

# Main function to process model training
def process_training(status_log):
    try:
        # Load the dataset
        with open('data.pickle', 'rb') as f:
            dataset = pickle.load(f)

        data = np.array(dataset['data'])
        labels = np.array(dataset['labels'])

        # Split dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels
        )

        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)

        # Evaluate the model
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        update_status(status_log, f"Training complete! Accuracy: {accuracy:.2f}%")

        # Save the trained model
        with open('model.pickle', 'wb') as f:
            pickle.dump(model, f)

        update_status(status_log, "Model saved as 'model.pickle'.")

    except FileNotFoundError:
        update_status(status_log, "Dataset not found. Please create the dataset first.")
    except Exception as e:
        update_status(status_log, f"Error during training: {str(e)}")

# Helper function to update the status log in the GUI
def update_status(status_log, message):
    status_log.insert(tk.END, message + "\n")
    status_log.see(tk.END)

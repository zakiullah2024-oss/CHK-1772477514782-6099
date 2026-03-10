import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

# --- Settings ---
MODEL_PATH = 'crop_model.h5'
LABELS_PATH = 'class_indices.json'

print("Loading the trained model... Please wait.")

# Load the trained model
if not os.path.exists(MODEL_PATH):
    print("Error: Model not found. Please ensure 'train.py' has finished running.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

# Load the class labels
with open(LABELS_PATH, 'r') as f:
    class_names = json.load(f)
    # Convert string keys from JSON back to integers
    class_names = {int(k): v for k, v in class_names.items()}

def predict_disease(img_path):
    if not os.path.exists(img_path):
        print("Error: The specified image file was not found.")
        return

    # Image Preprocessing (matching the training format)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values

    # Make Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    disease_name = class_names[predicted_index]

    # Display Results
    print("\n" + "="*50)
    print(f"Image Path  : {img_path}")
    print(f"Prediction  : {disease_name}")
    print(f"Confidence  : {confidence:.2f}%")
    print("="*50 + "\n")
    # this is for print the those values
# --- Main Execution ---
if __name__ == "__main__":
    print("Ready for prediction!")
    path = input("Enter the path or name of the image (e.g., test_leaf.jpg): ")
    predict_disease(path.strip().replace('"', '').replace("'", ""))

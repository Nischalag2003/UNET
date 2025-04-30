import os
import numpy as np
import cv2
import tensorflow as tf
from model import build_unet

# Define image dimensions
H, W = 256, 256

# Define paths
MODEL_PATH = "files/model.h5"
INPUT_FOLDER = "img"
OUTPUT_FOLDER = "output"

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the model
model = build_unet((H, W, 3))
model.load_weights(MODEL_PATH)

# Function to read and preprocess image
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return np.expand_dims(x, axis=0)

# Function to save predicted mask
def save_mask(mask, save_path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask)

# Process all images in the input folder
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)  # Save with the same name

        # Read and predict
        image = read_image(input_path)
        pred_mask = model.predict(image)[0]
        pred_mask = (pred_mask > 0.5).astype(np.float32)

        # Save the predicted mask
        save_mask(pred_mask[:, :, 0], output_path)
        print(f"Saved: {output_path}")

print("Prediction complete. All masks saved in 'output' folder.")

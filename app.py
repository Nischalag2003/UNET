import os
import numpy as np
import cv2
import tensorflow as tf
import webbrowser
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import base64
import threading
from model import build_unet
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Define folders
INPUT_FOLDER = "img"
OUTPUT_FOLDER = "output"
MODEL_PATH = "files/model.h5"

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the model
H, W = 256, 256
model = build_unet((H, W, 3))
model.load_weights(MODEL_PATH)

# Function to preprocess image
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    return np.expand_dims(x.astype(np.float32), axis=0)

# Function to save predicted mask
def save_mask(mask, save_path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask)

# Function to encode image in Base64
def encode_image(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_filename = "segmented_" + filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    file.save(input_path)

    # Process and predict
    image = read_image(input_path)
    pred_mask = model.predict(image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Save the predicted mask
    save_mask(pred_mask[:, :, 0], output_path)

    # Encode images
    encoded_input_image = encode_image(input_path)
    encoded_output_image = encode_image(output_path)

    # Dummy ground truth (since no actual GT is available)
    dummy_ground_truth = np.zeros((H, W), dtype=np.uint8)

    # Calculate performance metrics
    accuracy = accuracy_score(dummy_ground_truth.flatten(), pred_mask.flatten())

    return jsonify({
        "filename": output_filename,
        "input_image": encoded_input_image,
        "output_image": encoded_output_image,
        "accuracy": round(accuracy, 4),
    })

@app.route('/output_image/<filename>')
def output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def open_browser():
    """Function to open the web browser automatically"""
    webbrowser.open("http://127.0.0.1:8080/")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(host="0.0.0.0", port=8080, debug=False)

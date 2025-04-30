import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix,
    accuracy_score, f1_score, jaccard_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- CONFIG ---
IMAGE_DIR = "files/images"
MASK_DIR = "files/masks"
MODEL_PATH = "files/model.h5"
IMAGE_SIZE = (256, 256)
MAX_SAMPLES = 100
NUM_SAMPLE_PLOTS = 5
OUTPUT_DIR = "performance_metrics"

# --- ENSURE OUTPUT DIRECTORY EXISTS ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
model = load_model(MODEL_PATH, compile=False)

# --- LOAD IMAGES AND MASKS ---
def load_data(image_dir, mask_dir, max_samples=None):
    X, Y = [], []
    filenames = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    if max_samples:
        filenames = random.sample(filenames, min(max_samples, len(filenames)))
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        image_id = os.path.splitext(filename)[0]
        mask_name = f"{image_id}_segmentation.png"
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue
        image = load_img(image_path, target_size=IMAGE_SIZE)
        mask = load_img(mask_path, target_size=IMAGE_SIZE, color_mode="grayscale")
        X.append(img_to_array(image) / 255.0)
        Y.append((img_to_array(mask) / 255.0 > 0.5).astype(np.uint8))
    return np.array(X), np.array(Y)

# --- PLOT SAMPLE PREDICTIONS ---
def show_sample_predictions(X, Y_true, Y_pred, num_samples=5):
    idxs = np.random.choice(len(X), num_samples, replace=False)
    for i, idx in enumerate(idxs):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(X[idx])
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(Y_true[idx].squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(Y_pred[idx].squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"sample_prediction_{i+1}.png"), dpi=300)
        plt.close()

# --- MAIN EVALUATION ---
if __name__ == "__main__":
    X_test, Y_test = load_data(IMAGE_DIR, MASK_DIR, max_samples=MAX_SAMPLES)
    Y_pred = model.predict(X_test, verbose=1)
    Y_pred_probs = Y_pred.squeeze()
    Y_pred_bin = (Y_pred_probs > 0.5).astype(np.uint8)

    # Flatten for metrics
    y_true = Y_test.flatten()
    y_pred_probs = Y_pred_probs.flatten()
    y_pred_bin = Y_pred_bin.flatten()

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, zero_division=1)
    iou = jaccard_score(y_true, y_pred_bin, zero_division=1)
    conf = confusion_matrix(y_true, y_pred_bin)
    roc_auc = auc(*roc_curve(y_true, y_pred_probs)[:2])
    ap = average_precision_score(y_true, y_pred_probs)

    # Save metrics to text file
    with open(os.path.join(OUTPUT_DIR, "metrics_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (Dice): {f1:.4f}\n")
        f.write(f"IoU (Jaccard): {iou:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision (AP): {ap:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    plt.figure()
    plt.plot(recall, precision, label=f'PR (AP = {ap:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=300)
    plt.close()

    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Histogram of Predicted Probabilities
    plt.figure()
    plt.hist(y_pred_probs, bins=50, color='purple')
    plt.title("Histogram of Predicted Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_probability_histogram.png"), dpi=300)
    plt.close()

    # Sample predictions
    show_sample_predictions(X_test, Y_test, Y_pred_bin, num_samples=NUM_SAMPLE_PLOTS)

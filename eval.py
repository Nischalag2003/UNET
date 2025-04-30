import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir

H, W = 256, 256

def read_image(path):
    """ Load and preprocess image """
    x = cv2.imread(path, cv2.IMREAD_COLOR)  
    x = cv2.resize(x, (W, H))
    ori_x = x.copy()
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x  

def read_mask(path):
    """ Load and preprocess mask """
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
    x = cv2.resize(x, (W, H))
    ori_x = x.copy()
    x = x / 255.0
    x = x.astype(np.int32)  
    return ori_x, x

def save_results(ori_x, ori_y, y_pred, save_image_path):
    """ Save original image, ground truth, and prediction side by side """
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  
    ori_y = np.concatenate([ori_y] * 3, axis=-1)  

    y_pred = np.expand_dims(y_pred, axis=-1)  
    y_pred = np.concatenate([y_pred] * 3, axis=-1)  

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred * 255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving results """
    create_dir("results")

    """ Load the model """
    with custom_object_scope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the test data """
    dataset_path = "C:/Users/jainh/Downloads/Skin-Lesion-Segmentation-in-TensorFlow-2.0-main/"
    _, _, (test_x, test_y) = load_data(dataset_path)

    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the image name """
        name = os.path.basename(x)

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        """ Predicting the mask """
        y_pred = model.predict(x, verbose=0)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1).astype(np.int32)

        """ Saving the predicted mask """
        save_image_path = f"results/{name}"
        save_results(ori_x, ori_y, y_pred, save_image_path)

        """ Flatten the arrays for metric calculations """
        y_flat, y_pred_flat = y.flatten(), y_pred.flatten()

        """ Calculate evaluation metrics """
        acc_value = accuracy_score(y_flat, y_pred_flat)
        f1_value = f1_score(y_flat, y_pred_flat, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y_flat, y_pred_flat, labels=[0, 1], average="binary")
        recall_value = recall_score(y_flat, y_pred_flat, labels=[0, 1], average="binary")
        precision_value = precision_score(y_flat, y_pred_flat, labels=[0, 1], average="binary")

        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ Calculate and display mean metrics """
    score = np.mean([s[1:] for s in SCORE], axis=0)
    print(f"Accuracy: {score[0]:.5f}")
    print(f"F1 Score: {score[1]:.5f}")
    print(f"Jaccard Index: {score[2]:.5f}")
    print(f"Recall: {score[3]:.5f}")
    print(f"Precision: {score[4]:.5f}")

    """ Save results to CSV """
    df = pd.DataFrame(SCORE, columns=["Image Name", "Accuracy", "F1 Score", "Jaccard Index", "Recall", "Precision"])
    df.to_csv("files/score.csv", index=False)

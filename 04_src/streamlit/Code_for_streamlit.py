
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from datetime import date
import tensorflow as tf
#from tensorflow import keras
from keras.utils import image_dataset_from_directory
import itertools
import streamlit as st
import pickle
import matplotlib.cm as cm
import json
from tensorflow.keras.models import load_model
import os

#------------------------------------------------------
#Load the dataset

#DATE= date.today().strftime("%Y_%m_%d")
DATASET_PATH_train = r"C:\Users\Alex\Documents\GitHub\may25_bds_plants\05_data\original_data\New Plant Diseases Dataset\New Plant Diseases Dataset\train"

DATASET_PATH_valid = r"C:\Users\Alex\Documents\GitHub\may25_bds_plants\05_data\original_data\New Plant Diseases Dataset\New Plant Diseases Dataset\valid"
IMAGE_SIZE = (128,128, 3)  # (height, width, channels)
vector_size_1D = IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]  # Flattened size for each image
batch_size = 64

train = image_dataset_from_directory(
    DATASET_PATH_train,  # Path to the dataset
    labels='inferred',  # Automatically infer labels from subdirectory names
    #label_mode='categorical',  # Use categorical labels
    image_size=IMAGE_SIZE[:2],  # Resize images to the specified size
    batch_size=batch_size,  # Number of images per batch
    seed=42,  # Random seed for reproducibility
    )

valid = image_dataset_from_directory(
    DATASET_PATH_valid,  # Path to the dataset
    labels='inferred',  # Automatically infer labels from subdirectory names
    #label_mode='categorical',  # Use categorical labels
    image_size=IMAGE_SIZE[:2],  # Resize images to the specified size
    batch_size=batch_size,  # Number of images per batch
    seed=42,  # Random seed for reproducibility
    )

class_names = train.class_names #Saves all class names in a list 
classes_healthy = [class_name for class_name in train.class_names if "healthy" in class_name.lower()]
classes_sick = [class_name for class_name in train.class_names if "healthy" not in class_name.lower()]


dataset = train #Choose the dataset to work with, e.g., train, valid, test_healthy, test_sick
dataset_valid = valid #Choose the dataset to work with, e.g., train, valid, test_healthy, test_sick

# Configuration for performance optimization
# This is used to optimize the performance of data loading and preprocessing
AUTOTUNE = tf.data.AUTOTUNE
# Apply performance optimizations to the datasets
dataset = dataset.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
dataset_valid = dataset_valid.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)

#------------------------------------------------------
# To load the model
BASE_DIR = r"C:\Users\Alex\Documents\GitHub\may25_bds_plants"
MODEL_DIR = os.path.join(BASE_DIR, "05_data", "Model")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "01_notebooks", "02_preprossing")
DATE = "2025_07_29"

MODEL_PATH = os.path.join(MODEL_DIR, f"model_{DATE}.keras")
HISTORY_PATH = os.path.join(NOTEBOOK_DIR, f"model_history_{DATE}.json")
LAYER_PATH = os.path.join(NOTEBOOK_DIR, "first_model_layers.json")

with open(LAYER_PATH, "r") as f:
    layers = json.load(f)
model = load_model(MODEL_PATH)
#------------------------------------------------------
# To calculate predictions and labels from the model

def get_predictions_and_labels(model, dataset):
    """
    Runs inference on the dataset and collects predictions, true labels, and images.
    """
    true_labels = []
    pred_labels = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        pred_labels.extend(np.argmax(preds, axis=-1))
        true_labels.extend(labels.numpy())

    return np.array(true_labels), np.array(pred_labels)


def generate_classification_report(y_true, y_pred):
    """
    Returns classification report as dictionary.
    """
    return classification_report(y_true, y_pred, output_dict=True)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Creates a normalized confusion matrix heatmap and returns the figure.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    return fig


def get_misclassified_images(images, y_true, y_pred):
    """
    Returns misclassified images and corresponding true/predicted labels.
    """
    misclassified_indices = np.where(y_true != y_pred)[0]
    return y_true[misclassified_indices], y_pred[misclassified_indices]


#------------------------------------------------------
# To plot Grad-CAM

def grad_cam(model, image, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image[None, ...], tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Resize and overlay
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image.shape[0], image.shape[1])).numpy()
    heatmap = np.squeeze(heatmap)
    heatmap_color = cm.jet(heatmap)[..., :3]
    overlay = heatmap_color * 0.4 + image / 255.0
    return np.clip(overlay, 0, 1), class_idx.numpy()

def get_sample_images(dataset, num_samples=4):
    images, labels = [], []
    for batch_images, batch_labels in dataset:
        for img, label in zip(batch_images, batch_labels):
            images.append(img.numpy())
            labels.append(label.numpy())
            if len(images) >= num_samples:
                return np.array(images), np.array(labels)
    return np.array(images), np.array(labels)
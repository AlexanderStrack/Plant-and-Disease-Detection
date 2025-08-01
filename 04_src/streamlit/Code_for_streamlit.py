import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.cm as cm
import re

# This file now only contains functions and does not load any data by itself.

def get_predictions_and_labels(model, dataset):
    """
    Runs inference on the dataset and collects predictions and true labels.
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
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


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


def grad_cam(model, image, layer_name):
    """
    Creates a Grad-CAM heatmap for a given image.
    """
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

    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image.shape[0], image.shape[1])).numpy()
    heatmap = np.squeeze(heatmap)
    heatmap_color = cm.jet(heatmap)[..., :3]
    overlay = heatmap_color * 0.4 + (image.astype("float32") / 255.0)
    return np.clip(overlay, 0, 1), class_idx.numpy()

def get_sample_images(dataset, num_samples=4):
    """
    Gets a few sample images and labels from a dataset.
    """
    images, labels = [], []
    for batch_images, batch_labels in dataset.take(1):
        for img, label in zip(batch_images, batch_labels):
            if len(images) < num_samples:
                images.append(img.numpy())
                labels.append(label.numpy())
    return np.array(images), np.array(labels)


def clean_label(label):
    label = re.sub(r"_\([^)]*\)", "", label)
    label = re.sub(r",_bell", "", label)
    return label

# Utility function to format class names for display etc. "Apple___Black_Spot" -> "Apple (Black Spot)"
def format_class_name(label):
    plant, disease = label.split("___")
    disease = disease.replace("_", " ")
    return f"{plant} ({disease})"
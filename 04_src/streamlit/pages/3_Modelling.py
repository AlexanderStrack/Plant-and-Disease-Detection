import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import glob
from PIL import Image
import os
import subprocess
import streamlit.components.v1 as components
import time
import socket
from datetime import datetime

# Corrected Import Order: Import utils first to set up the path
import utils
import Code_for_streamlit
from Code_for_streamlit import grad_cam, get_sample_images
import streamlit.components.v1 as components
# Other imports
import shap
from tensorflow.keras.layers import Conv2D
import tensorflow as tf

st.header("First Model attempt")
st.write(
    "This section is about the first model attempt. "
    "A simple convolutional neural network (CNN) is built to classify "
)
train, valid = utils.load_images()
class_names = [Code_for_streamlit.clean_label(name) for name in train.class_names]
train.class_names = [name.replace(' ', '_') for name in class_names]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Model Structure", "Training History", "Evaluation", "Grad-CAM", "SHAP", "TensorBoard"])

with tab1:
    st.subheader("Model Layers (Table View)")
    layer_path = utils.get_path('layers')
    try:
        with open(layer_path, "r") as f:
            layers = json.load(f)
        st.dataframe(pd.DataFrame(layers))
    except FileNotFoundError:
        st.error(f"Layer file not found: `{layer_path}`")

    #st.subheader("Load Model")
    #st.success("Model loaded successfully.")

with tab2:
    st.subheader("Training History")
    history_path = utils.get_path('history')
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        df_hist = pd.DataFrame(history)
        st.title("Loss Function")
        st.write("These function tell us how much the predicted output of the model differs from the actual output."
                 "For classification problems, the loss function is typically categorical crossentropy"
                 "(Negative Log Likelihood).")
        st.line_chart(df_hist[["loss", "val_loss"]])
        st.write(
            "### 📝 Summary of the Training History Plot\n\n"
            "- **Training loss** consistently decreases — indicating effective learning.\n"
            "- **Validation loss** stays relatively flat and fluctuates slightly — no real "
            "improvement after epoch 1.\n\n"
            "### ✅ Conclusion\n\n"
            "The model is learning well on the training set, but its performance on the "
            "validation set is stagnating. This is a sign of **early overfitting**. "
            "Consider using **early stopping, regularization**, or **more data "
            "augmentation** to improve generalization."
        )

        st.title("Accuracy Function")
        st.write("These function tell us how well the model is performing on the training and validation sets."
                 "For classification problems, the accuracy is the percentage of correct predictions.")
        st.line_chart(df_hist[["accuracy", "val_accuracy"]])
        st.write(
            "### 📝 Summary of the Accuracy Plot\n\n"
            "- **Training accuracy** increases steadily and reaches nearly 100%.\n"
            "- **Validation accuracy** improves initially and then plateaus slightly below "
            "90%.\n\n"
            "### ✅ Conclusion\n\n"
            "The model learns the training data very well, but **validation accuracy lags "
            "behind**, suggesting possible **overfitting**. Further tuning (e.g., dropout, "
            "data augmentation, early stopping) may help boost generalization."
        )
    except FileNotFoundError:
        st.error(f"History file not found: `{history_path}`")

with tab3:

    st.subheader("Evaluation on Validation Set")
    classification_report_path = utils.get_path('classification_report')
    try:
        with open(classification_report_path, "rb") as f:
            results = pickle.load(f)
        report = results['report']
    except FileNotFoundError:
        st.error(f"Classification report file not found: `{classification_report_path}`")

    df_report = pd.DataFrame(report).transpose()
    index_to_label = {i: Code_for_streamlit.format_class_name(label) for i, label in enumerate(train.class_names)}
    numeric_labels = [idx for idx in df_report.index if str(idx).isdigit()]
    df_report.rename(index={str(i): name for i, name in index_to_label.items()}, inplace=True)
    st.subheader("Classification Report")
    st.dataframe(df_report.style.format("{:.2f}"))



    st.subheader("Confusion Matrix")
    cm = results['confusion_matrix']
    formatted_class_names = [Code_for_streamlit.format_class_name(name) for name in train.class_names]
    all_keywords = sorted(set(name.split(" ")[0] for name in formatted_class_names))
    selected_keywords = st.multiselect("Filter Confusion Matrix by plant name:", all_keywords, default=["Apple", "Tomato"])
    # Filter logic – find indexes with matching keywords
    if selected_keywords:
        selected_indices = [
            i for i, name in enumerate(formatted_class_names)
            if any(keyword.lower() in name.lower() for keyword in selected_keywords)
        ]
        # Filtered matrix and labels
        filtered_cm = cm[np.ix_(selected_indices, selected_indices)]
        filtered_labels = [formatted_class_names[i] for i in selected_indices]
    else:
        filtered_cm = cm
        filtered_labels = formatted_class_names

    # --- Confusion Matrix Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(filtered_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=filtered_labels, yticklabels=filtered_labels, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Filtered Confusion Matrix")
    st.pyplot(fig)


with tab4:
    st.write("Visual explanation of model predictions using Grad-CAM")

    gradcam_path = os.path.normpath(utils.get_path('gradcam_images'))
    #gradcam_path = utils.get_path('gradcam_images')

    # UI
    st.title("Example for Grad-CAM-results for the first model")

    available_classes = Code_for_streamlit.get_class_names_from_files(gradcam_path)
    display_to_raw = Code_for_streamlit.get_class_names_from_files(gradcam_path)

    if not available_classes:
        st.warning("No Grad-CAM-Images found.")
    else:
        selected_display_name = st.selectbox("Choose a plant class:", list(display_to_raw.keys()))
        selected_class = display_to_raw[selected_display_name]

        image_paths = Code_for_streamlit.get_images_for_class(selected_class,gradcam_path)

        # Show images
        if image_paths:
            class_nice = selected_class.replace("___", " (").replace("_", " ") + ")"
            st.subheader(f"Grad-CAM for: {class_nice}")
            cols = st.columns(len(image_paths))
            for i, img_path in enumerate(image_paths):
                with cols[i]:
                    st.image(Image.open(img_path), caption=f"Example {i+1}", use_column_width=True)
        else:
            st.info("No Grad-CAM images found for this class.")

with tab5:
    st.subheader("SHAP Interpretability")

    shap_path = utils.get_path('shap_images')  # eg.. "04_src/images_shap/first_model_2025_07_29"

    all_files = [f for f in os.listdir(shap_path) if f.endswith(".png")]

    # Extract all classes, e.g., Tomato___Early_blight
    class_names = sorted(
        list(set("_".join(f.split("_")[:-2]) for f in all_files))
    )

    # Mapping: nice display name → file name
    display_to_raw = {
        cname.replace("___", " (").replace("_", " ") + ")": cname
        for cname in class_names
    }

    if not class_names:
        st.warning("No SHAP images found.")
    else:
        selected_display_name = st.selectbox("Choose a plant class:", list(display_to_raw.keys()))
        selected_class = display_to_raw[selected_display_name]

        # Get all associated files
        class_files = sorted([
            f for f in all_files if f.startswith(selected_class)
        ])

        # Group by img1, img2, ...
        image_groups = {}
        for f in class_files:
            group_key = f.split("_")[-2]  # z. B. img1, img2
            image_groups.setdefault(group_key, []).append(f)

        # Display
        class_nice = selected_class.replace("___", " (").replace("_", " ") + ")"
        st.subheader(f"SHAP für: {class_nice}")

        for group_id, filenames in sorted(image_groups.items()):
            cols = st.columns(2)
            overlay_img, original_img = None, None

            for f in filenames:
                path = os.path.join(shap_path, f)
                if "overlay" in f:
                    overlay_img = Image.open(path)
                elif "original" in f:
                    original_img = Image.open(path)

            # Display in adjacent columns
            if original_img:
                cols[1].image(original_img, caption=f"🖼️ Originalbild {group_id[-1]}", use_column_width=True)
            if overlay_img:
                cols[0].image(overlay_img, caption=f"🔶 SHAP-Overlay {group_id[-1]}", use_column_width=True)
            

# 🔧 CONFIG
BASE_LOG_DIR = "logs/image"
TENSORBOARD_PORT = 6006

# 🚀 Start TensorBoard
def start_tensorboard(log_dir, port=TENSORBOARD_PORT):
    try:
        command = [
            "tensorboard",
            "--logdir", log_dir,
            "--port", str(port),
            "--host", "localhost"
        ]
        subprocess.Popen(command)
        time.sleep(3)  # Give TensorBoard time to start
        st.success("✅ TensorBoard started.")
    except Exception as e:
        st.error(f"❌ Failed to start TensorBoard: {e}")

# 🔍 Find all log directories with event files
def get_all_valid_log_dirs(base_dir=BASE_LOG_DIR):
    valid_dirs = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                valid_dirs.append(root)
                break  # Found a valid file in this dir

    return sorted(set(valid_dirs))

# 🧠 Get latest event file info in a directory
def get_event_file_info(log_dir):
    for file in os.listdir(log_dir):
        if file.startswith("events.out.tfevents"):
            path = os.path.join(log_dir, file)
            return file, os.path.getmtime(path)
    return None, None

with tab6:
    # === STREAMLIT APP ===
    st.subheader("📊 TensorBoard Integration")

    valid_dirs = get_all_valid_log_dirs()

    if not valid_dirs:
        st.warning("⚠️ No valid TensorBoard log directories found.")
    else:
        selected_dir = st.selectbox(
            "🗂️ Select a log directory",
            valid_dirs[::-1],  # Show newest first
            format_func=lambda d: f"{d} ({datetime.fromtimestamp(get_event_file_info(d)[1]).strftime('%Y-%m-%d %H:%M:%S')})"
        )

        st.markdown(f"📂 **Selected directory:** `{selected_dir}`")

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🚀 Start TensorBoard"):
                start_tensorboard(selected_dir)

        with col2:
            event_file, file_time = get_event_file_info(selected_dir)
            if event_file:
                st.markdown(f"📝 Event file: `{event_file}`")
                st.markdown(f"🕒 Modified: `{datetime.fromtimestamp(file_time)}`")

        # 🔍 Embed the TensorBoard view
        st.markdown("---")
        st.markdown("### 📈 TensorBoard Preview")
        try:
            components.iframe(f"http://localhost:{TENSORBOARD_PORT}", height=800, scrolling=True)
        except:
            st.warning("⚠️ Could not embed TensorBoard. Make sure it is running.")

# --- Sidebar Configuration ---
st.sidebar.title("Table of Contents")
st.sidebar.info(
    "Select a page above to explore different aspects of the project."
)


import io
import itertools
from packaging import version
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow import keras
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import image_dataset_from_directory
import plotly.express as px
import re
import textwrap
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import Code_for_streamlit
from Code_for_streamlit import grad_cam, get_sample_images

@st.cache_data
def load_data():
    # Load the dataset from the Excel file
    df = pd.read_excel(
        r"C:\Users\Alex\Documents\GitHub\may25_bds_plants\05_data"
        r"\computed_data\plant_disease_dataset_analysis.xlsx"
    )
    return df


df = load_data()
DATE = date.today().strftime("%Y_%m_%d")

# Removes any characters after a comma or underscore in the 'plant' column
df["plant"] = df["plant"].str.replace(
    r"[,_].*", " ", regex=True
)
df["disease"] = df["disease"].str.replace(
    r"[_]", " ", regex=True
)
# -------------------------------------------------------------------------------------------


st.title("Plant recognition app")
st.write(
    "This dataset contains images of various plant species along with their "
    "labels. The goal is to build a model that can recognize these plants "
    "based on their images."
)
st.write(
    "The dataset consists of images of plants,"
    "each labeled with the species name. "
    "The images are stored in a directory structure where each subdirectory "
    "corresponds to a different plant species. "
    "The dataset is used to train a machine learning model to recognize and "
    "classify these plants based on their images."
)
st.sidebar.title("Table of contents")

main_pages = ["Exploration", "Data Vizualization", "Modelling", "Interpretation"]
main_page = st.sidebar.radio("Go to", main_pages)

if main_page == "Modelling":
    sub_page = st.sidebar.radio("Select a modelling step", ["First Model", "Advanced Model"])
else:
    sub_page = None


if main_page == "Exploration":
    st.header("Exploration")

    st.write("### Presentation of data")
    st.write("The dataset contains the following columns:")
    st.dataframe(df.head(5))
    st.write("The total number of images in the dataset is:", len(df))
    st.write("Different plant species in the dataset:", df['plant'].nunique())
    st.write("List of plant species in the dataset:")
    st.write(df['plant'].unique())

    fig = plt.figure(figsize=(12, 6))
    df['plant'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Distribution of Plant Species", fontsize=16)
    plt.xlabel("Plant Species", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Displaying the number of images per plant species:")
    grouped = (
        df.groupby(["plant", "disease"])
        .size()
        .reset_index(name="Number of images")
    )
    plant_options = sorted(grouped['plant'].unique())
    selected_plant = st.selectbox(
        "Select a plant species to show the number of file",
        options=plant_options,
        key='plant_selection'
    )
    filtered = grouped[grouped["plant"] == selected_plant]
    st.write(f"Number of images for {selected_plant}:")
    st.dataframe(filtered)   
    fig = plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x="plant",
        y="Number of images",
        hue="disease",
        data=filtered,
        palette="viridis"
    )
    plt.title(f"Number of images for {selected_plant}", fontsize=16)
    plt.xlabel("Plant Species", fontsize=14)
    plt.ylabel("Number of Images", fontsize=14)
    plt.tight_layout()
    sns.move_legend(
        ax,
        title="Disease",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize='large',
        title_fontsize='large'
    )
    st.pyplot(fig)

    st.write("### Distribution of perceptual brightness")
    st.write(
        "This section shows the distribution of perceptual brightness values "
        "for the images in the dataset."
    )

    df["perceptual_brightness"].value_counts().sort_values(ascending=False)
    # Define thresholds
    too_dark_threshold = 30
    too_bright_threshold = 220
    # Create histogram of perceptual brightness
    brightness_values = df["perceptual_brightness"].dropna()
    counts, bin_edges = np.histogram(brightness_values, bins=30)

    # Normalize the brightness values for color mapping
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    normalized = (
        bin_centers - bin_centers.min()
    ) / (bin_centers.max() - bin_centers.min())
    colors = [(v, v, v) for v in normalized]
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        bin_centers,
        counts,
        width=np.diff(bin_edges),
        color=colors,
        edgecolor='black',
        align='center'
    )
    # Add vertical lines for thresholds
    ax.axvline(too_dark_threshold,
               color='red',
               linestyle='--',
               label='Too Dark Threshold')
    ax.axvline(too_bright_threshold,
               color='orange',
               linestyle='--',
               label='Too Bright Threshold')

    ax.set_title("Distribution of Perceptual Brightness", fontsize=16)
    ax.set_xlabel("Perceptual Brightness", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    st.pyplot(fig)
    st.write(
        "The plot shows that the brightness of the images is generally "
        "well-distributed, with a few images being too dark or too bright. "
        "This information can be useful for preprocessing the images before "
        "training a model."
    )

elif main_page == "Data Vizualization":
    st.header("Data Visualization")
    st.write("### Presentation of data")
    
    IMAGE_SIZE = (128, 128, 3)  # (height, width, channels)
    # Flattened size for each image
    vector_size_1D = (
        IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]
    )
    batch_size = 64
    DATE = date.today().strftime("%Y_%m_%d")
    DATASET_PATH_train = (
        r"C:\Users\Alex\Documents\GitHub\may25_bds_plants\05_data"
        r"\original_data\New Plant Diseases Dataset"
        r"\New Plant Diseases Dataset\train"
    )

    DATASET_PATH_valid = (
        r"C:\Users\Alex\Documents\GitHub\may25_bds_plants\05_data"
        r"\original_data\New Plant Diseases Dataset"
        r"\New Plant Diseases Dataset\valid"
    )
        
    @st.cache_resource(show_spinner="load Imagaes ...")
    def load_images(train_path, valid_path):
        train = image_dataset_from_directory(
            train_path,  # Path to the dataset
            labels='inferred',  # Automatically infer labels from
                                # subdirectory names
            image_size=IMAGE_SIZE[:2],  # Resize images to the specified size
            batch_size=batch_size,  # Number of images per batch
            seed=42,  # Random seed for reproducibility
            )
        valid = image_dataset_from_directory(
            valid_path,  # Path to the dataset
            labels='inferred',  # Automatically infer labels from
                                # subdirectory names
            image_size=IMAGE_SIZE[:2],  # Resize images to the specified size
            batch_size=batch_size,  # Number of images per batch
            seed=42,  # Random seed for reproducibility
            )
        return train, valid

    def clean_label(label):
        label = re.sub(r"_\([^)]*\)", "", label)   # delete e.g. _(maize)
        label = re.sub(r",_bell", "", label)    # delete e.g. ,_bell
        return label

    train, valid = load_images(DATASET_PATH_train, DATASET_PATH_valid)

    # Saves all class names in a list
    class_names = [clean_label(name) for name in train.class_names] 
    train.class_names = class_names
    train.class_names = [name.replace(' ', '_') for name in train.class_names]
    
    if train:
        st.success("Trainings- and validationdata successfully loaded!")
        st.subheader("Example images from the training dataset")
        st.write(
            "Here are some example images from the training dataset. "
            "The images are displayed along with their corresponding labels."
        )
        st.write(
            f"The images are resized to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels and "
            f"are displayed in batches of {batch_size}."
        )
    
    images, labels = next(iter(train))  # Load first Batch

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        disease = class_names[labels[i]].split('___')[1]
        disease_clean = re.sub(r"[,_]", " ", disease)   
        axes[i].imshow(images[i].numpy().astype("uint8"))
        #axes[i].set_title(f"{class_names[labels[i]].split('___')[0]}\n{re.sub(r'[,_]', ' ', class_names[labels[i]].split('___')[1])}")
        axes[i].set_title(re.sub(r"[,_]", " ", class_names[labels[i]].split('___')[0]) + "\n" +
                          "\n".join(textwrap.wrap(re.sub(r"[,_]", " ", class_names[labels[i]].split('___')[1]), width=13)))
        axes[i].axis("off")

    st.pyplot(fig)
    #----------------------------------------------------------------------------
    st.subheader("Discover the different plant species")

    #Cleaning class names

    plant_options = sorted(set(name.split('___')[0].strip() for name in class_names))
    plant_select = st.selectbox("Pick a Plant Species",
                options=plant_options,
                key='plant_species_checkbox')
    #st.write(st.session_state['plant_species_checkbox'])
    disease_options = sorted(set(
        name.split('___')[1].strip() for name in class_names
        if st.session_state['plant_species_checkbox'] in name))
    disease_options = [re.sub(r"[,_]", " ", name) for name in disease_options]
    disease_select = st.selectbox("Choose the disease to display",
                 disease_options,
                 key='desease_selection')

    #-------------------------------------------------------------------
    # Display a image of the selected plant species and disease
    target_label = f"{plant_select}___{disease_select.replace(' ', '_')}"
    
    try:
        label_index = class_names.index(target_label)
    except ValueError:
        st.error(f"Label '{target_label}' not found.")
        st.stop()

    found_image = None
    for images, labels in train:
        for img, label in zip(images, labels):
            if label.numpy() == label_index:
                found_image = img.numpy().astype("uint8")
                break
        if found_image is not None:
            break

    # show image
    if found_image is not None:
        st.image(found_image, caption=target_label, use_column_width=True)
    else:
        st.warning("No image found for the selected label.")
        
elif main_page == "Modelling":
    if sub_page == "First Model":
        st.header("First Model attempt")
        # First model attempt
        st.write(
            "This section is about the first model attempt. "
            "A simple convolutional neural network (CNN) is built to classify "
        )        
        # -------- Paths --------
        BASE_DIR = r"C:\Users\Alex\Documents\GitHub\may25_bds_plants"
        MODEL_DIR = os.path.join(BASE_DIR, "05_data", "Model")
        NOTEBOOK_DIR = os.path.join(BASE_DIR, "01_notebooks", "02_preprossing")
        DATE = "2025_07_29"

        MODEL_PATH = os.path.join(MODEL_DIR, f"model_{DATE}.keras")
        HISTORY_PATH = os.path.join(NOTEBOOK_DIR, f"model_history_{DATE}.json")
        LAYER_PATH = os.path.join(NOTEBOOK_DIR, "first_model_layers.json")

        # -------- Streamlit --------
        if "First Model" in st.session_state.get("sub_page", "First Model"):
            st.header("🧪 First Model Attempt")
            st.markdown("This section presents the first version of a simple CNN model for image classification.")

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📚 Model Structure", "📈 Training History", "📉 Evaluation", "🔥 Grad-CAM", "🎯 SHAP", "🔍 TensorBoard"])

            # ---------- Tab 1: Layers ----------
            with tab1:
                st.subheader("Model Layers (Table View)")
                try:
                    with open(LAYER_PATH, "r") as f:
                        layers = json.load(f)
                    st.dataframe(pd.DataFrame(layers))
                except FileNotFoundError:
                    st.error(f"Layer file not found: `{LAYER_PATH}`")

                st.subheader("🔁 Load Model")
                try:
                    model = load_model(MODEL_PATH)
                    st.success("Model loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading model: {e}")

            # ---------- Tab 2: History ----------
            with tab2:
                st.subheader("Training History")
                try:
                    with open(HISTORY_PATH, "r") as f:
                        history = json.load(f)
                    df_hist = pd.DataFrame(history)
                    st.line_chart(df_hist[["loss", "val_loss"]])
                    st.line_chart(df_hist[["accuracy", "val_accuracy"]])
                except FileNotFoundError:
                    st.error(f"History file not found: `{HISTORY_PATH}`")

            # ---------- Tab 3: Evaluation ----------
            with tab3:
                st.subheader("Evaluation on Validation Set")
                y_true, y_pred = Code_for_streamlit.get_predictions_and_labels(model, Code_for_streamlit.dataset_valid)
                st.subheader("📋 Classification Report")
                report = Code_for_streamlit.generate_classification_report(y_true, y_pred)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.format(precision=2))

            # ---------- Tab 4: Grad-CAM ----------
            with tab4:
                st.write("Visual explanation of model predictions using Grad-CAM")

            if 'model' in locals():
                # 1. Get images
                images, labels = get_sample_images(Code_for_streamlit.dataset_valid, num_samples=4)

                # 2. Select last Conv2D layer
                conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
                if conv_layers:
                    target_layer = conv_layers[-1]
                else:
                    st.warning("No Conv2D layer found in model.")
                    target_layer = None

                # 3. Show Grad-CAMs
                if target_layer:
                    for i, image in enumerate(images):
                        overlay, pred_class = grad_cam(model, image, target_layer)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption=f"True Label: {labels[i]}", use_column_width=True)
                        with col2:
                            st.image(overlay, caption=f"Grad-CAM Prediction: {pred_class}", use_column_width=True)
            else:
                st.warning("Model not loaded.")

            # ---------- Tab 5: SHAP ----------
            with tab5:
                st.subheader("SHAP Interpretability")
                if 'model' in locals():
                    try:
                        dummy_images = np.random.rand(4, 128, 128, 3).astype(np.float32)
                        masker = shap.maskers.Image("inpaint_telea", dummy_images[0].shape)
                        explainer = shap.Explainer(model, masker)
                        shap_values = explainer(dummy_images, max_evals=500, outputs=shap.Explanation.argsort.flip[:3])
                        shap.image_plot(shap_values)
                        st.pyplot(bbox_inches='tight')
                    except Exception as e:
                        st.error(f"SHAP explanation failed: {e}")
                else:
                    st.warning("Model must be loaded for SHAP analysis.")

            # ---------- Tab 6: TensorBoard ----------
            with tab6:
                st.subheader("TensorBoard")
                st.markdown("ℹ️ Launch TensorBoard manually using:")
                st.code("tensorboard --logdir logs/image")
                st.markdown("[📊 Open TensorBoard in browser](http://localhost:6006)", unsafe_allow_html=True)
        
    elif sub_page == "Advanced Model":
        st.header("Advanced Model")
                



elif main_page == "Interpretation":
    st.header("Interpretation")
    # Interpretation of the model results
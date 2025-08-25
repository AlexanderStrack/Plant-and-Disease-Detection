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

st.header("First Model Attempt")
st.write(
    "This section is about the first model attempt. "
    "A simple convolutional neural network (CNN) is built to classify "
)
train, valid = utils.load_images()
class_names = [Code_for_streamlit.clean_label(name) for name in train.class_names]
train.class_names = [name.replace(' ', '_') for name in class_names]

tab1, tab2, tab3, tab4, tab5= st.tabs([
    "Model Structure", "Training History", "Evaluation", "Grad-CAM", "SHAP"])

with tab1:
    st.subheader("1. Model Layers (Table View)")
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
    st.subheader("2. Training History")
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
            "###  Summary of the Training History Plot\n\n"
            "- **Training loss** consistently decreases — indicating effective learning.\n"
            "- **Validation loss** stays relatively flat and fluctuates slightly — no real "
            "improvement after epoch 1.\n\n"
            "###  Conclusion\n\n"
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
            "###  Summary of the Accuracy Plot\n\n"
            "- **Training accuracy** increases steadily and reaches nearly 100%.\n"
            "- **Validation accuracy** improves initially and then plateaus slightly below "
            "90%.\n\n"
            "###  Conclusion\n\n"
            "The model learns the training data very well, but **validation accuracy lags "
            "behind**, suggesting possible **overfitting**. Further tuning (e.g., dropout, "
            "data augmentation, early stopping) may help boost generalization."
        )
    except FileNotFoundError:
        st.error(f"History file not found: `{history_path}`")

with tab3:

    st.subheader("3. Evaluation on Validation Set")
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
    st.markdown("""
    The **Confusion Matrix** below provides a comprehensive view of our model's classification 
    performance across different plant species. Each cell shows the proportion of predictions 
    for each true class, helping identify where the model excels and where it struggles.
    """)
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

    st.markdown("---")
    
    st.subheader(" How to Read the Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
         Matrix Structure:
        - **Rows (Y-axis)**: True Labels (actual plant species)
        - **Columns (X-axis)**: Predicted Labels (model predictions)
        - **Diagonal values**: Correct classifications
        - **Off-diagonal values**: Misclassifications
        """)
    
    with col2:
        st.markdown("""
         Color Interpretation:
        - **Dark Blue**: High values (0.8-1.0) - Strong performance
        - **Medium Blue**: Moderate values (0.4-0.7) - Average performance  
        - **Light Blue**: Low values (0.0-0.3) - Weak performance
        - **White/Near White**: Zero or minimal values
        """)
    


with tab4:
    st.write("Visual explanation of model predictions using Grad-CAM")

    gradcam_path = os.path.normpath(utils.get_path('gradcam_images'))
    

    # UI
    st.title("4. Example for Grad-CAM-results for the first model")

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

    st.markdown("---")
    
    st.subheader(" What These Visualizations Tell Us")
    
    # Key insights section
    st.markdown("""
    ### **Color Interpretation:**
    - **Bright/Warm colors **: High model attention - these regions are crucial for the prediction
    - **Dark/Cool colors **: Low model attention - these areas have minimal impact on classification
    - **Intensity**: The brighter the color, the more important that region is for the model's decision
    """)
    
    st.markdown("""
    ### **Model Behavior Analysis:**
    
     Positive Indicators:
    - The model focuses on **relevant biological features** (leaf structure, veins, edges)
    - **Consistent attention patterns** across similar samples
    - **Sharp boundaries** between important and unimportant regions
    - No focus on background noise or irrelevant elements
    
     Potential Concerns to Watch For:
    - Model focusing on background elements instead of the main subject
    - Inconsistent attention patterns for similar images
    - Attention scattered across irrelevant image regions
    """)
    
    st.markdown("""
    ### **What We Can Conclude:**
    
     **Biological Relevance**: The model correctly identifies botanically important features like:
    - Leaf morphology and shape characteristics
    - Vein patterns and structural details
    - Texture and surface features
    
     **Model Quality**: These visualizations suggest our model has learned to:
    - Distinguish between relevant and irrelevant image regions
    - Focus on discriminative features for plant classification
    - Ignore background distractions
    
     **Trust & Interpretability**: The clear focus on leaf structures increases confidence in the model's decisions
    """)
    
    st.info("""
     **Key Takeaway**: These Grad-CAM visualizations demonstrate that our model has successfully 
    learned to identify and focus on the most relevant botanical features for accurate plant classification, 
    rather than relying on spurious correlations or background elements.
    """)
    
    st.markdown("""
    ### **Technical Notes:**
    - Grad-CAM works by using gradients flowing into the final convolutional layer
    - The technique is class-specific: different classes may show different attention patterns
    - These visualizations help validate that the model's reasoning aligns with human expertise
    """)


with tab5:
    st.subheader("5. SHAP Interpretability")

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
                cols[1].image(original_img, caption=f" Original Image {group_id[-1]}", use_column_width=True)
            if overlay_img:
                cols[0].image(overlay_img, caption=f" SHAP-Overlay {group_id[-1]}", use_column_width=True)
    
    st.markdown("---")
    
    st.subheader(" Understanding SHAP Color Coding")
    
    # Color interpretation with visual indicators
    col_red, col_blue, col_neutral = st.columns(3)
    
    with col_red:
        st.markdown("""
         **Red Regions**
        - **Positive contribution** to the predicted class
        - These pixels **increase** the model's confidence
        - **Support** the final classification decision
        """)
    
    with col_blue:
        st.markdown("""
         **Blue Regions**  
        - **Negative contribution** to the predicted class
        - These pixels **decrease** the model's confidence
        - **Oppose** the final classification decision
        """)
    
    with col_neutral:
        st.markdown("""
         **Neutral/Green Areas**
        - **Minimal contribution** (near zero)
        - Neither support nor oppose the prediction
        - **Background** or less relevant features
        """)
    
    st.markdown("---")
    
    st.subheader(" What This SHAP Analysis Reveals")
    
    st.markdown("""
    ### **Pixel-Level Insights:**
    
     Detailed Attribution:
    - **Every pixel** receives a contribution score (positive, negative, or neutral)
    - **Quantitative measure** of each pixel's importance to the final prediction
    - **Additive property**: All pixel contributions sum to the difference between baseline and current prediction
    
     Biological Feature Analysis:
    - **Leaf texture patterns**: Red areas likely indicate important surface characteristics
    - **Edge definitions**: Sharp boundaries between leaf and background
    - **Vein structures**: Linear patterns that help distinguish plant species
    - **Color variations**: Natural pigmentation that aids classification
    """)
    
    st.markdown("""
    ### **Model Interpretation:**
    
     Positive Indicators in This Example:
    - Strong red regions on **central leaf areas** (positive contribution)
    - Blue regions primarily on **background/edges** (correctly identified as less relevant)
    - **Concentrated attribution** on botanically meaningful features
    - **Minimal scattered noise** in attribution patterns
    
     Classification Confidence:
    - **High-contrast attribution** suggests confident prediction
    - **Localized red regions** indicate specific discriminative features
    - **Clean separation** between positive and negative contributions
    """)
    


# --- Sidebar Configuration ---
st.sidebar.title("Table of Contents")
st.sidebar.info(
    "Select a page above to explore different aspects of the project."
)
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

# Corrected Import Order: Import utils first to set up the path
import utils
import Code_for_streamlit
from Code_for_streamlit import grad_cam, get_sample_images

# Other imports
import shap
from tensorflow.keras.layers import Conv2D
import tensorflow as tf


st.header("First Model attempt")
st.write(
    "This section is about the first model attempt. "
    "A simple convolutional neural network (CNN) is built to classify "
)        

#model = utils.load_keras_model()
#if not model:
#    st.stop()
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
        st.line_chart(df_hist[["loss", "val_loss"]])
        st.line_chart(df_hist[["accuracy", "val_accuracy"]])
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
    selected_keywords = st.multiselect("Filter Confusion Matrix by plant name:", all_keywords, default=[])
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
    class_names = [Code_for_streamlit.clean_label(name) for name in train.class_names]
    
    GRADCAM_PATH = utils.get_path('gradcam_images')

    # Finde alle verfügbaren Klassen (einmal aus Dateinamen extrahieren)
    @st.cache_data
    def get_class_names():
        all_files = glob.glob(os.path.join(GRADCAM_PATH, "*.jpg"))
        class_names = sorted(set(
            os.path.basename(f).rsplit("_img", 1)[0] for f in all_files
        ))
        return class_names

    # Bildpfade für bestimmte Klasse laden
    def get_images_for_class(class_name):
        pattern = os.path.join(GRADCAM_PATH, f"{class_name}_img*.jpg")
        return sorted(glob.glob(pattern))[:2]

    # UI
    st.title("Beispielhafte Grad-CAM-Ergebnisse")

    available_classes = get_class_names()
    selected_class = st.selectbox("Wähle eine Pflanzenklasse:", available_classes)

    image_paths = get_images_for_class(selected_class)

    # Anzeige
    if image_paths:
        st.subheader(f"Grad-CAM für: {selected_class.replace('___', ' (')})")
        cols = st.columns(len(image_paths))
        for i, img_path in enumerate(image_paths):
            with cols[i]:
                st.image(Image.open(img_path), caption=f"Beispiel {i+1}", use_column_width=True)
    else:
        st.info("Keine Grad-CAM-Bilder für diese Klasse gefunden.")

with tab5:
    st.subheader("SHAP Interpretability")
    

with tab6:
    st.subheader("TensorBoard")
    st.markdown("Launch TensorBoard manually using:")
    st.code("tensorboard --logdir logs/image")
    st.markdown("[Open TensorBoard in browser](http://localhost:6006)", unsafe_allow_html=True)

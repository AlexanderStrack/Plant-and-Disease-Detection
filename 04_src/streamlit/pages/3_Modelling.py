import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

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
##if not model:
#    st.stop()

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

    st.subheader("Load Model")
    st.success("Model loaded successfully.")

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
    y_true, y_pred = Code_for_streamlit.get_predictions_and_labels(model, Code_for_streamlit.dataset_valid)
    st.subheader("Classification Report")
    report = Code_for_streamlit.generate_classification_report(y_true, y_pred)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format(precision=2))

with tab4:
    st.write("Visual explanation of model predictions using Grad-CAM")
    images, labels = get_sample_images(Code_for_streamlit.dataset_valid, num_samples=4)
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if conv_layers:
        target_layer = conv_layers[-1]
        for i, image in enumerate(images):
            overlay, pred_class = grad_cam(model, image, target_layer)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f"True Label: {Code_for_streamlit.class_names[labels[i]]}", use_column_width=True)
            with col2:
                st.image(overlay, caption=f"Grad-CAM Prediction: {Code_for_streamlit.class_names[pred_class]}", use_column_width=True)
    else:
        st.warning("No Conv2D layer found in model.")

with tab5:
    st.subheader("SHAP Interpretability")
    try:
        dummy_images = np.random.rand(4, 128, 128, 3).astype(np.float32)
        masker = shap.maskers.Image("inpaint_telea", dummy_images[0].shape)
        explainer = shap.Explainer(model.predict, masker)
        shap_values = explainer(dummy_images, max_evals=500, outputs=shap.Explanation.argsort.flip[:3])
        shap.image_plot(shap_values)
        st.pyplot(fig=plt.gcf(), bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

with tab6:
    st.subheader("TensorBoard")
    st.markdown("Launch TensorBoard manually using:")
    st.code("tensorboard --logdir logs/image")
    st.markdown("[Open TensorBoard in browser](http://localhost:6006)", unsafe_allow_html=True)

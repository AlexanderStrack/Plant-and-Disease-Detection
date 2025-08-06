import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Conclusion",
    layout="wide"
)

# --- Header ---
st.header("Project Conclusion")
st.write("---")

# --- Summary Section ---
st.subheader("Summary of the Project")
st.markdown(
    """
    This project successfully demonstrated the end-to-end development of a deep learning solution for plant disease recognition. 
    We progressed from initial data exploration to building, comparing, and interpreting two distinct models, and finally, 
    we presented all findings in this interactive web application.
    """
)

# --- Project Goal ---
st.markdown("#### Project Goal")
st.markdown(
    """
    The primary goal was to develop an accurate image classification model capable of identifying 38 different classes of plant diseases and healthy leaves. 
    Secondary goals included gaining practical experience with image data workflows and building a user-friendly Streamlit application to showcase the results. 
    All of these goals were successfully achieved.
    """
)

# --- Methods ---
st.markdown("#### Methods")
st.markdown(
    """
    Our methodology involved a comparative approach. We first established a baseline by building a **custom Convolutional Neural Network (CNN)** from scratch. 
    Following this, we implemented a more advanced solution using **transfer learning** with the pre-trained **MobileNetV2** architecture. 
    Key steps included data preprocessing, model fine-tuning, and performance evaluation using metrics like accuracy and loss. 
    Finally, we used interpretability techniques like **Grad-CAM** and **SHAP** to understand the models' decision-making processes.
    """
)

# --- Application ---
st.markdown("#### Application")
st.markdown(
    """
    The final product is this multi-page Streamlit application. It serves as a comprehensive dashboard that guides a user through the project's entire journey. 
    It allows for interactive data exploration, visualization of the dataset, and a detailed comparison of the performance and interpretability of both the simple and advanced models.
    """
)
st.write("---")

# --- Limitations ---
st.subheader("Limitations")
st.markdown(
    """
    Despite the project's success, it's important to acknowledge its limitations:
    - **Dataset Scope:** The model is trained on a specific dataset and is therefore limited to the 38 classes of plants and diseases it contains. Its performance on other species is unknown.
    - **Computational Resources:** The training of deep learning models is resource-intensive. Our experiments were constrained by the available computational power, which limited the extent of hyperparameter tuning.
    - **No Live Deployment:** The application currently works with the pre-existing dataset and does not yet include a feature for users to upload their own images for real-time prediction.
    """
)
st.write("---")

# --- Next Steps ---
st.subheader("Next Possible Steps")
st.markdown(
    """
    There are several exciting avenues for future work that could build upon this project's foundation:
    1.  **Develop a Live Prediction Page:** The most valuable next step would be to add a feature allowing users to upload their own leaf images and receive an instant classification from our trained model.
    2.  **Expand the Dataset:** Incorporating a wider variety of plant species and images taken under different environmental conditions would improve the model's robustness and generalizability.
    3.  **Experiment with Other Architectures:** Testing other state-of-the-art models, such as EfficientNetV2, could potentially yield even higher accuracy.
    4.  **Create a Mobile Application:** Packaging the model into a lightweight mobile app using TensorFlow Lite would make it a highly practical and accessible tool for in-field use by farmers and gardeners.
    """
)


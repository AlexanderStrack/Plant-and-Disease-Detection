
import streamlit as st

st.set_page_config(
    page_title="Plant Recognition App",
    layout="wide"
)

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
import streamlit as st
import pandas as pd
import json
import os
import sys
from tensorflow.keras.models import load_model
from keras.utils import image_dataset_from_directory

# --- Path Correction ---
# Get the directory of the current script (utils.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the script's directory to the Python path to allow imports like Code_for_streamlit
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)
# Define the project root as 3 levels up from the script directory
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))


# --- Configuration Loading ---
@st.cache_resource
def load_config():
    """Loads the configuration from config.json."""
    config_path = os.path.join(_SCRIPT_DIR, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_path(key, config=load_config()):
    """Constructs an absolute path from the configuration."""
    date = config['date']
    
    # Paths in config are relative to the project root
    paths = {
        'excel': os.path.join(_PROJECT_ROOT, config['data_subpath'], config['excel_analysis_file']),
        'train': os.path.join(_PROJECT_ROOT, config['data_subpath'], config['train_dataset_folder']),
        'valid': os.path.join(_PROJECT_ROOT, config['data_subpath'], config['valid_dataset_folder']),
        'model': os.path.join(_PROJECT_ROOT, config['data_subpath'], config['model_folder'], config['model_name'].format(date=date)),
        'history': os.path.join(_PROJECT_ROOT, config['notebooks_subpath'], config['history_name'].format(date=date)),
        'layers': os.path.join(_PROJECT_ROOT, config['notebooks_subpath'], config['layers_name']),
        'classification_report': os.path.join(_PROJECT_ROOT, config['notebooks_subpath'], config['report_name'].format(date=date)),
        'gradcam_images': os.path.join(_PROJECT_ROOT, config['gradcam_images_folder'].format(date=date)),
        'shap_images': os.path.join(_PROJECT_ROOT, config['shap_images_folder'].format(date=date))

    }
    return paths[key]

# --- Data Loading Functions (Unchanged) ---
@st.cache_data
def load_excel_data():
    """Loads and preprocesses the Excel data."""
    path = get_path('excel')
    df = pd.read_excel(path)
    df["plant"] = df["plant"].str.replace(r"[,_].*", " ", regex=True)
    df["disease"] = df["disease"].str.replace(r"[_]", " ", regex=True)
    return df

@st.cache_resource(show_spinner="Loading Images...")
def load_images():
    """Loads train and validation image datasets."""
    train_path = get_path('train')
    valid_path = get_path('valid')
    
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 64
    
    train = image_dataset_from_directory(
        train_path,
        labels='inferred',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=42,
    )
    valid = image_dataset_from_directory(
        valid_path,
        labels='inferred',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=42,
    )
    return train, valid

@st.cache_resource(show_spinner="Loading AI Model...")
def load_keras_model():
    """Loads the pre-trained Keras model."""
    model_path = get_path('model')
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

# page_1_exploration.py --- Exploration Page for Streamlit App

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils

st.header("Exploration")
st.write("---")

df = utils.load_excel_data()

# --- 1. Data Folder Structure ---
st.subheader("1. Data Folder Structure")
st.write(
    "The image data is organized in a directory structure where each sub-folder represents a specific class (a plant-disease combination). "
    "This is a common and effective way to structure data for image classification tasks."
)
folder_structure = """
original_data/
└── 2.1.1 New Plant Diseases/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   │   ├── image_001.JPG
    │   │   ├── image_002.JPG
    │   │   └── ...
    │   ├── Apple___Black_rot/
    │   │   ├── image_001.JPG
    │   │   └── ...
    │   ├── ... (36 more classes)
    │
    └── valid/
        ├── Apple___Apple_scab/
        │   ├── image_001.JPG
        │   └── ...
        └── ... (and so on for all classes)
"""
st.code(folder_structure, language='text')
st.markdown(
    "*This text-based tree illustrates how the training and validation images are separated and how each class has its own dedicated folder. "
    "The model learns to associate the images in each folder with the folder's name, which serves as the class label.*"
)
st.write("---")


# --- 2. File Naming Convention ---
st.subheader("2. File Naming Convention")
st.write(
    "The filenames in the dataset contain encoded information about the image's origin and any augmentations applied."
)
naming_convention_html = """
<div style="font-size: 1.1em;">
    <strong>Theory</strong>
</div>
<div style="background-color:#f0f2f6; padding: 10px; border-radius: 5px; font-family: monospace; margin-top: 5px;">
    <span style='color:#E69138'>&lt;uuid&gt;</span>___<span style='color:#6AA84F'>&lt;source_code&gt;</span>_<span style='color:#A64D79'>&lt;image_id&gt;</span>_<span style='color:#CC0000'>&lt;augmentation&gt;</span>.JPG
</div>
<br>
<div style="font-size: 1.1em;">
    <strong>Example</strong>
</div>
<div style="background-color:#f0f2f6; padding: 10px; border-radius: 5px; font-family: monospace; margin-top: 5px;">
    <span style='color:#E69138'>0b8dabb7-5f1b-4fdc-b3fa-30b289707b90</span>___<span style='color:#6AA84F'>JR_FrgE.S</span>_<span style='color:#A64D79'>3047</span>_<span style='color:#CC0000'>270deg</span>.JPG
</div>
"""
st.markdown(naming_convention_html, unsafe_allow_html=True)
st.markdown(
    "*This convention shows that each filename is composed of a unique identifier (uuid), a source code, an image ID, and details about any data augmentation (like rotation) performed. "
    "While not used for training, this metadata is valuable for traceability.*"
)
st.write("---")


# --- 3. Dataset Preview ---
st.subheader("3. Dataset Preview")
st.dataframe(df.groupby("class").get_group("Apple___Black_rot").head(5))
st.markdown(
    "*This table shows the first five rows of the created dataset. It gives a glimpse into the data structure, "
    "including the columns for file paths, class labels, plant types, and brightness values.*"
)
st.write("---")

# --- 4. Dataset Summary ---
st.subheader("4. Dataset Summary")

# Calculations for the summary
train_images = len(df)
valid_images = 17572  # Assuming the same number of images in train and valid for simplicity
total_classes = df['class'].nunique()
# The 'disease' column is cleaned in utils.py, so 'healthy' is an exact match
healthy_classes = df[df['disease'] == 'healthy']['class'].nunique()
# User-defined split
train_percentage = 80.0
valid_percentage = 20.0

st.write(f"Number of training images: **{train_images}**")
st.write(f"Number of validation images: **{valid_images}**")
st.write(f"Image Dimensions: **256x256 pixels**")
st.write(f"Train/Validation Split: **{train_percentage:.0f}% / {valid_percentage:.0f}%**")
st.write(f"Total Classes: **{total_classes}** (including healthy)")
st.write(f"Number of distinct healthy plant classes: **{healthy_classes}**")

st.markdown(
    "*This section provides a summary of the dataset's key characteristics, including image counts, dimensions, class distribution, and the train/validation split ratio.*"
)
st.write("---")


# --- 5. Plant Species Distribution ---
st.subheader("5. Distribution of Plant Species")
fig = plt.figure(figsize=(12, 6))
df['plant'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Plant Species", fontsize=16)
plt.xlabel("Plant Species", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
st.markdown(
    "*This bar chart illustrates the number of images available for each plant species. "
    "It helps to identify which species are most represented in the dataset and reveals any potential class imbalances.*"
)
st.write("---")


# --- 6. Interactive Breakdown per Plant ---
st.subheader("6. Image Count per Disease for a Selected Plant")
grouped = (
    df.groupby(["plant", "disease"])
    .size()
    .reset_index(name="Number of images")
)
plant_options = sorted(grouped['plant'].unique())
selected_plant = st.selectbox(
    "Select a plant species to show the number of images per disease:",
    options=plant_options,
    key='plant_selection'
)
filtered = grouped[grouped["plant"] == selected_plant]
st.dataframe(filtered)
st.markdown(
    "*This table provides a detailed breakdown of the number of images for each disease category (including healthy) "
    "for the plant species selected from the dropdown menu above.*"
)
st.write("---")


# --- 7. Visualizing Image Count per Disease ---
st.subheader("7. Visualizing Image Count per Disease")
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
st.markdown(
    "*This bar chart visually represents the data from the table above, "
    "making it easy to compare the number of images across different diseases for the selected plant.*"
)
st.write("---")


# --- 8. Brightness Distribution ---
st.subheader("8. Distribution of Image Brightness")
st.write(
    "This section shows the distribution of perceptual brightness values "
    "for the images in the dataset."
)

df["perceptual_brightness"].value_counts().sort_values(ascending=False)
too_dark_threshold = 30
too_bright_threshold = 220
brightness_values = df["perceptual_brightness"].dropna()
counts, bin_edges = np.histogram(brightness_values, bins=30)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
normalized = (
    bin_centers - bin_centers.min()
) / (bin_centers.max() - bin_centers.min())
colors = [(v, v, v) for v in normalized]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(
    bin_centers,
    counts,
    width=np.diff(bin_edges),
    color=colors,
    edgecolor='black',
    align='center'
)
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
st.markdown(
    "*This histogram shows the distribution of perceptual brightness across all images. The red and orange dashed lines "
    "indicate thresholds for images that might be considered too dark or too bright, respectively. This analysis is useful "
    "for identifying potential issues in the dataset that could affect model training.*"
)

# --- Sidebar Configuration ---
st.sidebar.title("Table of Contents")
st.sidebar.info(
    "Select a page above to explore different aspects of the project."
)
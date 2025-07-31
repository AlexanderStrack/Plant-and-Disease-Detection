 # page_1_exploration.py --- Exploration Page for Streamlit App

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils

st.header("Exploration")

st.write("### Presentation of data")
df = utils.load_excel_data()

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
st.write(
    "The plot shows that the brightness of the images is generally "
    "well-distributed, with a few images being too dark or too bright. "
    "This information can be useful for preprocessing the images before "
    "training a model."
)
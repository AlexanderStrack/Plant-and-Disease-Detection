# """pages/2_Data_Visualization.py --- Data Visualization Page for Streamlit App"""

import streamlit as st
import matplotlib.pyplot as plt
import re
import textwrap
import utils
import Code_for_streamlit

st.header("Data Visualization")
st.write("### Presentation of data")

train, valid = utils.load_images()

class_names = [Code_for_streamlit.clean_label(name) for name in train.class_names]
train.class_names = [name.replace(' ', '_') for name in class_names]

if train:
    st.success("Trainings- and validationdata successfully loaded!")
    st.subheader("Example images from the training dataset")
    st.write(
        "Here are some example images from the training dataset. "
        "The images are displayed along with their corresponding labels."
    )
    st.write(
        f"The images are resized to 128x128 pixels and "
        f"are displayed in batches of 64."
    )

images, labels = next(iter(train))

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    class_name_parts = train.class_names[labels[i]].split('___')
    plant_name = re.sub(r"[,_]", " ", class_name_parts[0])
    disease_name = "\n".join(textwrap.wrap(re.sub(r"[,_]", " ", class_name_parts[1]), width=13))
    axes[i].imshow(images[i].numpy().astype("uint8"))
    axes[i].set_title(f"{plant_name}\n{disease_name}")
    axes[i].axis("off")
st.pyplot(fig)

st.subheader("Discover the different plant species")
plant_options = sorted(set(name.split('___')[0].strip() for name in train.class_names))
plant_select = st.selectbox("Pick a Plant Species", options=plant_options, key='plant_species_checkbox')

disease_options = sorted(set(
    name.split('___')[1].strip() for name in train.class_names
    if st.session_state['plant_species_checkbox'] in name))
disease_options = [re.sub(r"[,_]", " ", name) for name in disease_options]
disease_select = st.selectbox("Choose the disease to display", disease_options, key='desease_selection')

target_label = f"{plant_select}___{disease_select.replace(' ', '_')}"
try:
    label_index = train.class_names.index(target_label)
    found_image = None
    for images, labels in train.unbatch().batch(1):
        if labels.numpy()[0] == label_index:
            found_image = images.numpy()[0].astype("uint8")
            break

    if found_image is not None:
        st.image(found_image, caption=target_label, use_column_width=True)
    else:
        st.warning("No image found for the selected label.")
except ValueError:
    st.error(f"Label '{target_label}' not found.")
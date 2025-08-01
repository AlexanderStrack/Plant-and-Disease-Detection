import streamlit as st

st.set_page_config(
    page_title="About the Project",
    layout="wide"
)

# --- Header ---
st.header("About This Project")
st.write("---")

# --- Project Context ---
st.markdown(
    """
    This project is the result of a collaborative capstone project developed during an intensive, 
    four-month full-time data science training program. 

    The training was conducted by **[DataScientest](https://datascientest.com/de/)**, a leading European provider 
    of data science education, in partnership with **Sorbonne University** for certification.
    """
)
st.write("---")

# --- The Team ---
st.subheader("The Team")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Alexander Strack")
    st.write("Data Scientist and Physician")

with col2:
    st.markdown("#### Felix Schramm")
    st.write("Data Scientist and Sociologist")
    st.markdown(
        "[View LinkedIn Profile](https://www.linkedin.com/in/felixschramm49/)"
    )

st.write("---")

# --- Project Supervision ---
st.subheader("Project Supervision")
st.markdown("#### Kylian Santos")
st.markdown(
    "[View LinkedIn Profile](https://www.linkedin.com/in/kylian-santos-5a96a3267/)"
)

# --- Sidebar Configuration ---
st.sidebar.title("Table of Contents")
st.sidebar.info(
    "Select a page above to explore different aspects of the project."
)
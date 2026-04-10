import streamlit as st
import os
from app import load_model, load_features

st.title("Fashion Recommender System")

st.write("Loading model...")
model = load_model()

st.write("Loading features...")
feature_list, filenames = load_features(model)

st.success("Ready!")

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())

        return 1
    except Exception as e:
        st.error(str(e))
        return 0

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("File uploaded successfully")
    else:
        st.error("Some error occurred in file upload")
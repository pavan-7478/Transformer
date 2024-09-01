import streamlit as st
from PIL import Image
import numpy as np
import torch
from io import BytesIO
import matplotlib.pyplot as plt

st.markdown("<h1 style='text-align: center;'>Image Colorizer</h1>",
            unsafe_allow_html=True)

opt = st.sidebar.selectbox("Choose Image Type", [
                           "Landscape", "Person", "Animals", "Flowers"])
opt = opt.lower()

img_file = st.sidebar.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    try:
        image = Image.open(BytesIO(img_file.read()))
        H, W = image.size
        image = image.resize((256, 256))
        btn = st.sidebar.button("Colorize")
        if btn:
            out = # pass into the model
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_column_width=True, caption="Original")
            with col2:
                st.image(out, use_column_width=True, caption="Coloured")
    except Exception as e:
        st.write(e)
        st.write("Unable to open file")
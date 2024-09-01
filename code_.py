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

img_file_input = st.sidebar.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png"])
img_file_target= st.sidebar.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png"])
if img_file_input is not None and img_file_target is not None:
    # Your code here

    try:
        image_input = Image.open(BytesIO(img_file_input.read()))
        image_target= Image.open(BytesIO(img_file_target.read()))
        image_input = image_input.resize((256, 256))
        image_target = image_target.resize((256, 256))
        btn = st.sidebar.button("Colorize")
        if btn:
            out = # pass the generate_images function to get the required output
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_target, use_column_width=True, caption="Original")
            with col2:
                st.image(out, use_column_width=True, caption="Coloured")
    except Exception as e:
        st.write(e)
        st.write("Unable to open file")
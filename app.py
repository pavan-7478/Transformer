import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, UpSampling2D, Dropout, Flatten, Dense, Input, LeakyReLU, Conv2DTranspose,AveragePooling2D, Concatenate
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.compat.v1 import set_random_seed
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
from io import BytesIO
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

def model_load():
    model = load_model('models/generator_model.h5')
    return model

def read_img_url(url, size = (256,256)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((256, 256))
    img = image.img_to_array(img)
    return img

def rgb_to_lab(img, l=False, ab=False):
    img = img / 255
    l_chan = color.rgb2lab(img)[:,:,0]
    l_chan = l_chan / 50 - 1
    l_chan = l_chan[...,np.newaxis]

    ab_chan = color.rgb2lab(img)[:,:,1:]
    ab_chan = (ab_chan + 128) / 255 * 2 - 1
    if l:
        return l_chan
    else: 
    	return ab_chan

def lab_to_rgb(img):
    new_img = np.zeros((256,256,3))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pix = img[i,j]
            new_img[i,j] = [(pix[0] + 1) * 50,(pix[1] +1) / 2 * 255 - 128,(pix[2] +1) / 2 * 255 - 128]
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img


def url_generator(url):
    
    img = read_img_url(url,size=(256,256)).astype('int64')
    l_channel = rgb_to_lab(img,l=True)
    model = model_load()
    fake_ab = model.predict(l_channel.reshape(1,256,256,1))
    fake = np.dstack((l_channel,fake_ab.reshape(256,256,2)))
    fake = lab_to_rgb(fake).astype('int64')
    return img, fake

st.sidebar.title("What do you want to do?")
app_mode = st.sidebar.selectbox("Choose the mode", [  "Color Your Own Image","Model Architecture" ])

if app_mode == "Color Your Own Image":
   with st.spinner("Coloring..."):
        st.title("Pick any image and lets color it in!")
        link = st.text_input('Put Image URL Here', 'Type Here')
        if link != 'Type Here':
            real,fake = url_generator(link)
            st.image(real,width=325)
            st.image(fake,width=325)
elif app_mode == "Model Architecture":
    with st.spinner('Gathering the data...'):
        st.title("Architecture of My Networks")
        st.markdown("Let me introduce you to the model and its architecture")
        image = Image.open('model_architecture.png')
        st.image(image)
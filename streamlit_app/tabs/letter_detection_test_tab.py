import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from numpy import asarray
import pickle
from streamlit_drawable_canvas import st_canvas

title = "Letter detection test"
sidebar_name = "Letter detection test"


def run():
    st.title(title)

    magnifier = 5
    img_height = 32
    img_width = 128


    # Params
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=img_height * magnifier,
        width=img_width * magnifier,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    # Do something interesting with the image data
    if canvas_result.image_data is not None:
        with st.spinner('Predicting using the model...'):
            img_data = canvas_result.image_data
            im = Image.fromarray(img_data.astype("uint8"), mode="RGBA").convert('L')
            im = im.resize((img_height, img_width))
            img_arr = np.array(im)
            img_arr = img_arr / 255.0
            X_test = np.array(img_arr).reshape(-1, img_height, img_width, 1)
            model = tf.keras.models.load_model('../pickle/image_letter_counter_mlp_model')
            y_pred = model.predict(X_test)
            st.write("Predicted length: {}".format(round(y_pred[0][0])))

            

def load_mlp_model():
    model = tf.keras.models.load_model('../pickle/image_letter_counter_mlp_model')
    # {'loss': 1.7622537994384766, 'mae': 0.96719146}
    # Out of 200 inputs 71 OK, 91 have one letter diff, 38 more than one

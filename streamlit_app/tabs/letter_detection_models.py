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


def show_models():
    magnifier = 5
    img_height = 32
    img_width = 128

    # Params
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image= None,
        update_streamlit=realtime_update,
        height=img_height * magnifier,
        width=img_width * magnifier,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    # Do something interesting with the image data
    if canvas_result.image_data is not None:
        with st.spinner('Predicting using the model...'):
            predict_canvas_image(canvas_result.image_data, (img_height, img_width))



def predict_canvas_image(img_data, img_size = ()):
    im = Image.fromarray(img_data.astype("uint8"), mode="RGBA").convert('L')
    im = im.resize(img_size)
    img_arr = np.array(im)
    img_arr = img_arr / 255.0
    X_test = np.array(img_arr).reshape(-1, img_size[0], img_size[1], 1)
    model = tf.keras.models.load_model('../pickle/image_letter_counter_mlp_model')
    y_pred = model.predict(X_test)
    st.write("Predicted length: {}".format(round(y_pred[0][0])))
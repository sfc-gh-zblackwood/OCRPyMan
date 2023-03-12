import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import random
import string

def get_random_string(length = 16):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
    
def render_file_uploader(default_filename, callback, callback_args = {}, uploader_key="uploader"):
    """
    Render a file uploader saving the given file into a temporary 
    file for easier manipulation and triggering a callback
    with given args (dict) AND filename
    filename must therefore be used as a parameter for the callback
    """
    uploaded_file = st.file_uploader("Choose a file", key=uploader_key)
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        filename = default_filename
        with open(filename, "wb") as binary_file:
            # Write bytes to file
            binary_file.write(bytes_data)
        callback(**callback_args, filename=filename)

def render_canvas(callback, callback_args = {}, canvas_key="canvas", filename=None):
    """
    Render a canvas and trigger a callback
    with given args (dict) AND img_arr
    img_arr must therefore be used as a parameter for the callback
    """
    magnifier = 5
    img_height = 32
    img_width = 128

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")


    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image = None,
        update_streamlit = False,
        height=img_height * magnifier,
        width=img_width * magnifier,
        drawing_mode="freedraw",
        point_display_radius=0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key=canvas_key,
    )
    if canvas_result.image_data is not None:
        with st.spinner('Loading...'):
            img_data = canvas_result.image_data
            if not filename is None:
                im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
                im.save(filename)
                callback(**callback_args, filename=filename)
                return

            im = Image.fromarray(img_data.astype("uint8"), mode="RGBA").convert('L')
            img_arr = np.array(im)
            img_arr = img_arr / 255.0
            callback(**callback_args, img_arr=img_arr)

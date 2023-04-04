import streamlit as st
from tabs import text_detection_model_lib as td_lib
from tabs import text_detection_st
from tabs import st_lib
from PIL import Image

title = "Text detection"
sidebar_name = "Text detection"

UPLOADER_DEFAULT_FILENAME = "tmp/uploaded_file.png"
CANVAS_DEFAULT_FILENAME = "tmp/canvas_file.png"

TEXT_DET_MODEL_ALL = "All"
TEXT_DET_MODEL_DEFAULT = "Default"
TEXT_DET_MODEL_CUT_FINE_TUNING = "Cut fine tuning"
TEXT_DET_MODEL_FINE_TUNING_FINAL = "Fine tuning final"

INPUT_METHOD_CANVAS = 'Canvas'
INPUT_METHOD_FILE_UPLOADER = 'File uploader'
INPUT_METHOD_IMG_SELECT = 'Image selector'

DEFAULT_CANVAS_SIZE = (3542, 2479)

input_random_key = 'init'

def run():
    st.title(title)
    tab1, tab2, tab3 = st.tabs(["Theory", "Model", "Comparison"])
    with tab1:
        text_detection_st.show_theory()
    with tab2:
        show_model()
    with tab3:
        text_detection_st.show_comparison()

def get_text_detection_model_path(model_name):
    if model_name == TEXT_DET_MODEL_ALL:
        return 'all'
    if model_name == TEXT_DET_MODEL_DEFAULT:
        return None
    if model_name == TEXT_DET_MODEL_CUT_FINE_TUNING:
        model_path_name = 'cut_fine_tuning'
    if model_name == TEXT_DET_MODEL_FINE_TUNING_FINAL:
        model_path_name = 'fine_tuning_final'
    return "../notebooks/text_detection/{}/weights".format(model_path_name)


# We are using the global input_random_key to reload the model
# when something is uploaded because of a change in the 
# input_random_key parameter
# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def render_selected_input_method(method_name, models, input_random_key):
    if method_name == INPUT_METHOD_FILE_UPLOADER:
        st_lib.render_file_uploader(
            UPLOADER_DEFAULT_FILENAME,
            on_image_uploaded, 
            {'models': models}
        )
        return
    if method_name == INPUT_METHOD_CANVAS:
        # height = st.slider("Canvas height: ", 50, DEFAULT_CANVAS_SIZE[0], 500)
        # width = st.slider("Canvas width: ", 50, DEFAULT_CANVAS_SIZE[1], 750)
        height = 500
        width = 750

        col1, col2, col3 = st.columns(3)
        with col1:
            stroke_width = st.slider("Stroke width: ", 1, 10, 3)
        with col2:
            stroke_color = st.color_picker("Stroke color hex: ")
        with col3:
            bg_color = st.color_picker("Background color hex: ", "#fff")


        st_lib.render_canvas(
            on_image_uploaded, 
            {'models': models},
            filename = CANVAS_DEFAULT_FILENAME,
            size=(height, width),
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            bg_color=bg_color
        )
        return
    if method_name == INPUT_METHOD_IMG_SELECT: 
        text_detection_st.render_img_select()
        return
    raise Exception("Undefined input method.")


def show_model():
    st.header("Model selection")
    model_name = st.radio("Choose a model", (    
            TEXT_DET_MODEL_ALL,    
            TEXT_DET_MODEL_CUT_FINE_TUNING,
            TEXT_DET_MODEL_DEFAULT,
            TEXT_DET_MODEL_FINE_TUNING_FINAL,
        )
    )
    models = td_lib.load_text_detection_model(get_text_detection_model_path(model_name), [get_text_detection_model_path(model_name) for model_name in [TEXT_DET_MODEL_DEFAULT, TEXT_DET_MODEL_CUT_FINE_TUNING, TEXT_DET_MODEL_FINE_TUNING_FINAL]])
    
    st.header("Input selection")
    input_method = st.selectbox(
        'Choose your input method',
        (INPUT_METHOD_CANVAS, INPUT_METHOD_FILE_UPLOADER,)
    )
    render_selected_input_method(input_method, models, input_random_key)
    st_lib.add_bottom_space()


def on_image_uploaded(models, filename):
    global input_random_key
    input_random_key = st_lib.get_random_string()
    st.header("Result")
    cols = st.columns(len(models))

    legends = ['Default', 'Cut fine tuning', 'Fine tuning']

    for i, det_model in enumerate(models):
        with cols[i]:
            doc, result = td_lib.show_bbox_from_file(det_model, filename)
            if len(models) > 1:
                st.caption(legends[i])

    st_lib.add_bottom_space()



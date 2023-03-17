import streamlit as st
from tabs import text_detection_model_lib as td_lib
from tabs import text_detection_st
from tabs import st_lib
from PIL import Image

title = "Text detection"
sidebar_name = "Text detection"

UPLOADER_DEFAULT_FILENAME = "tmp/uploaded_file.png"
CANVAS_DEFAULT_FILENAME = "tmp/canvas_file.png"

TEXT_DET_MODEL_DEFAULT = "Default"
TEXT_DET_MODEL_CUT_FINE_TUNING = "Cut fine tuning"
TEXT_DET_MODEL_FINE_TUNING_FINAL = "Fine tuning final"

INPUT_METHOD_CANVAS = 'Canvas'
INPUT_METHOD_FILE_UPLOADER = 'File uploader'
INPUT_METHOD_IMG_SELECT = 'Image selector'

input_random_key = 'init'

def run():
    st.title(title)
    tab1, tab2 = st.tabs(["Theory", "Model"])
    with tab1:
        # text_detection_st.show_theory()
        show_model()
    # with tab2:
    #     show_model()

def get_text_detection_model_path(model_name):
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
def render_selected_input_method(method_name, det_model, input_random_key):
    st.write("Using the cache key *{}*".format(input_random_key))
    if method_name == INPUT_METHOD_FILE_UPLOADER:
        st_lib.render_file_uploader(
            UPLOADER_DEFAULT_FILENAME,
            on_image_uploaded, 
            {'det_model': det_model}
        )
        return
    if method_name == INPUT_METHOD_CANVAS:
        st_lib.render_canvas(
            on_image_uploaded, 
            {'det_model': det_model},
            filename = CANVAS_DEFAULT_FILENAME,
            size=(3542, 2479)
        )
        return
    if method_name == INPUT_METHOD_IMG_SELECT: 
        text_detection_st.render_img_select()
        return
    raise Exception("Undefined input method.")


def show_model():
    st.header("Model selection")
    model_name = st.radio("Choose a model", (        
            TEXT_DET_MODEL_DEFAULT,
            TEXT_DET_MODEL_CUT_FINE_TUNING,
            TEXT_DET_MODEL_FINE_TUNING_FINAL,
        )
    )
    det_model = td_lib.load_text_detection_model(get_text_detection_model_path(model_name))
    
    st.header("Input selection")
    input_method = st.selectbox(
        'Choose your input method',
        (INPUT_METHOD_CANVAS, INPUT_METHOD_FILE_UPLOADER, INPUT_METHOD_IMG_SELECT)
    )
    render_selected_input_method(input_method, det_model, input_random_key)
    st_lib.add_bottom_space()


def on_image_uploaded(det_model, filename):
    global input_random_key
    input_random_key = st_lib.get_random_string()
    st.header("Result")
    doc, result = td_lib.show_bbox_from_file(det_model, filename)
    st_lib.add_bottom_space()

    # result.show(doc)


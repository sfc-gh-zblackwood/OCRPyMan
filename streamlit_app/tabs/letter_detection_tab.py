import streamlit as st
from tabs import letter_detection_theory as theory
from tabs import letter_detection_models as models

title = "Letter detection"
sidebar_name = "Letter detection"


def run():
    st.title(title)
    tab1, tab2 = st.tabs(["Theory", "Models"])
    with tab1:
        theory.render_theory()

    with tab2:
        models.show_models()

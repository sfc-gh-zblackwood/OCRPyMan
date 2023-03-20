import streamlit as st
from tabs import letter_detection_1 as subtab1
from tabs import letter_detection_2 as subtab2
from tabs import letter_detection_3 as subtab3
from tabs import letter_detection_4 as subtab4
from tabs import letter_detection_5 as subtab5
from tabs import letter_detection_theory as theory
from tabs import letter_detection_models as models

title = "Letter detection"
sidebar_name = "Letter detection"


def run():
    st.title(title)
    tab1, tab2, tab3, tab4, tab5= st.tabs(["First tries",
                                "CTC model",
                                "Data augmentation",
                                "Training & Tuning",
                                "Spelling corrector"])
    with tab1:
        subtab1.show_content()

    with tab2:
        subtab2.show_content()

    with tab3:
        subtab3.show_content()

    with tab4:
        subtab4.show_content()

    with tab5:
        subtab5.show_content()


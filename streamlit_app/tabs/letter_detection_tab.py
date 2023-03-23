import streamlit as st
from tabs import letter_detection_1 as subtab1
from tabs import letter_detection_2 as subtab2
from tabs import letter_detection_3 as subtab3
from tabs import letter_detection_4 as subtab4
from tabs import letter_detection_5 as subtab5
from tabs import letter_detection_theory as theory
from tabs import letter_detection_models as models

# Le tab était initiallement dédié au letter detection.
# Mais, pour garder la cohérence avec le rapport, je propose de le transformer en Text Recognition Modeling

title = "Text recognition modeling"
sidebar_name = "Text recognition modeling"


def run():
    st.title(title)
    st.markdown("---")
    tab1, tab2, tab3, tab5= st.tabs(["First tries",
                                "Data augmentation",
                                "Model construction",
                                "Model predicting"])
    with tab1:
        subtab1.show_content()

    with tab2:
        subtab2.show_content()

    with tab3:
        subtab3.show_content()

    with tab5:
        subtab5.show_content()


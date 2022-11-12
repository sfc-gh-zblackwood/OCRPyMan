import streamlit as st


title = "OCRpyMAN"
sidebar_name = "Introduction"


def run():
    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        **Optical Character Recognition (OCR)** is nowadays heavily used in many different fields. 
        **Recognizing the text in an image** can be used to digitalise it and even detects a specific
        structure so as to categorize the text into a specific kind of document.

        In this project, we are trying to **implement from scratch** an Handwritten Text Recognition
        (HTR) model to better understand the underlying deep learning algorithms used in this task.

        To achieve this, we will use the *IAM Dataset*.
        """
    )

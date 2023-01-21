import streamlit as st


sidebar_name = "Introduction"


def run():
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.title("OCRpyMAN")
    st.markdown("---")
    st.markdown(
        """
        **Optical Character Recognition (OCR)** is nowadays heavily used in many different fields.\n 
        **Recognizing the text in an image** can be used to simply digitalise it or even detect a specific
        structure to later categorize it into a specific kind of document.
        """
        )
    st.markdown("""
        In this project, we are trying to **implement from scratch** an Handwritten Text Recognition
        (HTR) model to better understand the underlying deep learning algorithms used in this task.
        To achieve this, we will use the *IAM Dataset*.
        """
    )

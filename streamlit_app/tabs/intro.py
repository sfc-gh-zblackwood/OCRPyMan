import streamlit as st
from tabs import st_lib


sidebar_name = "Introduction"


def run():
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.title("OCRpyMAN")
    st.markdown("---")

    st.subheader("Context")
    
    st.markdown(
        """
        **Optical Character Recognition (OCR)** is nowadays heavily used in many different fields.\n 
        **Recognizing the text in an image** can be used to simply digitalise it or even detect a specific
        structure to later categorize it into a specific kind of document.
        """
        )
    
    st.subheader("Project objectives")
    
    st.markdown("""
        ##### 1. Implementation of a Handwritten Text Recognition model from scratch
        ##### 2. Implementation of a Text Detection model with the pretrained package docTR
        ##### 3. Combining the two previous models
       
        """
    )

    st_lib.add_bottom_space()

import streamlit as st
#import pandas as pd
#import numpy as np
#from PIL import Image
#import cv2
#import matplotlib.pyplot as plt

title = "Dataset"
sidebar_name = "Dataset"


def run():
    # TODO 1 slide de présentation des données (volumétrie, architecture, etc.).

    st.title(title)
    tab1, tab2, tab3 = st.tabs(["Description", "Forms", "Words"])
    with tab1:
        show_description()
    with tab2:
        show_forms()
    with tab3:
        show_words()



#@st.cache
#def read_pickle(filepath):
#    return pd.read_pickle(filepath)

def show_description():
    st.subheader("Dataset description")
    st.markdown(
        """
        The IAM Handwriting Database 3.0 is structured as follows:
        - 657 writers contributed samples of their handwriting,
        - 1'539 pages of scanned text (forms),
        - 5'685 isolated and labeled sentences,
        - 13'353 isolated and labeled text lines,
        - 115'320 isolated and labeled words (transcription) with the coordinates and size of the
        respective bounding boxes and a boolean label (seg_res) for segmentation success.
        
        The words are all in the English language.
        All form, line and word images are provided as PNG files in grayscale.
        """
    )
    st.image("../images/dataset_DB6.png")
    st.markdown(" ")

        
def show_forms():
    st.subheader("Forms")
    st.markdown(
        """
        The following Figure shows 10 random forms from the 1539 forms on the dataset.
        """)
    st.image("../images/dataset_DB2.png")
    
    st.markdown(
        """
        The images in the dataset are all produced in good conditions in terms of the contrast between the text and the background. Generally, there are no bumps, colors or patterns that can make text extraction difficult. Here are the min and max gray level forms:
        """)
    st.image("../images/dataset_DB3.png")

    st.markdown(
        """
        ***
        The following Figure shows a heatmap of distribution of words on the form format. We can see a uniform distribution along a linear pattern of evenly spaced lines.
        """)
    st.image("../images/text_box_position.png")
    
    st.markdown(" ")
    
            
def show_words():
    st.subheader("Words")
    st.markdown(
        """
        The following Figure shows 50 of the 115320 images of words on the dataset:
        """)
    st.image("../images/dataset_DB4.png")
    
    st.markdown(
        """
        This last Figure shows a great variability of handwriting images on the dataset: styles, sizes, thickness, contrast and background noise. We can also identify the presence of punctuation on the dataset and this subject will be addressed in the cleaning steps of the pre-processing phase. Nonetheless, images of regular words are generally recognizable by a human and correspond to the respective transcriptions.
        """)
    st.markdown(
        """
        ***
        The Figure below is a word cloud of the dataset and gives the reader an intuition of the most common words and characters.
        """)
    st.image("../images/dataset_DB5.png")

    st.markdown(
        """
        ***
        The following scatter plot shows characters of the english language plotted with x as percentage of the english language and y as percentage in our dataset. The distribution is approximately linear which indicates the corpus of our dataset is representative of the english language.
        """)
    st.image("../images/dataset_DB8.png")

    st.markdown(" ")
    
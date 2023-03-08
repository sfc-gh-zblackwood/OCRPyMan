import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

title = "The dataset"
sidebar_name = "The dataset"

def run():
    # TODO 1 slide de présentation des données (volumétrie, architecture, etc.).

    st.title(title)
    st.markdown(
        """
        The first step is to analyse our dataset. 

        The IAM Handwriting Database 3.0 is structured as follows:
        - 657 writers contributed samples of their handwriting
        - 1'539 pages of scanned text
        - 5'685 isolated and labeled sentences
        - 13'353 isolated and labeled text lines
        - 115'320 isolated and labeled words

        All form, line and word images are provided as **PNG files**.
        """
    )

    df_before_prepro = read_pickle('../pickle/preprocessing_word_df_before.pickle')
    df = read_pickle('../pickle/df.pickle')
    # TODO Add some images of:
    # - form, 
    # - lines,
    # - sentences,
    # - words

    st.markdown(
        """
        The dataset is exclusively composed of english words. As a consequence, our OCR will perform significantly better 
        when detecting the english language. 
        """
    )




    st.write("However, only")
    st.write(len(df), "words out of ", len(df_before_prepro), " initials have been used. Why ? Because a *processing* phase was required.")


@st.cache
def read_pickle(filepath):
    return pd.read_pickle(filepath)
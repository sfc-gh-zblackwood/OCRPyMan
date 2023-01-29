import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

title = "The dataset"
sidebar_name = "The dataset"

def run():
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

    df_before_prepro = pd.read_pickle('../pickle/preprocessing_word_df_before.pickle')
    st.write(df_before_prepro[['word_id', 'seg_res', 'transcription']].describe(include='all').fillna("").astype("str"))

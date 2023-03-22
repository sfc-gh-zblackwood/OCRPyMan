import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../notebooks/')
import rendering as rd

title = "Dataset visualisation"
sidebar_name = "Dataset visualisation"


def show_word(path, transcription):
    img = plt.imread(path)
    fig = plt.figure(figsize=(2,2))
    ax1 = fig.add_subplot(111)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax1.set_title(transcription)
    st.pyplot(fig)
    



def run():
    st.title(title)
    st.subheader("Word dataframe")
    st.write("This dataframe displays 5 lines of the **Words** data:")
    word_df = pd.read_pickle('../pickle/preprocessing_word_df_bad_contrast.pickle')
    word_df=word_df.drop('michelson_contrast', axis=1)
    word_df['word_img_path'] = word_df['word_img_path'].apply(lambda x: x[3:])

    st.dataframe(word_df.head())
    size = word_df.shape[0]
    st.write('The dataset has ',size,'words.')

    st.markdown("---")
    
    st.subheader("Word visualization")
    st.write('Insert the index to visualize:')
    query_index = st.number_input('Index',min_value=0, max_value = size, format='%i',step =1)
    st.write('The index is ', query_index)
    show_word(word_df.iloc[query_index].word_img_path,word_df.iloc[query_index].transcription)

    st.markdown("---")
    
    st.subheader("Form dataframe")
    st.write("This dataframe displays 5 lines of the **Forms** data:")
    form_df = pd.read_pickle('../pickle/preprocessing_form_df_before.pickle')
    st.dataframe(form_df.head())

    size = form_df.shape[0]

    st.write('The dataset has ',size,'forms.')
    

@st.cache
def read_pickle(filepath):
    return pd.read_pickle(filepath)
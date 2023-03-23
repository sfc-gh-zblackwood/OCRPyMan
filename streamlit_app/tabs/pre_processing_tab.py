import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

title = "Pre-Processing"
sidebar_name = "Pre-Processing"

def show_bad_constrast_imgs():
    bad_contrast_df = pd.read_pickle('../pickle/preprocessing_word_df_bad_contrast.pickle')
    bad_contrast_df = bad_contrast_df[bad_contrast_df['michelson_contrast'] == 0]
    #st.dataframe(bad_contrast_df.head(5))
    nb_rows = 2
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols)
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            path = bad_contrast_df.iloc[i + nb_cols * j].word_img_path[3:]
            ax = axs[i, j]
            img = plt.imread(path)
            ax.axis('off')
            ax.set_title(bad_contrast_df.iloc[i + nb_cols * j].transcription)
            ax.imshow(img, cmap='gray')
    st.pyplot(fig)



def run():

    st.title(title)
    st.markdown("---")
    st.markdown(
        """
        The pre-processing of our Words dataset consisted of the following 4 steps represented in the following flowchart:
        """
    )
    st.image('../images/preprocessing_DB0.png')
    st.markdown("---")
    st.subheader("Step 1:")
    st.markdown(
        """
        2 image files were corrupted (indexes 4152 and 113621) and were excluded from the dataset.
        """
    )
    st.markdown("---")

    st.subheader("Step 2:")
    st.markdown(
        """
        19864 word images are labeled as having a faulty segmentation. The following Figure shows 45 of these images. We can see that these images include multiple words. So, all these words were excluded from our dataset.
        """
    )
    st.image('../images/preprocessing_DB1.png')
    st.markdown("---")
        
    st.subheader("Step 3:")    
    st.markdown(
        """
        The investigation of Michelson contrast of the word images revealed that 132 of them have a contrast of 0, which means the images are all in black color as shown in the next Figure.
        """
    )
    show_bad_constrast_imgs() 
    st.markdown(
        """
        The histogram below shows that most of these images correspond to the “full stop” and other punctuation signs. There are also some words with 0 contrast images. All the 0 contrast images were excluded from the dataset.
        """
    )
    
    #st.code(
    #"""
    #def get_michelson_contrast(img_path):
    #    try:
    #        img = cv2.imread(img_path)
    #        Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    #    except Exception:
    #        return -1
    #    min = int(np.min(Y))
    #    max = int(np.max(Y))
    #    if min == 0 and max == 0:
    #        return 0 
    #    return ((max - min) / (min + max))
    #""", language="python")
    
    st.image('../images/preprocessing_DB2.png')
    st.markdown("---")

    st.subheader("Step 4:")    
    st.markdown(
        """
        The last step of the preprocessing phase consists of filtering out words with unauthorized characters. This list of authorized characters was defined as follows:
        """
    )
    st.code("""charList = list(string.ascii_letters)+[' ', ',', '.', '\'', '"', '-', '#']""", language = 'python')
    st.markdown(
        """
        This list has 59 characters in total. And this is an important parameter for the dimension of our model later on.
        """
    )
    st.markdown("---")

    #st.subheader("Conclusion")   
    #df_before = pd.read_pickle('../pickle/preprocessing_word_df_before.pickle')
    #df = pd.read_pickle('../pickle/df.pickle')
    st.markdown(
        "Thus, after cleaning our dataset, there are 94897 images of words remaining out of the initial 115320. The list of these images was saved to a pickle file for later use on the text recognition model construction.")
    st.markdown(" ")
    
    #st.markdown(
    #    "Thus, after cleaning our dataset, only **{}** pictures of words remain out of the **{}** given.".format(len(df), len(df_before))
    #)
    #st.dataframe(df.head(5))

    #st.markdown(
    #    """
    #    Now, we can take a look at a few images from our cleaned dataset.
    #    """
    #)

    #show_cleaned_imgs(df)


def show_cleaned_imgs(df):
    nb_rows = 2
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols)
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            path = df.iloc[i + nb_cols * j].word_img_path[3:]
            ax = axs[i, j]
            img = plt.imread('../' + path)
            ax.axis('off')
            ax.imshow(img, cmap='gray')
    st.pyplot(fig)
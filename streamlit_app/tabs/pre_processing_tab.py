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
    st.dataframe(bad_contrast_df.head(5))
    nb_rows = 2
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols)
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            path = bad_contrast_df.iloc[i + nb_cols * j].word_img_path[3:]
            ax = axs[i, j]
            img = plt.imread(path)
            ax.axis('off')
            ax.imshow(img, cmap='gray')
    st.pyplot(fig)



def run():

    st.title(title)

    st.markdown(
        """
        Once, we have our dataset, we can start the pre-processing phase.
        This step is **one of the most important step** in a datascience project.
        In the `preprocessing` notebook, we are reading the numerous files and folders and we are
        building a dataframe out of it.
        """
    )

    df_before = pd.read_pickle('../pickle/preprocessing_word_df_before.pickle')
    st.dataframe(df_before.head(5))

    st.markdown(
        """
        However the dataset was far from being clean and some images were either not readable
        or simply of poor quality. 
        """
    )
    show_bad_constrast_imgs()

    st.subheader("Cleaning steps:")

    st.markdown(
        """
        We have therefore:
        - removed the pictures where the segmentation was faulty to be sure that our labels/targets are correct
        - calculated for each image the michelson contrast to filter the picture where the contrast was 0; indicating either a dot 
        or a badly parsed image.

        We have also gathered together the information we have on a specific word in the context of its 
        own form.  
        """
    )
    st.code(
    """
    def get_michelson_contrast(img_path):
        try:
            img = cv2.imread(img_path)
            Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
        except Exception:
            return -1
        min = int(np.min(Y))
        max = int(np.max(Y))
        if min == 0 and max == 0:
            return 0 
        return ((max - min) / (min + max))
    """, language="python")

    df = pd.read_pickle('../pickle/df.pickle')
    st.markdown(
        "Thus, after cleaning our dataset, only **{}** pictures of words remain out of the **{}** given.".format(len(df), len(df_before))
    )
    st.dataframe(df.head(5))

    st.markdown(
        """
        Now, we can take a look at th
        """
    )

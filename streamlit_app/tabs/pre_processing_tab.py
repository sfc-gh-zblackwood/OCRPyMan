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
        The first step is to analyse our dataset. 
        In the `preprocessing` notebook, we are reading the numerous files and folders in order to
        build a dataframe out of it.

        We have gathered together the information on a specific word in the context of its 
        own form.  
        """
    )

    df_before = pd.read_pickle('../pickle/preprocessing_word_df_before.pickle')
    st.dataframe(df_before.head(5))
    # st.write(
    #     df_before.describe()
    # )

    st.markdown(
        """
        However the dataset was far from being clean and some images were either not readable
        or simply of poor quality. We have therefore:
        - remove the pictures where the segmentation was faulty to be sure that our labels are correct
        - calculate for each image the michelson contrast to filter the picture where the contrast was 0. 
        """
    )
    show_bad_constrast_imgs()


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

    # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))
    # st.line_chart(chart_data)

    # st.markdown(
    #     """
    #     ## Test 2
    #     The dataset is not perfect and some pictures were not 

    #     """
    # )

    # st.area_chart(chart_data)

    # st.markdown(
    #     """
    #     ## Test 3

    #     You can also display images using [Pillow](https://pillow.readthedocs.io/en/stable/index.html).

    #     ```python
    #     import streamlit as st
    #     from PIL import Image

    #     st.image(Image.open("assets/sample-image.jpg"))

    #     ```

    #     """
    # )

    # st.image(Image.open("assets/sample-image.jpg"))

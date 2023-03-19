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
    



    
    
    ### DB : Proposition de supprimer plus tard
    # Distribution des mots sur l'image (position)
    st.markdown("""
    #### [DB] : La partie suivante sera remplacée
    """)
    st.image("../images/text_box_position.png")
    st.markdown(
        """
        As we can see, the text boxes in our dataset are 
        always positioned in lines, in a delimited space in our forms.
        This biais could pushes our model to recognize only *straight* boxes
        not too far from the center. 
        
        As a consequence, a word placed in a corner and slightly 
        rotated may not be easily detected 
        """
    )

    # Graphe distribution des lettres + comparaison langue anglaise
    st.image("../images/viz_letters.png")
    st.image("../images/english_letter_distribution.png")
    st.image("../images/viz_letters2.png")
    st.markdown(
        """
        TODO Dire que c'est fortement lié à la langue anglaise 
        donc si on prend une langue comme le grecque, pas ouf car
        pas même frequence + bien sur des lettres non présentes
        
        Souligner le pb des majuscules sous représentées (éventuellement rajouter 
        un ratio du nombre de lettres majuscules minuscules) 
        ratio_a = nb_a / nb_A ?
        """
    )

    st.image("../images/viz_letter_count_word.png")
    st.markdown(
        """
        TODO Dire que les mots ne sont généralement pas longs <= 5 voire 10
        => L'allemand risque de poser problème si on devait détecter des boxes 
        ...
        """
    )

@st.cache
def read_pickle(filepath):
    return pd.read_pickle(filepath)
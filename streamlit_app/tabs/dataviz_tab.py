import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

title = "Dataset visualisation"
sidebar_name = "Dataset visualisation"

def run():
    st.title(title)

    # Distribution des mots sur l'image (position)
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
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../notebooks/')
import rendering as rd


def show_word(path, transcription):
    img = plt.imread(path)
    fig = plt.figure(figsize=(2,2))
    ax1 = fig.add_subplot(111)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax1.set_title(transcription)
    st.pyplot(fig)
    



def show_content():
    st.subheader("Words")
    st.write("This dataframe displays 5 lines of the **Words** data:")
    word_df = pd.read_pickle('../pickle/preprocessing_word_df_bad_contrast.pickle')
    word_df=word_df.drop('michelson_contrast', axis=1)
    word_df['word_img_path'] = word_df['word_img_path'].apply(lambda x: x[3:])

    st.dataframe(word_df.head())
    size = word_df.shape[0]
    st.write('The dataset has ',size,'words. The following Figure shows 50 random words:')
    st.image("../images/dataset_DB4.png")
    st.markdown(
        """
        This last Figure shows a great variability of handwriting images on the dataset: styles, sizes, thickness, contrast and background noise. We can also identify the presence of punctuation on the dataset and this subject will be addressed in the cleaning steps of the pre-processing phase. Nonetheless, images of regular words are generally recognizable by a human and correspond to the respective transcriptions.
        """)
    st.markdown("---")
    
    st.markdown("### Word visualization")
    st.write('Insert the index to visualize:')
    query_index = st.number_input('Index',min_value=0, max_value = size, format='%i',step =1, value = 1000)
    st.write('The index is ', query_index)
    show_word(word_df.iloc[query_index].word_img_path,word_df.iloc[query_index].transcription)

    
    
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
    
    

@st.cache
def read_pickle(filepath):
    return pd.read_pickle(filepath)
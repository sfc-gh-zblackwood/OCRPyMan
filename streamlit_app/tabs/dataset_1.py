import streamlit as st
import pandas as pd

import sys
sys.path.insert(1, '../notebooks/')
import rendering as rd

    

def show_content():
    st.subheader("Forms")
    
    st.write("This dataframe displays 5 lines of the **Forms** data:")
    form_df = pd.read_pickle('../pickle/preprocessing_form_df_before.pickle')
    st.dataframe(form_df.head())

    size = form_df.shape[0]

    st.write('The dataset has ',size,'forms. The following Figure shows 10 random forms:')
    
    st.image("../images/dataset_DB2.png")
    
    st.markdown(
        """
        ***
        The images in the dataset are all produced in good conditions in terms of the contrast between the text and the background. Generally, there are no bumps, colors or patterns that can make text extraction difficult. Here are the min and max gray level forms:
        """)
    st.image("../images/dataset_DB3.png")

    st.markdown(
        """
        ***
        The following Figure shows a heatmap of distribution of words on the form format. We can see a uniform distribution along a linear pattern of evenly spaced lines.
        """)
    st.image("../images/text_box_position.png", width=320)
    
    st.markdown(" ")
    
    

@st.cache
def read_pickle(filepath):
    return pd.read_pickle(filepath)
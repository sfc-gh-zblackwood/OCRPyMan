import streamlit as st
from tabs import demo_full_ocr_ressources as demo

import sys
sys.path.insert(1, '../notebooks')
import ressources as rss

title = "Demo Full OCR"
sidebar_name = "Demo Full OCR"


def run():
    rss.init()
    
    st.title(title)
    st.write("This page is intended to let you use the models we created.")
    st.markdown(
        """
        There is 3 ways to use them : 
        - Drawing with the mouse  
        - Using an extract from the original data : 1 form ramdomly choosed among a 20-forms subset
        - Uploading a local image 
              
        """)

    
    tab1, tab2, tab3 = st.tabs(["Drawing", "Data extract", "Local image"])
    # st.write(tab1)
    # st.write(tab2)
    with tab1:
        demo.show_drawing()

    with tab2:
        demo.show_data_extract()
        
    with tab3:
        demo.show_local()

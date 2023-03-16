import streamlit as st
from tabs import demo_text_reco_ressources as demo

title = "Demo Text Recognition"
sidebar_name = "Demo Text Recognition"


def run():
    st.title(title)
    st.write("This page is intended to let you use the models we created.")
    st.markdown(
        """
        There is 3 ways to use them : 
        - Drawing with the mouse  
        - Using an extract from the original data : 12 images ramdomly choosed among an 100-images subset
        - Uploading a local image 
              
        """)
    st.markdown("**There is no spellchecking on these predictions**")
    
    tab1, tab2, tab3 = st.tabs(["Drawing", "Data extract", "Local image"])
    # st.write(tab1)
    # st.write(tab2)
    with tab1:
        demo.show_drawing()

    with tab2:
        demo.show_data_extract()
        
    with tab3:
        demo.show_local()

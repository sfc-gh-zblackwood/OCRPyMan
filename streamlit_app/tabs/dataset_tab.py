import streamlit as st
from tabs import dataset_1 as subtab1
from tabs import dataset_2 as subtab2


title = "Dataset"
sidebar_name = "Dataset"


def run():
    # TODO 1 slide de présentation des données (volumétrie, architecture, etc.).

    st.title(title)
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Description", "Forms", "Words"])
    with tab1:
        show_description()
    with tab2:
        subtab1.show_content()
    with tab3:
        subtab2.show_content()




def show_description():
    st.subheader("Dataset description")
    st.markdown(
        """
        The **IAM Handwriting Database 3.0** is structured as follows:
        - 657 writers contributed samples of their handwriting,
        - 1'539 pages of scanned text (forms),
        - 5'685 isolated and labeled sentences,
        - 13'353 isolated and labeled text lines,
        - 115'320 isolated and labeled words (transcription) with the coordinates and size of the
        respective bounding boxes and a boolean label (seg_res) for segmentation success.
        
        The words are all in the English language.
        All form, line and word images are provided as PNG files in grayscale.
        """
    )
    st.image("../images/dataset_DB6.png")
    st.markdown(" ")

    
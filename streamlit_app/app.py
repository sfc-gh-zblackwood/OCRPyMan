from collections import OrderedDict

import streamlit as st
 
import config

from tabs import intro ,pre_processing_tab, dataviz_tab, text_recognition_tab, text_detection_tab, dataset_tab, letter_detection_tab
from tabs import demo_text_recognition


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (dataset_tab.sidebar_name, dataset_tab),
        (dataviz_tab.sidebar_name, dataviz_tab),
        (pre_processing_tab.sidebar_name, pre_processing_tab),
        (letter_detection_tab.sidebar_name, letter_detection_tab),
        (text_detection_tab.sidebar_name, text_detection_tab),
        (text_recognition_tab.sidebar_name, text_recognition_tab),
        (demo_text_recognition.sidebar_name, demo_text_recognition),
    ]
)


def run():
    st.sidebar.image(
         "assets/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("Menu", list(TABS.keys()), 5)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)
    tab = TABS[tab_name]
    tab.run()


if __name__ == "__main__":
    run()

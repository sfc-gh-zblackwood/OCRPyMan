from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

from tabs import intro , pre_processing_tab, text_recognition_tab, text_detection_tab, dataset_tab, letter_detection_tab


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (dataset_tab.sidebar_name, dataset_tab),
        (pre_processing_tab.sidebar_name, pre_processing_tab),
        (letter_detection_tab.sidebar_name, letter_detection_tab),
        (text_detection_tab.sidebar_name, text_detection_tab),
        (text_recognition_tab.sidebar_name, text_recognition_tab),
    ]
)


def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200,
    )
    #TODO Uncomment to get to first page on reload
    # tab_name = st.sidebar.radio("Menu", list(TABS.keys()), 0)
    tab_name = st.sidebar.radio("Menu", list(TABS.keys()), 3)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)
    tab = TABS[tab_name]
    tab.run()


if __name__ == "__main__":
    run()

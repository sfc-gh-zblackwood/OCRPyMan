import streamlit as st
import pandas as pd
import numpy as np


title = "Text detection"
sidebar_name = "Text detection"


def run():

    st.title(title)

    st.markdown(
        """
        To do the text detection part, we have decided to make use of the Doctr
        API.
        """
    )


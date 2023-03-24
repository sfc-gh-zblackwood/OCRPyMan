import streamlit as st

def show_content():
    st.header("First modeling tries")
    st.markdown(
        """
        As part of the learning process, the first attempts of classification were incremental in scope and complexity.

        The first attempt was to determine the number of characters in the input words by using a simple MLP model where words were fed to the model as a one dimension array. The results are summarized below and show a poor performance. The main reason is that the one dimension array format loses the geometric patterns information.
        """
    )
    st.image("../images/letter_detection_DB11.png")
    st.markdown("---")
    st.markdown("""
    Later, one attempt was made to create a text recognition model using a Lenet CNN. The plot below shows the accuracy of the model on the train set in blue and the accuracy on the test set in green. It is noticeable that performance keeps improving on the train set with epochs but quickly stagnates on the test set, which indicates the model was “memorizing” the train set (overfitting). 
    """)
    st.image("../images/letter_detection_DB12.png")
    st.markdown(" ")
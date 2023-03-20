import streamlit as st

def show_content():
    st.subheader("Training & Tuning")
    st.markdown(
        """
        Regarding the model training, we have varied the number of epochs and the learning rate. We also used some callbacks for our convenience : early stop, checkpoint, and, sometimes, reduce learning rate on plateau. Some training runs were made including the generated words.

        We can see below what happened in some of our tests. We observe improvements in each of these tests : the loss is decreasing consistently, and the model's precision is improving.
        """
    )
    st.image("../images/letter_detection_DB41.png")
    st.image("../images/letter_detection_DB42.png")
    st.image("../images/letter_detection_DB43.png")
    st.image("../images/letter_detection_DB44.png")
    st.markdown(" ")
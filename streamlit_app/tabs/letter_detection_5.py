import streamlit as st

def show_content():
    st.subheader("CTC model")
    st.markdown(
        """
        As a complement to the text recognition model, a spelling corrector was tested and finally added to improve the final accuracy. The spelling corrector is independent from the CTC classification.  

        The metrics used to evaluate the spelling corrector were: 
        - Levenshtein distance.
        - CER (Character Error Rate). 
        - WER (Word Error Rate). 
        """
    )
    st.markdown("""
        To find a better spelling corrector solution, we tried an out-of-the box solution : the autocorrect library [ref4]. The overall word corrections seem better and faster.
        Using the exact word metric, we see significant improvements in the results. But using the Levenshtein distance metric (per character), we didn’t see much improvement.The reason behind this is that, sometimes, the corrector has to make a choice between words with equal probability (for example, it can fix “fead” in “dead” or “head”) and thus the distance can increase a lot for longer words.
        When used with our best model, which has a precision per character of 93.45% and per word 79.9%, we get a precision per character of 93.78%, which is almost negligible, but the precision per word gets a significant improvement : 85.7%.
        Here is an example of corrected words:
            """)
    st.image("../images/letter_detection_DB51.png")
    st.markdown(" ")
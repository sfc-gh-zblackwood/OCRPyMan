import streamlit as st

def show_content():
    st.subheader("Prediction")
    st.markdown(
        """
        The model prediction for given input image is represented bellow: 
        """
    )
    st.image("../images/letter_detection_DB50.png")
    
    st.markdown(
        """
        - The input image is fed through CNN and RNN layers to yield a probability matrix for all possible paths.
        - The CTC Decoding will get the prediction output by looking for the best path in the probability matrix (with the highest probability). In this project, we used a CTC greedy decoder. This decoder selects the most probable character at each timestep. It is a simpler and faster decoding technique.
        - The Figure below represented this decoding process:
"""
    )
    st.image("../images/letter_detection_DB23.png", width=400)
    st.markdown("---")
    
    st.subheader("Metrics")
    st.markdown(
        """
        The metrics used to evaluate the spelling corrector were: 
        - Levenshtein distance
        - CER (Character Error Rate)
        - WER (Word Error Rate)
        """
    )
    st.markdown("---")
    
    st.subheader("Spelling corrector")
    st.markdown(
        """
        As a complement to the text recognition model, a spelling corrector was tested and finally added to improve the final accuracy. The spelling corrector is independent from the CTC classification.  
        To find a better spelling corrector solution, we tried an out-of-the box solution : the **autocorrect library**.
        """
    )
    
    st.markdown("""
        The improvements with the spelling corrector are:
        - CER: from 93.45% to 93.78% (+0.33%)
        - WER: from 79.9% to 85.7% (+5.8%)
        
        Using the WER metric, we see significant improvements in the results. But using the Levenshtein distance metric (per character), we didn’t see much improvement. The reason behind this is that, sometimes, the corrector has to make a choice between words with equal probability (for example, it can fix “fead” in “dead” or “head”) and thus the distance can increase a lot for longer words.
        
        Here is an example of predictions with and without the spelling corrector:
            """)
    st.image("../images/letter_detection_DB51.png")
    
    st.markdown("---")
    st.subheader("Final results")
    
    st.markdown("""
        The results of the trained model are:
        - CER: 93.8%
        - WER: 85.7%
        """)
    st.markdown(" ")
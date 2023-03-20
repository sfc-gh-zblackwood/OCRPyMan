import streamlit as st

def show_content():
    st.subheader("CTC model")
    st.markdown(
        """
        Handwriting is intrinsically very variable in both the geometric style (different patterns) but also in geometric disposition (different distortions, different spacing, etc.). And because of the latter, the main difficulty to overcome in handwriting recognition problems is the random alignment between the handwritten input and the digital output along the horizontal axis. 
        """
    )
    st.markdown(
        """
        The strategy for this type of model can be summarized as follows:

        - The image is cut in fragments along the writing axis which are fed through different CNN layers for visual feature extraction.
        """
    )
    st.image("../images/letter_detection_DB21.png", width = 320)
    #st.markdown("---")
    st.markdown(
        """
        - The result of the previous step which can be seen as a group of timesteps is fed into a bidirectional RNN which will map the probabilities of each fragment corresponding to a given character.
        """
    )
    st.image("../images/letter_detection_DB22.png", width = 320)
    st.markdown(
        """
        - Concerning the CTC layer, when the model is being trained:
            - The CTC loss function will encode all the possible alignments (or paths) for the known output (transcription) on the RNN output matrix. Each path has a calculated probability.
            - The objective of the training is to maximize the probability of matching the output and to minimize the loss. So, the loss is then calculated as the negative sum of log-probabilities of all the encoded paths.
            - To train the model, the gradient of the loss function is calculated in respect to the different trainable parameters which are updated at each iteration.
        """
    )
    st.markdown(
        """
        - Concerning the CTC layer, when the model is predicting:
            - With an input image, the forward propagation through the CNN and RNN layers will yield a matrix of probabilities for all possible paths.
            - The predicted output is obtained by looking for the best path (with the highest probability). In this project with a CTC greedy decoder.
            - The Figure below represented this prediction process.
"""
    )
    st.image("../images/letter_detection_DB23.png", width=400)
    st.markdown(
        """
        The following diagram gives a representation of the chosen model:
        - When training: back-propagation to minimize the CTC loss and update the NN parameters.
        - When predicting: forward propagation and CTC decoding of the best path to predict the output.
        """
    )
    st.image("../images/letter_detection_DB24.png")
    st.markdown(
        """
        The model was implemented with the TensorFlow framework. It has 1.4 million trainable parameters."""
    )
    st.markdown(" ")
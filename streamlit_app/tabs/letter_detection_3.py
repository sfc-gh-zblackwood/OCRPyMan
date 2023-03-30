import streamlit as st

def show_content():
    st.subheader("Handwritting: the alignement problem")
    st.markdown(
        """
        The main difficulty to overcome in handwriting recognition is the random alignment between the handwritten input and the digital output along the horizontal axis.
        These two images have the same word but with different alignements (the annotations are different). But the model should output the same prediction. CTC solves this issue. 
        """
    )
    st.image("../images/letter_detection_DB401.png", width = 320)
    st.markdown("---")
    st.subheader("Connectionnist Temporal Classification (CTC)")
    st.markdown(
        """
        The model and training strategy are represented here:
        """
    )
    st.image("../images/letter_detection_DB40.png")
    st.markdown(
        """
        This training strategy can be summarized as follows:

        - The image is fed through different CNN layers for visual feature extraction.
        - The result of the previous step which can be seen as a group of timesteps is fed into a bidirectional RNN which will map the probabilities of each timestep fragment corresponding to a given character.
        - The CTC loss function will encode all the possible alignments (or paths) for the known output (transcription) on the probability matrix. Each path has a calculated probability.
        - The objective of the training is to maximize the probability of matching the output and to minimize the loss. So, the **loss is then calculated as the negative sum of log-probabilities of all the encoded paths**.
        - To train the model, the gradient of the loss function is calculated in respect to the different trainable parameters which are updated at each iteration.
            """
    )
    st.markdown("---")
    st.subheader("Model training & tuning")
    
    st.markdown(
        """
        Characteristics:
        - Model implemented with the TensorFlow framework
        - The model has 1.4 million trainable parameters
        - Callbacks:
            - Early stop('val_loss')
            - Checkpoint
            - Reduce learning rate on plateau
        - Optimizer: adam, learning_rate = .001
        - epochs = 20
        - batch size : downed to 32 because of crash when training on GPU
        - test set : 10% of the dataset
        """
    )
    st.markdown(
        """
        Regarding the model training, we have varied the number of epochs and the learning rate. Some training runs were made including the generated words. We kept here 4 significant models. 

        We can see below what happened in some of our tests. We observe improvements in each of these tests : the loss is decreasing consistently, and the model's precision is improving.
        """
    )
    st.image("../images/letter_detection_DB41.png")
    st.image("../images/letter_detection_DB42.png")
    st.image("../images/letter_detection_DB43.png")
    st.image("../images/letter_detection_DB44.png")
    st.markdown(" ")
    ####
    
    
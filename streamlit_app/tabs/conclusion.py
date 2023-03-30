import streamlit as st
from tabs import st_lib


sidebar_name = "Conclusion"


def run():
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.title("OCRpyMAN")
    st.markdown("---")

    st.subheader("Conclusion")
    
    st.markdown(
        """
The project conclusions concerning the project objectives are the following:

__Implementation of a Handwritten Text Recognition (HTR) model with Deep Learning (DL) techniques from scratch.__

We built and trained a CRNN to recognize the text in an image normally containing one word. From the results we got, we can conclude that our model has almost 94% precision per character and 86% per word when used with the database containing all the (well segmented) words.

__Implementation of a Text Detection model with the pretrained package docTR.__

We have refined the base DocTR detection model having already an excellent precision to reach 93%. The training had, however, a non-significant improvement impact on the model. 

__Combine the two previous models and present them through a Streamlit web application.__

At first, the two models were running separately, so we had to update the output of the first and the input of the second to make them work together. When done, it was possible to run the whole solution and get the results : an image as an input, some text as the output.

        """
        )
        
    st.subheader("Way forward")
    
    st.markdown("""
        Several improvements can be further made to this project. We list some of them here:

__Span of the dataset:__
Some characters were found to be under-represented on the dataset, especially for uppercase letters. One path of investigation would be to check if the classification errors of the text recognition model correlate with the under-representation of characters on the dataset. If confirmed, a possible solution would be to fine-tune the random word generator accordingly.

__Preprocessing:__
Image processing techniques like contour detection or enhancement can be used on word images to see if the text recognition performance can be improved.


__On the data augmentation functions, new techniques could be evaluated:__
compensating for the slanted writing,
introducing slight random rotations to the original images,
introducing better filtering for the background noise. 

__Random word generator:__
The decision was made in the project to generate words as an assembly of characters without lexical meaning. One path for improvement would be to generate words from the English language (or any other language) in a proportion representative of texts expected to be recognized.

__On the text recognition model, concerning the decoding algorithm when predicting a word:__
Our model uses a greedy decoder to go from a probability matrix to the predicted word. A beam search decoder may find better paths for the prediction with higher probability than the greedy decoder. However the performance will be impacted. An investigation into this trade between increased accuracy versus decreased performance is a possible path of investigation.
The use of the Google Colab Pro solution for training the model should be looked into again. If it works correctly, the performance improvement can be significant and this would allow to increase the number of parameters of the model and/or the size of the dataset.

__Spelling corrector:__
The spelling corrector can be improved to detect and adapt to the most likely language of the presented document.
A more advanced corrector, which takes into account the context of the word, could be used to avoid inconsistent corrections.
Another possibility still would be to create a spelling corrector coupled with a beam search decoder on the CTC model. When making a prediction, this kind of decoder will compute the probabilities for different predictions which could be coupled with a spelling corrector to find the highest probability word that is correctly spelled.

__Box detection:__
Having more time could have enabled us to redo the model training using different preprocess methods, even using data generation to make our model more flexible (to bend texts for example). 

__Text reconstruction:__
The bounding boxes from the detection model are not ordered as a human would read the text. We made a basic sorting of the boxes to get words in their natural order, but this is dependent on the mean height of lines. 

       
        """
    )

    st_lib.add_bottom_space()














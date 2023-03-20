import streamlit as st

def show_content():
    st.header("Data augmentation")
    st.subheader("#1: Image preprocessing")
    st.markdown(
        """
        The following Figure shows a batch of 64 original word images (without any preprocessing).
        """
    )
    st.image("../images/letter_detection_DB31.png")
    st.markdown(
        """
        Before being fed to a training model theses images with go through a preprocessing function which objectives are the following:
        - to introduce some random geometric variability to the images (as a reminder, the dataset has a limited number of writers),
        - to clean the images (for example, the background noise),
        - to harmonize the size of all files.
        These techniques were also tested with the ImageGenerator of the Keras package but this solution was not kept, essentially due to image stretching.
        """
    )
    st.markdown(
        """
        #### Aspect ratio

        The aspect ratio of each image is changed by tinkering randomly with the horizontal dimension. So in the following Figure, the images have all different stretches:
        - images stretched horizontally have a positive w_delta,
        - images stretched vertically have a negative w_delta.
        """
    )
    st.image("../images/letter_detection_DB32.png")
    st.markdown(
        """
        #### Resizing and shifting

        To be used on our model, images need to have the same size. For our model, the chosen size was 32 by 128 pixels.

        To harmonize the size of the images, padding is added either to horizontal dimension or vertical dimension. In the following Figure, all the images have been padded (padding is represented as gray bands for visualization) and are now the same size.

        Once padding is added, the image is shifted randomly from left to right (for images padded horizontally) or from top to bottom (for images padded vertically).The parameter ‘shift’ shows how much (from 0 to 1) images were shifted.
        """
    )
    st.image("../images/letter_detection_DB33.png")
    
    st.markdown(
        """
        #### Background noise filtering

        Different images have different levels of background noise. To help our classification algorithm to differentiate the letters and the background, a simple filtering technique was applied (threshold). Pixels with value < 0.8 (0 being white and 1 black) are considered noise. These pixels, shown in the following Figure in black color, are filtered out of all the images.
        """
    )
    st.image("../images/letter_detection_DB34.png")
    
    st.markdown("---")
    st.subheader("#2: Random word generator")

    st.markdown(
        """
        Another technique used for data augmentation of our dataset was to add automatically generated words using the fontpreview package with handwritten-like font types. The Figure below shows a sample of the generated words.
        """
    )
    st.image("../images/letter_detection_DB35.png")
    st.markdown(
        """
        Various elements vary randomly for each generated word:
        - the number of letters (between 2 and 11)
        - the font size (ranging from single to double)
        - a coefficient applied to the height of the image
        - whether or not there is a capital letter at the beginning of the word (30%).

        In this project, we chose to generate 50 000 words with this algorithm, equally distributed on 30 fonts, which represents a 52% increase in the size of the dataset. The word generator notebook runs independently from the rest of the project.

        It should be noted that since the generated words correspond to a random assembly of characters, they have no lexical meaning. Furthermore, because we are going to implement a spelling corrector to improve the accuracy of the classification model, these words are excluded from the test set.

        """
    )
    st.markdown(" ")
    
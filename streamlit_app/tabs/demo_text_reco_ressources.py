import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt


import sys
sys.path.insert(1, '../notebooks')
import preprocessing as pp
import letter_detection_utils as ld_util
import ressources as rss
import modele as mdl

from tabs import st_lib


models = [['tj_ctc_base_10epochs_LRe-4', 'Original dataset, 10 epochs, LR 1e-4'],
          ['tj_ctc_base_20epochs_LRe-4', 'Original dataset, 20 epochs, LR 1e-4'],
          ['tj_ctc_base_20epochs_LR-plateau', 'Original dataset, 20 epochs, LR on Plateau'],
          ['tj_ctc_augmented_20epochs_LR-plateau', 'Augmented dataset, 20 epochs, LR on Plateau']]

data_extract_words = '../images/data_extract/words'

CANVAS_DEFAULT_FILENAME = "tmp/canvas_file.png"


def show_data_extract():
    rss.init()
    
    st.write("See below the predictions with 4 different models for a selection of images.")
    # st.button(label='Make new predictions', on_click=show_data_extract)
    
    #liste les fichiers
    images = load_images_from_path(data_extract_words, 12)
    
    for model, title in models:           
        desc = f"""Predictions using a model with : **{title}**"""
        st.write(desc)
        texts = get_predictions(model, images)
        show_predictions_images(images, texts)
        
        st.text("")  # saut de ligne
        st.text("")  # saut de ligne
 

    
    
       
def show_local():
    st.write("Browse your local files to get an image prediction")
        
def on_image_uploaded(filename):
    global input_random_key
    input_random_key = st_lib.get_random_string()
    
    image  = ld_util.load_image(filename)       
    
    st.write("Image avant preprocessing :")
    fig = plt.figure()
    plt.imshow(image.numpy(), cmap='gray')
    plt.axis('off')
    st.pyplot(fig)
    
    
    image = ld_util.preprocess(image, img_size=rss.img_size,  data_augmentation=False, is_threshold=False)
    image = tf.expand_dims([image], -1)
    image = tf.squeeze(image, [3])
    
    text = get_predictions(models[3][0], image)
    
    st.write("Vous avez écrit le texte : ", text[0]) 
    st.write("")
    st.write("")
    
    st.write("Image après preprocessing :")
    fig = plt.figure()
    plt.imshow(image.numpy().reshape(32, 128), cmap='gray')
    plt.axis('off')
    st.pyplot(fig)
    # result.show(doc)
        
def show_drawing():
    st.write("Use your mouse to write and get a prediction")  
      
    st_lib.render_canvas(
            on_image_uploaded, 
            filename = CANVAS_DEFAULT_FILENAME
        )


@st.cache  
def get_predictions(model, images):
    loaded_model = tf.keras.models.load_model("../pickle/"+model, custom_objects={"CTCLoss": mdl.CTCLoss})
    
    text_probs = loaded_model.predict(images) 
    text = ld_util.greedy_decoder(text_probs, rss.charList)
    
    return text



def load_images_from_path(path, max_files=-1):
    
    #liste les fichiers
    images_path = pp.get_files(path, ext='png', sub=False)
    
    word_imgs = tf.zeros([1, 32, 128, 1])
    
    for image_path in images_path:
        image  = ld_util.load_image(image_path)       
        image = ld_util.preprocess(image, img_size=rss.img_size,  data_augmentation=False, is_threshold=True)
        image = tf.expand_dims([image], -1)
        image = tf.squeeze(image, [3])
        word_imgs = tf.concat([word_imgs, image], 0)
        
        
    # for i in range(len(word_imgs_prepro)):
    #         img = ld_util.process_1_img_from_form(img_path, *bounding_boxes_xyhw[i])
    #         img = tf.expand_dims([img], -1)
    #         img = tf.squeeze(img, [3])
    #     else:
    #         img = word_imgs[i] / 255
    #     word_imgs_prepro = tf.concat([word_imgs_prepro, img], 0)
    
    
    # Removing extra img due to initialization
    word_imgs = word_imgs[1:]
    
    if max_files > 0:
        shuffled_word_imgs = tf.random.shuffle(word_imgs)
        word_imgs = shuffled_word_imgs[:max_files]
        
    
    return word_imgs


def show_predictions_images(images, textes):
    fig = plt.figure(figsize=(20, 12))

    lin = 5
    col = 6
    
    # Parcourir les images et les étiquettes
    for i in range(len(textes)):
        img = images[i]
        label = textes[i]

        # Ajouter la sous-figure à la figure principale
        fig.add_subplot(lin, col, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(label, fontsize=20)
        plt.axis('off')

    # Afficher la figure dans Streamlit
    st.pyplot(fig)
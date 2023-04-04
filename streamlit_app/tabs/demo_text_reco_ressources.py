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

models_loaded={}

def show_data_extract():
    
    st.write("See below the predictions with 4 different models for a selection of images.")
    if st.button(label='Load new predictions') == True:
        images = load_images_from_path(data_extract_words, 12)
        for model, title in models:           
            desc = f"""Predictions using a model with : **{title}**"""
            st.write(desc)
            texts = get_predictions(model, images)
            show_predictions_images(images, texts)
            
            st.text("")  # saut de ligne
            st.text("")  # saut de ligne


        
    
def show_images_batch():
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



def remove_white_space(image):
    # Convertir l'image en niveaux de gris
    # image_gray = tf.image.rgb_to_grayscale(image)
    image_gray = image
    

    # Inverser les couleurs de l'image
    image_inv = tf.math.subtract(255, image_gray)

    # Appliquer un filtre de seuillage
    threshold_value = 200
    image_thresh = tf.where(image_inv > threshold_value, 255, 0)

    # Recadrer l'image autour du texte en supprimant les zones blanches
    bbox = tf.image.crop_to_bounding_box(image_thresh, 0, 0, tf.shape(image_thresh)[0], tf.shape(image_thresh)[1])
    bbox = tf.cast(bbox, tf.uint8)
    
    # Inverser les couleurs de l'image
    bbox_inv = tf.math.subtract(255, bbox)

    return bbox_inv


@tf.function     
def is_empty_image(image):
    # Convertir l'image en niveaux de gris
    # image_gray = tf.image.rgb_to_grayscale(image)

    # Somme de tous les pixels de l'image
    sum_of_pixels = tf.reduce_sum(image)

    # Vérifier si l'image contient quelque chose ou non
    if sum_of_pixels > 0:
        return False
    else:
        return True

@tf.function   
def on_image_uploaded(filename):
  
    image  = ld_util.load_image(filename)  
    if is_empty_image(image): 
        return
    
    # image = ld_util.crop_image(image.numpy())   
    # image = remove_white_space(image)      
    
    image = ld_util.preprocess(image, img_size=rss.img_size,  data_augmentation=False, is_threshold=False)
    image = tf.expand_dims([image], -1)
    image = tf.squeeze(image, [3])
    
    for i in range(len(models)):        
        text = get_predictions(models[i][0], image)    
        st.write("Utilisation du modele : ", models[i][1])
        st.write("Vous avez écrit le texte : ", text[0]) 
        st.write("")
        st.write("")
    
    
        
def show_drawing():    
    st.write("Use your mouse to write and get a prediction by clicking the \"Send to Streamlit\" button")  
      
    st_lib.render_canvas(
            on_image_uploaded, 
            filename = CANVAS_DEFAULT_FILENAME
        )



def get_predictions(model, images):
    loaded_model = load_model("../pickle/"+model, {"CTCLoss": mdl.CTCLoss}) # tf.keras.models.load_model("../pickle/"+model, custom_objects={"CTCLoss": mdl.CTCLoss})
    
    text_probs = loaded_model.predict(images) 
    text = ld_util.greedy_decoder(text_probs, rss.charList)
    
    return text

 
def load_model(path, cust_obj):
    if path in models_loaded.keys():
        return models_loaded[path]
    else:
        model = tf.keras.models.load_model(path, custom_objects=cust_obj)
        models_loaded[path] = model
        return models_loaded[path]

def load_images_from_path(path, max_files=-1):
    
    #liste les fichiers
    images_path = pp.get_files(path, ext='png', sub=False)
    
    if max_files > 0:
        shuffled_word_imgs = tf.random.shuffle(images_path)
        images_path = shuffled_word_imgs[:max_files]
        
    word_imgs = tf.zeros([1, 32, 128, 1])
    
    for image_path in images_path:
        image  = ld_util.load_image(image_path)       
        image = ld_util.preprocess(image, img_size=rss.img_size,  data_augmentation=False, is_threshold=True)
        image = tf.expand_dims([image], -1)
        image = tf.squeeze(image, [3])
        word_imgs = tf.concat([word_imgs, image], 0)
        
        

    # Removing extra img due to initialization
    word_imgs = word_imgs[1:]
    
    # if max_files > 0:
    #     shuffled_word_imgs = tf.random.shuffle(word_imgs)
    #     word_imgs = shuffled_word_imgs[:max_files]
        
    
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
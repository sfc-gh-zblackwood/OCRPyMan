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

data_extract_forms = '../images/data_extract/forms'
tmp_image = "tmp/tmp_image.png"
CANVAS_DEFAULT_FILENAME = "tmp/canvas_file_full.png"
UPLOADER_DEFAULT_FILENAME = "tmp/uploaded_form.png"
models_loaded={}

def show_data_extract():
        
    st.write("See below the predictions with 4 different models for one random form.")
    
    if st.button(label='Load new prediction') == True:
        image_path = random_file_path(data_extract_forms)        
        on_image_uploaded(image_path)
        

    
    
       
def show_local():
    st.write("Browse your local files to get an image prediction")    
    
    uploaded_file = st_lib.render_file_uploader(UPLOADER_DEFAULT_FILENAME, on_image_uploaded)

    # if uploaded_file is not None:
    #     file_path = st_lib.uploaded_file_manager.save_file(uploaded_file)
    #     st.write(f"Le fichier a été enregistré à l'emplacement suivant : {file_path}")


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
    
    if tf.is_tensor(filename):
        str_filename = filename.numpy().decode("utf-8")
    else:
        str_filename = filename
        
    text_detection_model = mdl.load_text_detection_model("../notebooks/text_detection/fine_tuning_final/weights")
    text_reco_model = load_model("../pickle/tj_ctc_augmented_20epochs_LR-plateau", {"CTCLoss": mdl.CTCLoss})
    
    text, fig = mdl.make_ocr(text_detection_model, text_reco_model, str_filename, with_display=True, return_fig=True)
    st.pyplot(fig)


    st.write("Prédictions : ", *text)

    
    # image = ld_util.preprocess(image, img_size=rss.img_size,  data_augmentation=False, is_threshold=False)
    # image = tf.expand_dims([image], -1)
    # image = tf.squeeze(image, [3])
    
    # for i in range(len(models)):        
    #     text = get_predictions(models[i][0], image)    
    #     st.write("Utilisation du modele : ", models[i][1])
    #     st.write("Vous avez écrit le texte : ", text[0]) 
    #     st.write("")
    #     st.write("")
    
    
        
def show_drawing():
    st.write("Use your mouse to write and get a prediction by clicking the \"Send to Streamlit\" button")  
      
    st_lib.render_canvas(
            on_image_uploaded, 
            filename = CANVAS_DEFAULT_FILENAME,
            size = (32 * 5 * 4, 128 * 5 *1.5)
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

def random_file_path(path, ext='png'):
    
    #liste les fichiers
    image_path = pp.get_files(path, ext=ext, sub=False)
    
    shuffled_word_imgs = tf.random.shuffle(image_path)
    image_path = shuffled_word_imgs[0]  # on prend arbitrairement le premier, c'est au hasard
    
    return image_path


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
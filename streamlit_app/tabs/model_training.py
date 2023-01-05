import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from matplotlib import cm

import sys
sys.path.insert(1, '../notebooks/')
import rendering as rd
import letter_detection_utils as ld_util
import ressources as rss
import modele as mdl


title = "Model training"
sidebar_name = "Model training"



def run():

    models = []
    models_desc = []

    
    st.title(title)

    st.markdown(
        """
        We train our model using a preprocess function applied on all input images which allows us to set a data augmentation
        and resize images to get them all at 128x32. 
        
        Depending on the model parameters and the data used for the training, the preformances will change : we look at the validation loss value, and the precision per word (named PPW from here).
        
        Our base database contains around 95k words. When using random generated words, we add 50k words.
        
        You can choose below which models and data to show in the graph :
        """
    )
    

    base_10epochs = st.checkbox('With original database, 10 epochs, LR=1.e-4') 
    base_20epochs = st.checkbox('With original database, 20 epochs, LR=1.e-4')
    base_20epochs_plateau = st.checkbox('With original database, 20 epochs, LR plateau') 

    

    if base_10epochs:
        models.append('tj_ctc_base_10epochs_LRe-4')
        models_desc.append('Base, 10 epochs, LR=0.0001. PPW : 86%')

    if base_20epochs:
        models.append('tj_ctc_base_20epochs_LRe-4')
        models_desc.append('Base, 20 epochs, LR=0.0001. PPW : 88%')
    
    if base_20epochs_plateau:
        models.append('tj_ctc_base_20epochs_LR-plateau')
        models_desc.append('Base, 20 epochs, LR plateau. PPW : 91%')

    
    
    show_loss(models, models_desc)
    


    # TO FINISH : pour utiliser cette fonction, il faudra importer un (petit) pack d'image dans le repo git et l'utiliser
    show_predicts(models, models_desc)
    
    # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))
    # st.line_chart(chart_data)

    # st.markdown(
    #     """
    #     ## Test 2
    #     The dataset is not perfect and some pictures were not 

    #     """
    # )

    # st.area_chart(chart_data)

    # st.markdown(
    #     """
    #     ## Test 3

    #     You can also display images using [Pillow](https://pillow.readthedocs.io/en/stable/index.html).

    #     ```python
    #     import streamlit as st
    #     from PIL import Image

    #     st.image(Image.open("assets/sample-image.jpg"))

    #     ```

    #     """
    # )

    # st.image(Image.open("assets/sample-image.jpg"))


def show_predicts(models, models_desc):
    

    ### Chargement des données ###
    dataset_test = tf.data.Dataset.load('../pickle/dataset_test')
    with open('../pickle/x_test.pickle', "rb") as file_pi:
        X_test = pickle.load(file_pi)
    with open('../pickle/y_test.pickle', "rb") as file_pi:
        y_test = pickle.load(file_pi)
    rss.init()
    ############################
    # TJ : maj du chemin, car les data viennent de l'autre machine
    X_test = [x.replace('C:/ocr_data/', '../data/') for x in X_test]
    
    for model_name in models:
        model = tf.keras.models.load_model("../pickle/"+model_name, custom_objects={"CTCLoss": mdl.CTCLoss})
        
        y_pred = model.predict(dataset_test)
        predicted_transcriptions = ld_util.greedy_decoder(y_pred, rss.charList)
        eval_data = list(zip(y_test, predicted_transcriptions))

        eval_df = pd.DataFrame(data=np.array(eval_data), columns=['real', 'predicted'])
        eval_df['cer'] = [ld_util.evaluate_character_level_accuracy(row.real, row.predicted) for index, row in eval_df.iterrows()]

        #  print("Le modèle ", model, " a une précision par mot de", eval_df['cer'].mean(), ' pour ', eval_df.shape[0], ' mots.')
        st.write('xtest0:', X_test[0])
        
        
        
        # rd.show_words_predictions_errors(X_test, y_test, y_pred, predicted_transcriptions)
        error_indexes = []
        for i in range(len(y_pred)):
            if (predicted_transcriptions[i] != y_test[i]):
                error_indexes += [i]

        j = 1
        fig = plt.figure(figsize=(20, 10))
        for i in np.random.choice(error_indexes, size = 20):
            img = cv2.imread(X_test[i]) 
            # img = img.reshape(32, 128)
            
            plt.subplot(4, 5, j)
            j = j + 1
            plt.axis('off')
            plt.imshow(img, cmap=cm.binary, interpolation='None')
            plt.title('True Label: ' + str(y_test[i]) \
                    + '\n' + 'Prediction: '+ str(predicted_transcriptions[i])) #\
                    #   + '\n' + 'Confidence: '+ str(round(test_pred[i][test_pred_class[i]], 2)))
        
        st.pyplot(fig)
    
    

def show_loss(models, models_desc):
    
    if len(models) == 0:
        return
    
    
    # plt.figure(figsize=(12,4))
    
    fig, ax = plt.subplots() # figsize=(15,10)
    
    
    for model in models:
        with open('../pickle/'+model+'.pickle', "rb") as file_pi:
            history = pickle.load(file_pi)   
        ax.plot(history['val_loss'])
        
    
    ax.grid(axis='y', which='minor')
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(models_desc, loc='upper right')
    
    
    # plt.show();
    st.pyplot(fig)


import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import re
import contextlib
import os
import matplotlib.patches as patches
import random
import sys
import string
from sklearn.model_selection import train_test_split
from sklearn.utils  import shuffle 

from ocr_project import ressources as rss
from ocr_project import preprocessing as pp


# load du pickle de base, generation d'un dataset avec map des fonctions de preprocessing et création des lots
# Attention, canny et augmented ne peuvent pas s'utiliser en meme temps (parceque les images canny sont générées indépendament, et non à la volée car ça crash, donc pas de canny pour les images générées)
def get_dataset(canny = False, augmented = False):    
    
    df = pd.read_pickle('../pickle/df.pickle')
    
    # on filtre les chaines vides et les caractères inconnus (TODO à déplacer dans le preprocess du dataframe?)
    df['clean_trans'] = df.transcription.apply(lambda x: extract_allowed_chars_from_string(rss.charList, x))
    df = df[(df['clean_trans'] != "") & (df['clean_trans'] == df['transcription'])]

    if canny:
        df['word_img_path'] = df['word_img_path'].apply(lambda x : '../data/canny/' + x.split('/')[-1])  # toutes les images au format canny seront stockées dans ce dossier

    elif augmented:
        # parcours des images générées artificiellement
        generated_images_path = '../data/generated/'
        generated_images = pp.get_files(generated_images_path, ext='png', sub=False)
        transcripts = [x.split('_')[-1][:-4] for x in generated_images]  # le nom de fichier contient la transcription
        # ajout au dataframe
        augmented_df = pd.DataFrame(list(zip(generated_images, transcripts)), columns=['word_img_path', 'transcription'])
        
        # Split features / target
        X_train_aug = augmented_df['word_img_path'].to_numpy()
        y_train_aug = augmented_df['transcription'].to_numpy()
            
    
    X_train, X_test, y_train, y_test = train_test_split(df['word_img_path'].values, df['transcription'].values, test_size=0.1, random_state=42)
    
    #Ajout des mots générés pour l'entrainement uniquement
    if augmented:
        X_train = np.concatenate((X_train, X_train_aug))
        y_train = np.concatenate((y_train, y_train_aug))

        X_train, y_train = shuffle(X_train, y_train)
        
        
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    if canny:
        dataset_train = dataset_train.map(process_1_img_canny)
        dataset_test = dataset_test.map(process_1_img_canny)
        
        # dataset_train = dataset_train.map(lambda x,y: tf.py_function(process_1_img_canny, [x, y], [tf.float32, tf.string]))            
        # dataset_test = dataset_test.map(lambda x,y: tf.py_function(process_1_img_canny, [x, y], [tf.float32, tf.string]))
        
        # pour debug :
        # img, y = tf.py_function(process_1_img_canny, [x, y], [tf.float32, tf.string])    
    
    else:
        dataset_train = dataset_train.map(process_1_img)
        dataset_test = dataset_test.map(process_1_img)
        
    dataset_train = dataset_train.batch(32)
    dataset_train = dataset_train.map(process_trancription)
    
    dataset_test = dataset_test.batch(32)
    dataset_test = dataset_test.map(process_trancription)
    
    
    # DEBUG :
    # for x,y in dataset_train:
    #     # process_trancription(x,y)
    #     print(x.numpy())
    #     # print(y)
    #     break
    
    # on renvoie aussi X_test, y_test car ils seront utilisés plus tard pour des comparaisons (à revoir!)
    return dataset_train, dataset_test, X_test, y_test

# Genere un dataset a partir des chemins des formulaire et des coordonnées des mots
# Si debug, alors on charge les données d'origines, sinon on charge les données fournies avec file_path et coords
def get_dataset_for_prediction(debug=False, file_path='', coords=None):    
            
    #DEBUG
    if debug:
        df = pd.read_pickle('../pickle/df.pickle')    
        # on filtre les chaines vides et les caractères inconnus (TODO à déplacer dans le preprocess du dataframe?)
        df['clean_trans'] = df.transcription.apply(lambda x: extract_allowed_chars_from_string(rss.charList, x))
        df = df[(df['clean_trans'] != "") & (df['clean_trans'] == df['transcription'])]
        df = df.head(20)    
        
        print(df['clean_trans'])
        for i in range(20):           
            # print('plouf', df.at[i,'form_img_path_y'], df.at[i,'y'], df.at[i,'x'], df.at[i,'h'], df.at[i,'w'])
            img = load_image_from_form(df.at[i,'form_img_path_y'], df.at[i,'y'], df.at[i,'x'], df.at[i,'h'], df.at[i,'w'])
            img = preprocess(img, img_size=rss.img_size,  data_augmentation=True, is_threshold=True)
            plt.imshow(img ,cmap='gray')
            plt.show()

        dataset_test = tf.data.Dataset.from_tensor_slices((df['form_img_path_y'], df['y'], df['x'], df['h'], df['w']))
    
    else:  # Cas "normal"    
        coords['file_path'] = file_path
        dataset_test = tf.data.Dataset.from_tensor_slices((coords['file_path'], coords['y'], coords['x'], coords['h'], coords['w']))

    dataset_test = dataset_test.map(process_1_img_from_form)
    dataset_test = dataset_test.batch(5)

    # DEBUG :
    # for x in dataset_test:
    #     # print('paf:', x)
    #     process_1_img_from_form(x)
    #     break
    
    return dataset_test

@tf.function
def process_1_img(x, y):
    path = x
     
    img = load_image(path)       
    img = preprocess(img, img_size=rss.img_size,  data_augmentation=True, is_threshold=True)

    return img, y

# crop une image numpy array
def crop_image(img):
    mask = img!=1
    mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    return img[np.ix_(mask1,mask0)]

@tf.function
def process_1_img_from_form(form_path, offset_height, offset_width, target_height, target_width, data_augmentation=False):
   
    img = load_image_from_form(form_path, offset_height, offset_width, target_height, target_width)     
    img = crop_image(img.numpy())      
    img = preprocess(img, img_size=rss.img_size,  data_augmentation=data_augmentation, is_threshold=True)
        
    return img

    
@tf.function
def process_1_img_canny(x, y):
    

    img = load_image(x) 
    img = preprocess(img, img_size=rss.img_size,  data_augmentation=True, is_threshold=True)  
    return img, y


@tf.function
def load_image(filepath):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=0)
    return im

@tf.function
def load_image_from_form(filepath, offset_height, offset_width, target_height, target_width):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=1)
    im = tf.image.crop_to_bounding_box(im, tf.cast(offset_width, tf.int32), tf.cast(offset_height, tf.int32), tf.cast(target_height, tf.int32), tf.cast(target_width, tf.int32))
    return im

### Specific methods

def show_mlp_result_for_row(X_test, y_test, y_pred, y_pred_proba, selected_row_index=0):
    selected_row = X_test[selected_row_index]
    predict_length = y_pred[selected_row_index]

    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.title("Proba d'être un mot de longueur...")
    plt.axvline(predict_length, color='red', ls='--')
    plt.axvline(y_test[selected_row_index], color='green')
    plt.plot(range(1, 1 + len(y_pred_proba[selected_row_index])), y_pred_proba[selected_row_index])

    plt.subplot(1,2,2),
    plt.title("Longueur de caractères: " + str(predict_length))
    plt.imshow(selected_row.reshape(32, 128), cmap='gray')
    plt.axis('off')
    plt.show()

def plot_avg_width_per_string_length(df):
    biggest_word_size = df['length'].max()
    width_means = []
    for l in range(1, biggest_word_size):
        width_means.append(df[df['length'] == l].w.mean())
        
    xaxis = range(1, biggest_word_size)
    plt.figure(figsize=(20,14))
    plt.title('Variation de la largeur en fonction de la taille du mot représenté')
    plt.xlabel('Taille du mot')
    plt.ylabel('Largeur du mot')
    plt.plot(xaxis, width_means, ls='--', color='navy');

    
### Processing
def process_df_img(df, img_size = (32, 128), with_edge_detection=True):
    nb_features = img_size[0] * img_size[1]
    
    # TJ à revoir : les images canny sont sur fond noir... il faudra surement les regénérer en inversant blanc/noir
    if with_edge_detection:
        data = np.ones((0, nb_features), float)
    else:
        data = np.empty((0, nb_features), float)
    
    for index, row in df.iterrows():
        path = row.word_img_path

        if with_edge_detection:
            file_name = path.split('/')[-1]
            path_tmp = '../data/temp/' + file_name 
            # new_row = cv2.Sobel(new_row,cv2.CV_64F,1,0, ksize=3)  # Sobel X
            # new_row = cv2.Sobel(new_row,cv2.CV_64F,0,1, ksize=5) # Sobel Y

            if not os.path.exists(path_tmp):
                image = cv2.imread(path) 
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(image, 30, 200)
                cv2.imwrite(path_tmp, edged)
            path = path_tmp
            # contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


          
        try:
            img = load_image(path) 
            new_row = preprocess(img, img_size=img_size,  data_augmentation=True, is_threshold=True).numpy()
            new_row = new_row.reshape(-1)
            data = np.append(data, [new_row], axis=0)
        except :
            print("Unexpected error:", sys.exc_info()[0])
        #     time.sleep(0.5)
    return data



# renvoie une chaine de caractere dont les caracteres non-autorisés (donc non présents dans char_list) ont été retirés
def extract_allowed_chars_from_string(char_list, str):
    res = ''
    for letter in str:
        if letter in char_list:
            res += letter
    return res


def process_trancription(x, y):
    return x, encode_labels(y, rss.charList)
    
def encode_labels(labels, charList):
    # Hash Table
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            charList,
            np.arange(len(charList)),
            value_dtype=tf.int32
        ),
        -1,  #TJ si label inconnu, alors -1 
        name='char2id'
    )
    return table.lookup(
    tf.compat.v1.string_split(labels, delimiter=''))
    
def decode_codes(codes, charList):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            np.arange(len(charList)),
            charList,
            key_dtype=tf.int32
        ),
        '',
        name='id2char'
    )
    return table.lookup(codes)

# Compte les majuscules et minuscules dans une chaine
def upper_lower(string): 
    upper = 0
    lower = 0
 
    for i in range(len(string)):         
        # For lower letters
        if (ord(string[i]) >= 97 and
            ord(string[i]) <= 122):
            lower += 1
 
        # For upper letters
        elif (ord(string[i]) >= 65 and
              ord(string[i]) <= 90):
            upper += 1
 
    print('Lower case characters = %s' %lower,
          'Upper case characters = %s' %upper)

@tf.function
def preprocess(img, img_size=(32, 128), data_augmentation=False, scale=0.8, is_threshold=False):
    # img = load_image(path)/255  
    
    img = img/255  # load_image(filepath)/255 # To work with values between 0 and 1
    
    padding_value = 1
    #####
    img_original_size = tf.shape(img)

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = tf.ones([img_size[0], img_size[1], 1])
        res = tf.expand_dims(img, -1)
        return res

    # increase dataset size by applying random stretches to the images
    if data_augmentation:

        # CHANGEMENT D'ASPECT RATIO
        # DEBUG DB : code initial corrigé pour fonctionner avec des images 32*128
        # INFO : scale est un scalaire entre 0 et +inf qui amplifie la distribution uniforme entre -50% et +50% de taille horizontale. si scale=1, pas d'effet sur la distribution uniforme
    
        stretch = scale*(tf.random.uniform([1], 0, 1)[0] - 0.5) # -0.5 .. +0.5
        w_stretched = tf.maximum(int(float(img_original_size[1]) * (1 + stretch)), 1) # random width, but at least 1
        img = tf.image.resize(img, (img_original_size[0] ,w_stretched)) # stretch horizontally by factor 0.5 .. 1.5


    # RESIZE DE L'IMAGE
    (ht, wt) = img_size
    h, w = float(tf.shape(img)[0]), float(tf.shape(img)[1])
    fx = h / ht
    fy = w / wt
    f = tf.maximum(fx, fy)
    newSize = (tf.minimum(ht,tf.maximum(1,int(h/f))),tf.minimum(wt,tf.maximum(1,int(w/f))))
    img = tf.image.resize(img, newSize)

    # AJOUT PADDING
    dx = ht - newSize[0]
    dy = wt - newSize[1]
    if data_augmentation:
        dx1=0
        dy1=0
        if dx != 0:
            dx1 = tf.random.uniform([1], 0, dx, tf.int32)[0]
        if dy != 0:
            dy1 = tf.random.uniform([1], 0, dy, tf.int32)[0]
        img = tf.pad(img[..., 0], [[dx1, dx-dx1], [dy1, dy-dy1]], constant_values=padding_value)
    else:
        img = tf.pad(img[..., 0], [[0, dx], [0, dy]], constant_values=padding_value)

    # FILTRE DU BRUIT DE L'IMAGE
    if is_threshold:
        img = 1-(1-img)*tf.cast(img < 0.6, tf.float32)

    img = tf.expand_dims(img, -1)
    return img


def greedy_decoder(logits, char_list):
    # ctc beam search decoder
    predicted_codes, _ = tf.nn.ctc_greedy_decoder(
        # shape of tensor [max_time x batch_size x num_classes] 
        tf.transpose(logits, (1, 0, 2)),
        [logits.shape[1]]*logits.shape[0]
    )
    # convert to int32
    codes = tf.cast(predicted_codes[0], tf.int32)
    # Decode the index of caracter
    text = decode_codes(codes, char_list)
    # Convert a SparseTensor to string
    text = tf.sparse.to_dense(text).numpy().astype(str)
    return list(map(lambda x: ''.join(x), text))


def levenshtein_distance(s, t):
    m, n = len(s) + 1, len(t) + 1
    d = [[0] * n for _ in range(m)]

    for i in range(1, m):
        d[i][0] = i

    for j in range(1, n):
        d[0][j] = j

    for j in range(1, n):
        for i in range(1, m):
            substitution_cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + substitution_cost)

    return d[m - 1][n - 1]

def evaluate_character_level_accuracy(original, predicted):
    original_length = len(original)
    if original_length == 0:
        return 1
    distance = levenshtein_distance(original, predicted)
    if distance > original_length:
        return 0
    return 1 - (distance / original_length)
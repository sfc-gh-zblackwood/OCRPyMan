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


### Image rendering
def show_plt_img(img):
    plt.imshow(img, cmap='gray');
    plt.axis('off');

def show_df_form_img(df, row_index):
    selected_row = df.iloc[row_index]
    img = plt.imread(selected_row.form_img_path)
    fig, ax = plt.subplots(figsize=(20,15))
    fig.figsize=(20,10)
    ax.imshow(img, cmap='gray')
    ax.add_patch(
    patches.Rectangle(
        (selected_row.x-8, selected_row.y-8),
        selected_row.w+16,
        selected_row.h+16,
        fill=False,
        color = 'red'      
    ) ) 
    plt.axis('off')
    plt.show()

def show_preprocess_img_from_df(df, row_index, img_size = (32, 128)):
    row = df.iloc[row_index]
    new_row = preprocess(row.word_img_path, img_size=img_size,  data_augmentation=True, is_threshold=True).numpy()
    plt.title(row.transcription + ' [' + str(row.length) + ']')
    plt.imshow(new_row, cmap='gray');
    plt.axis('off');

def show_preprocess_img_from_data(data, row_index, img_size = (32, 128)):
    img = data['preprocessed_imgs'][row_index].reshape(img_size)
    plt.imshow(img, cmap='gray');
    plt.axis('off');

def show_df_word_img(df, row_index):
    selected_row = df.iloc[row_index]
    img = plt.imread(selected_row.word_img_path)
    plt.figure(figsize = (10,8))
    plt.title("Texte: \"{}\" au format {} avec h={}, w={}".format(selected_row.transcription, img.shape, selected_row.h, selected_row.w));
    plt.axis('off')
    plt.imshow(img, cmap='gray');

def basic_bw_tensor_img_show(tensor_img, ax = None):
    if ax is None:
        ax = plt
    ax.axis('off')
    ax.imshow(tensor_img.numpy(), cmap='gray');

@tf.function
def load_image(filepath):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=0)
    return im


### Utility

def silence_method_call(callback=None, cargs=()):
    with open(os.devnull, 'w') as devnull_file:
        with contextlib.redirect_stdout(devnull_file):
            if callback is not None:
                return callback(*cargs)

def get_dataframe_with_preprocessed_imgs(nb_rows = 1000, img_size = (32, 128), load_pickle_if_exists = True, debug=True, pickle_name="letter_detection_data", with_edge_detection=True):
    full_df = pd.read_pickle('../pickle/df.pickle')
    if not pickle_name:
        raise Exception("Cannot have an empty pickle name")
    pickle_path = "../pickle/" + pickle_name + ".pickle"

    file_exists = os.path.exists(pickle_path)
    if file_exists and load_pickle_if_exists:
        if debug: 
            print("Loading existing data from ", pickle_path, "...")
        return pickle.load(open(pickle_path, "rb"))

    if debug: 
        print("Generating data...")
        

     # Only interested in letters, not punctation or decimal for the moment
    if debug: 
        print("Filtering data: taking only letters")
    r = r'[a-zA-Z]+'
    df = full_df[full_df['transcription'].str.contains(r)]
    np.random.seed(seed=42)

    # reducing row
    if nb_rows >= len(df):
        nb_rows = len(df)
        print('DataFrame only contains', len(df), ' rows => using full dataframe')
    if debug: 
        print("Using", nb_rows, "rows")

    df = df.iloc[random.sample(range(nb_rows), nb_rows)]

    df['length'] = df['transcription'].apply(lambda x: len(x.strip()))
    df.rename(columns = {'form_img_path_y': 'form_img_path'}, inplace = True)
    # reducing columns
    df = df[['michelson_contrast', 'gray_level_mot', 'word_id', 'gray_level', 'x', 'y', 'w', 'h', 'transcription', 'word_img_path', 'form_img_path', 'length']]
    df.reset_index(inplace=True)

    #filtrer les transcriptions vides
    df = df[df['length'] > 0]
    
    if debug: 
        print("Starting preprocessing of images with tensorflow")
        
    try:
        preprocessed_imgs = process_df_img(df, img_size, with_edge_detection=with_edge_detection)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        
    data = {
        'df': df,
        'preprocessed_imgs': preprocessed_imgs
    }
    if debug: 
        print("Creating pickle dump", pickle_path)
    pickle.dump(data, open(pickle_path, "wb" ))
    return data

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
            new_row = preprocess(path, img_size=img_size,  data_augmentation=True, is_threshold=True).numpy()
            new_row = new_row.reshape(-1)
            data = np.append(data, [new_row], axis=0)
        except :
            print("Unexpected error:", sys.exc_info()[0])
        #     time.sleep(0.5)
    return data


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
def preprocess(filepath, img_size=(32, 128), data_augmentation=False, scale=0.8, is_threshold=False, with_edge_detection=True):
    img = load_image(filepath)/255 # To work with values between 0 and 1
    img_original_size = tf.shape(img)

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = tf.ones([img_size[0], img_size[1], 1])
        res = tf.expand_dims(img, -1)
        return res

    # increase dataset size by applying random stretches to the images
    if data_augmentation:
        stretch = scale*(tf.random.uniform([1], 0, 1)[0] - 0.3) # -0.5 .. +0.5
        w_stretched = tf.maximum(int(float(img_original_size[0]) * (1 + stretch)), 1) # random width, but at least 1
        img = tf.image.resize(img, (w_stretched, img_original_size[1])) # stretch horizontally by factor 0.5 .. 1.5


    # Rescale
    # create target image and copy sample image into it
    (wt, ht) = img_size
    w, h = float(tf.shape(img)[0]), float(tf.shape(img)[1])
    fx = w / wt
    fy = h / ht
    f = tf.maximum(fx, fy)
    newSize = (tf.maximum(tf.minimum(wt, int(w / f)), 1), tf.maximum(tf.minimum(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = tf.image.resize(img, newSize)

    # Add padding
    dx = wt - newSize[0]
    dy = ht - newSize[1]
    if data_augmentation:
        dx1=0
        dy1=0
        if dx != 0:
            dx1 = tf.random.uniform([1], 0, dx, tf.int32)[0]
        if dy != 0:
            dy1 = tf.random.uniform([1], 0, dy, tf.int32)[0]
        img = tf.pad(img[..., 0], [[dx1, dx-dx1], [dy1, dy-dy1]], constant_values=1)
    else:
        img = tf.pad(img[..., 0], [[0, dx], [0, dy]], constant_values=1)

    if is_threshold:
        img = 1-(1-img)*tf.cast(img < 0.8, tf.float32)

    img = tf.expand_dims(img, -1)
    return img


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
from tqdm import tqdm

import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter

@tf.function
def load_image(filepath):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=0)
    return im

##
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def deskew_img(path):
    img = im.open(path)
    # convert to binary
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    plt.imshow(bin_img, cmap='gray')
    plt.savefig('binary.png')

    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle: {}'.formate(best_angle))
    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
    img.save('skew_corrected.png')

def thin_and_skeletonize_img(word_img_path):
    img = cv2.imread(word_img_path, 0)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img, kernel, iterations = 1)
    return erosion

@tf.function
def preprocess2(img, img_size=(32, 128), data_augmentation=False, scale=0.8, is_threshold=False, with_edge_detection=True):
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
        # Padding à droite
        img = tf.pad(img[..., 0], [[0, dx], [0, dy]], constant_values=1)

    if is_threshold:
        img = 1-(1-img)*tf.cast(img < 0.8, tf.float32)

    img = tf.expand_dims(img, -1)
    return img



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
        # Padding à droite
        img = tf.pad(img[..., 0], [[0, dx], [0, dy]], constant_values=1)

    if is_threshold:
        img = 1-(1-img)*tf.cast(img < 0.8, tf.float32)

    img = tf.expand_dims(img, -1)
    return img


def generate_preprocessed_imgs_from_df(df, img_size = (32, 128), folder_name = 'preprocessed_text_imgs', offset=0):
    # for index, row in (df.iterrows()):
    for index, row in tqdm(df.iterrows()):
        if index < offset :
            continue
        path = '../' +row.word_img_path
        img_array = preprocess(path, img_size=img_size,  data_augmentation=True, is_threshold=True).numpy()
        # new_row = cv2.Sobel(new_row,cv2.CV_64F,0,1, ksize=5) # Sobel Y
        # cv2.imwrite(folder_name + '/' + row.word_id + '.png', cv2.cvtColor(img_array * 255, cv2.COLOR))
        cv2.imwrite(folder_name + '/' + row.word_id + '.png', cv2.cvtColor(img_array * 255, cv2.COLOR_RGB2BGR))


def encode_labels(labels, char_list):
    # Hash Table
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            char_list,
            np.arange(len(char_list)),
            value_dtype=tf.int32
        ),
        -1,
        name='char2id'
    )
    return table.lookup(
    tf.compat.v1.string_split(labels, sep=''))

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
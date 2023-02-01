
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math 
from sklearn.model_selection import train_test_split
import tensorflow as tf

def render_theory():
    st.markdown(
        """
        As an introductory problem, we have been asked to predict 
        the number of letters in an image using only 1000 entries.
        """
    )

    st.subheader("Image pre-processing")

    st.markdown("""
    The previous preprocessing steps have been great to get rid of faulty images,
    but for this kind of task, we need to prepare our images to be efficiently used by our algorithm.

    Therefore, we have applied a more refine pre-processing method to improve the possible results
    of our model.
    """)
    data = pd.read_pickle('../pickle/letter_detection_data.pickle')
    df = data['df']
    st.dataframe(df.head(5))

    st.code(
        """
        @tf.function
        def preprocess(filepath, img_size=(32, 128), data_augmentation=False, scale=0.8, is_threshold=False, with_edge_detection=True):
            img = load_image(filepath)/255 # To work with values between 0 and 1
            img_original_size = tf.shape(img)

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
        """
    )
    st.subheader("A general idea of the expected result")

    st.markdown(
        """
        
        
        """
    )
    fig = plot_avg_width_per_string_length(df)
    st.pyplot(fig)

    # st.info('This is a purely informational message')
    st.subheader("Linear model")
    st.markdown(
        """
        Using this time the real dimension of each of our word images, we will try
        to infer the number of letters contained in a given image. 
        """
    )

    fig_linear = plot_linear_model(df)
    st.pyplot(fig_linear)

    st.subheader("MLP model")

    st.markdown(
        """
        MLP stands for Multi Layer Perceptron (Fully connected class of feedforward Artificial Neural Network).

        """
    )


    st.code(
        """
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout

        inputs = Input(shape=(X_train.shape[1], X_train.shape[2], 1))
        conv_1 = Conv2D(
            filters = 16,                    
            kernel_size = (5, 5),          
            activation = 'relu'
        )
        max_pool_1 = MaxPooling2D(pool_size = (2, 2))
        conv_2 = Conv2D(
            filters = 32,                    
            kernel_size = (3, 3),          
            activation = 'relu'
        )
        max_pool_2 = MaxPooling2D(pool_size = (2, 2))
        flatten = Flatten()
        dropout = Dropout(rate = 0.2)
        dense_1 = Dense(
            units = 128,
            activation = 'relu'
        )
        dense_2 = Dense(
            units = 1,
        )

        outputs=conv_1(inputs)
        outputs=max_pool_1(outputs)
        outputs=conv_2(outputs)
        outputs=max_pool_2(outputs)
        outputs=dropout(outputs)
        outputs=flatten(outputs)
        outputs=dense_1(outputs)
        outputs=dense_2(outputs)

        model = Model(inputs = inputs, outputs = outputs)
        model.compile(
            loss='mse', 
            optimizer='rmsprop',               
            metrics=['mae']
        )
        """
    )

    st.image(
        "../images/letter_counter_mlp_diff.png"
    )



@st.cache(allow_output_mutation=True)
def plot_avg_width_per_string_length(df):
    biggest_word_size = df['length'].max()
    reg_x = []
    reg_y = []

    width_means = []
    for l in range(1, biggest_word_size):
        m = df[df['length'] == l].w.mean()
        width_means.append(m)
        if not math.isnan(m):
            reg_x.append(l)
            reg_y.append(m)

    fig = plt.figure(figsize=(20, 14))

    xaxis = range(1, biggest_word_size)
    plt.title('Variation de la largeur en fonction de la taille du mot représenté')
    plt.xlabel('Taille du mot')
    plt.ylabel('Largeur du mot')
    plt.plot(xaxis, width_means, ls='--', color='navy')

    reg_x = np.array(reg_x).reshape(-1, 1)
    reg_y = np.array(reg_y).reshape(-1, 1)
    reg = LinearRegression().fit(reg_x, reg_y)
    score = reg.score(reg_x, reg_y)

    f_legend ="{} * X + {} = Y ($R^2$ = {})".format(reg.coef_[0][0].round(2), reg.intercept_[0].round(2), score.round(2))
    plt.plot(reg_x, reg.predict(reg_x), label=f_legend)
    plt.legend();
    return fig

@st.cache(allow_output_mutation=True)
def plot_linear_model(df):
    X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(df[['w']], df['length'], test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_lin_train, y_lin_train)
    score = lr.score(X_lin_test, y_lin_test)

    c = lr.coef_[0]
    offset = lr.intercept_
    xmin = df.w.min()
    xmax = df.w.max()
    X = np.linspace(xmin, xmax, 1000)
    y = X * c + offset
    
    f_legend ="{} * X + {} = Y ($R^2$ = {})".format(c.round(2), offset.round(2), score.round(2))

    fig = plt.figure(figsize=(20, 14))
    plt.plot(X, y);
    plt.scatter(df['w'], df['length'], ls='dotted', color='gold', label=f_legend)
    plt.legend();
    return fig

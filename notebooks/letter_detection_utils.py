import tensorflow as tf


@tf.function
def load_image(filepath):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=0)
    return im



@tf.function
def preprocess(filepath, imgSize=(32, 128), dataAugmentation=False, scale=0.8, isthreshold=False):

    img = load_image(filepath)/255

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = tf.ones([imgSize[0], imgSize[1], 1])

    # increase dataset size by applying random stretches to the images

    if dataAugmentation:
        stretch = scale*(tf.random.uniform([1], 0, 1)[0] - 0.3) # -0.5 .. +0.5
        wStretched = tf.maximum(int(float(tf.shape(img)[0]) * (1 + stretch)), 1) # random width, but at least 1
        img = tf.image.resize(img, (wStretched, tf.shape(img)[1])) # stretch horizontally by factor 0.5 .. 1.5


    # create target image and copy sample image into it
    (wt, ht) = imgSize

    w, h = float(tf.shape(img)[0]), float(tf.shape(img)[1])

    fx = w / wt
    fy = h / ht

    f = tf.maximum(fx, fy)

    newSize = (tf.maximum(tf.minimum(wt, int(w / f)), 1), tf.maximum(tf.minimum(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)

    img = tf.image.resize(img, newSize)



    dx = wt - newSize[0]
    dy = ht - newSize[1]

    if dataAugmentation :
        dx1=0
        dy1=0
        if dx!=0:
            dx1 = tf.random.uniform([1], 0, dx, tf.int32)[0]
        if dy!=0:
            dy1 = tf.random.uniform([1], 0, dy, tf.int32)[0]
        img = tf.pad(img[..., 0], [[dx1, dx-dx1], [dy1, dy-dy1]], constant_values=1)
    else :
        img = tf.pad(img[..., 0], [[0, dx], [0, dy]], constant_values=1)

    if isthreshold:
        return tf.expand_dims(1-(1-img)*tf.cast(img < 0.8, tf.float32), -1)
    return tf.expand_dims(img, -1)
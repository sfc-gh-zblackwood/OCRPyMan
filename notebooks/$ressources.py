DATA_PATH = "D:/$plouf/Formation DS/projet/dataset/data/"
WORDS_META_FILENAME = DATA_PATH + "ascii/words.txt"
FORMS_META_FILENAME = DATA_PATH + "ascii/forms.txt"
XML_FILES_PATH = DATA_PATH + "xml"
BASE_IMG_PATH = DATA_PATH + 'words'
COLUMNS = ['word_id', 'seg_res', 'gray_level', 'x', 'y', 'w', 'h', 'tag', 'transcription']




#parser les fichiers qui contiennent des commentaires (lignes commencant par #)
def parse_my_file(filename):
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            yield line.strip().split(' ',len(COLUMNS) - 1)

def get_word_image_path_by_word_id(word_id):
    path_parts = word_id.split('-')
    return BASE_IMG_PATH + '/' + path_parts[0] + '/' + "-".join(path_parts[0:2]) + '/' + word_id+ '.png'

def get_form_img_path_by_word_id(word_id):
    path_parts = word_id.split('-')
    first_letter = path_parts[0][0].lower()
    # if first_letter in ['a', 'b', 'c', 'd']:
    #     base_path = "../data/formsA-D/" 
    # elif first_letter in ['e', 'f', 'g', 'h']:
    #     base_path = "../data/formsE-H/" 
    # else:
    #     base_path = "../data/formsI-Z/" 
    base_path = DATA_PATH + "forms/"
    return base_path + "-".join(path_parts[0:2]) + '.png'







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


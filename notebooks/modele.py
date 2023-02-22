import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from doctr.models import ocr_predictor
from doctr.io import DocumentFile

from tensorflow.keras.layers import GRU, Bidirectional, Dense, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Dropout

import letter_detection_utils as ld_util
import ressources as rss

# fonction de perte utilisée à l'entrainement du modele
class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, logits_time_major=False, reduction=tf.keras.losses.Reduction.SUM, name='ctc'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        label_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_true)[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=-1)
        return tf.reduce_mean(loss)



# creation du modele, qui prend une image 128*32
def create_modele_128_32():
    model = tf.keras.Sequential()

    ############
    # Layer 1
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='SAME', input_shape = (128, 32, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Layer 2
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Layer 3
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2)))

    # Layer 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2)))

    # Layer 5
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2)))
    #####################


    # Remove axis 2
    model.add(Lambda(lambda x :tf.squeeze(x, axis=2)))
    numHidden = 256
    # Bidirectionnal RNN
    model.add(Bidirectional(GRU(numHidden, return_sequences=True)))
    model.add(Dense(100))
    model.summary()
    
    return model

# creation du modele, qui prend une image 32*128
def create_modele():
    model = tf.keras.Sequential()

    ############
    # Layer 1
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='SAME', input_shape = (32, 128, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Layer 2
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Layer 3
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

    # Layer 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

    # Layer 5
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))
    #####################


    # Remove axis 2
    model.add(Lambda(lambda x :tf.squeeze(x, axis=1)))  # pourquoi pas Flatten() ??
    numHidden = 256
    # Bidirectionnal RNN
    model.add(Bidirectional(GRU(numHidden, return_sequences=True)))
    model.add(Dense(100))
    model.summary()
    
    return model


def show_loss(history):
    ax = plt.figure(figsize=(12,4))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    plt.grid(axis='y', which='minor')
    plt.minorticks_on()
    plt.show();
    
def evaluate_prediction(target, predictions):
    eval_data = list(zip(target, predictions))

    eval_df = pd.DataFrame(data=np.array(eval_data), columns=['real', 'predicted'])
    eval_df.head(10)

    eval_df['cer'] = [ld_util.evaluate_character_level_accuracy(row.real, row.predicted) for index, row in eval_df.iterrows()]

    print("Notre modèle a une précision par mot de", eval_df['cer'].mean(), ' pour ', eval_df.shape[0], ' mots.')
    

def load_image(filepath, resize=None):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=0)
    if resize:
        return tf.image.resize(im, resize)
    return im

def load_text_detection_model():
    DET_CKPT = "text_detection/straight_model/weights"
    straight_model = ocr_predictor(det_arch='db_resnet50', pretrained=True)
    straight_model.det_predictor.model.load_weights(DET_CKPT)
    return straight_model

def get_image_bounding_boxes(img_path):
    straight_model = load_text_detection_model()
    doc = DocumentFile.from_images(img_path)
    return straight_model.det_predictor(doc)[0]

def format_bounding_boxes(bounding_boxes, size = (1, 1)):
    return [ [bbox[0] * size[1], bbox[1] * size[0], bbox[2] * size[1], bbox[3] * size[0]] for bbox in bounding_boxes ]

def format_bounding_boxes_xyhw(bounding_boxes, size = (1, 1)):
    return [ [bbox[0] * size[1], bbox[1] * size[0], (bbox[3] - bbox[1]) * size[0], (bbox[2] - bbox[0]) * size[1]] for bbox in bounding_boxes ]

def plot_bounding_boxes(bounding_boxes):
    for bbox in bounding_boxes:
        xmin = bbox[0]
        ymin = bbox[1] 
        xmax = bbox[2] 
        ymax = bbox[3] 
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='red')

def plot_bounding_boxes_xyhw(bounding_boxes):
    for bbox in bounding_boxes:
        xmin = bbox[0] 
        ymin = bbox[1] 
        xmax = (bbox[3] + bbox[0])
        ymax = (bbox[2] + bbox[1])
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='red')

def load_text_detection_model():
    DET_CKPT = "text_detection/straight_model/weights"
    straight_model = ocr_predictor(det_arch='db_resnet50', pretrained=True)
    straight_model.det_predictor.model.load_weights(DET_CKPT)
    return straight_model

def make_ocr(text_detection_model, recognition_model, img_path, with_display = False):
    # text_detection_model = load_text_detection_model()
    doc = DocumentFile.from_images(img_path)

    img_arr = load_image(img_path)
    img_size = (img_arr.shape[0], img_arr.shape[1])
    doctr_bboxes = text_detection_model.det_predictor(doc)[0]
    bounding_boxes = format_bounding_boxes(doctr_bboxes, (1,1))
    bounding_boxes_xyhw = format_bounding_boxes_xyhw(doctr_bboxes, img_size)

    box_texts = []

    # recognition_model = tf.keras.models.load_model("../pickle/tj_ctc_augmented_20epochs_LR-plateau", custom_objects={"CTCLoss": mdl.CTCLoss})
    
    # https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    word_imgs = tf.image.crop_and_resize(
        tf.expand_dims(img_arr, 0),
        boxes=list(map(lambda bbox: [bbox[1], bbox[0], bbox[3], bbox[2]], bounding_boxes)), # [y1, x1, y2, x2] NORMALIZED
        crop_size=[32, 128], # To resize the crop img to (32, 128)
        box_indices=[0 for i in range(len(bounding_boxes))] # We are always using the same img
    )
    
    # DEBUG
    # print("size :", img_size)
    # print("coords 0:", *bounding_boxes[0])
    # print("coords 1:", *bounding_boxes_xyhw[0])
    
    
    ### TEST AVEC NOTRE FONCTION PREPROCESS ###
    #TODO : virer cette fake premiere itération en utilisant un tensor vide?    word_imgs_prepro = tf.zeros([1, 128, 32, 1]) ?
    img = ld_util.process_1_img_from_form(img_path, *bounding_boxes_xyhw[0])
    img = tf.expand_dims([img], -1)
    img = tf.squeeze(img, [3])
    word_imgs_prepro = img
    
    for i in range(1,len(word_imgs)):
        img = ld_util.process_1_img_from_form(img_path, *bounding_boxes_xyhw[i])
        img = tf.expand_dims([img], -1)
        img = tf.squeeze(img, [3])
        word_imgs_prepro = tf.concat([word_imgs_prepro, img], 0)
    ######################################
    
    box_text_probs = recognition_model.predict(word_imgs_prepro) 
    box_text = ld_util.greedy_decoder(box_text_probs, rss.charList)
    box_texts.append(box_text)

    if with_display: 
        plt.imshow(img_arr, cmap='gray')
        for bounding_box in bounding_boxes_xyhw:
            x = bounding_box[0]
            y = bounding_box[1]
            h = bounding_box[2]
            w = bounding_box[3]
            plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y])
        plt.show()

    return box_texts

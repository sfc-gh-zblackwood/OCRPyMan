import matplotlib.pyplot as plt
import tensorflow as tf
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import streamlit as st

def load_image(filepath, resize=None):
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels=0)
    if resize:
        return tf.image.resize(im, resize)
    return im

def format_bounding_boxes(bounding_boxes, size = (1, 1)):
    return [[bbox[0] * size[1], bbox[1] * size[0], bbox[2] * size[1], bbox[3] * size[0]] for bbox in bounding_boxes ]

def format_bounding_boxes_xyhw(bounding_boxes, size = (1, 1)):
    return [[bbox[0] * size[1], bbox[1] * size[0], (bbox[3] - bbox[1]) * size[0], (bbox[2] - bbox[0]) * size[1]] for bbox in bounding_boxes ]

# @st.cache_resource
@st.cache(allow_output_mutation=True)
def load_text_detection_model(model_weight_path=None, paths = []):
    if model_weight_path == 'all' and len(paths) > 0:
        models = []
        for path in paths:
            model = ocr_predictor(det_arch='db_resnet50', pretrained=True)
            if not path is None:
                model.det_predictor.model.load_weights(path)
            models.append(model)
        return models
    model = ocr_predictor(det_arch='db_resnet50', pretrained=True)
    if not model_weight_path is None:
        model.det_predictor.model.load_weights(model_weight_path)
    return [model]

def plot_img_with_bboxes(img_arr, bounding_boxes_xyhw, figsize = (20, 15)):
    fig = plt.figure(figsize=figsize)
    plt.imshow(img_arr, cmap='gray')
    plt.axis('off')
    i = 0 
    for i, bounding_box in enumerate(bounding_boxes_xyhw):
        x = bounding_box[0]
        y = bounding_box[1]
        h = bounding_box[2]
        w = bounding_box[3]
        plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y])
    st.pyplot(fig)


def show_bbox_from_file(det_model, filepath, resize = None):
    doc = DocumentFile.from_images(filepath)
    img_arr = load_image(filepath, resize=resize)
    if not resize is None:
        img_arr /= 255
    res = det_model(doc)
    doctr_bboxes = det_model.det_predictor(doc)[0]
    bounding_boxes_xyhw = format_bounding_boxes_xyhw(doctr_bboxes, (img_arr.shape[0], img_arr.shape[1]))
    plot_img_with_bboxes(img_arr, bounding_boxes_xyhw)
    return doc, res

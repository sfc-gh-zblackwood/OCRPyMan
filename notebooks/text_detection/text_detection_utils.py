import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
import xml.etree.ElementTree as ET
import numpy as np
import hashlib
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import pandas as pd
import json 
from shapely.geometry import box
from typing import List, Tuple

IMG_DIM_ATTR_LABEL = 'img_dimensions'
IMG_HASH_ATTR_LABEL = 'img_hash'
POLYGON_ATTR_LABEL = 'polygons'
IS_TRAIN_ATTR_LABEL = 'is_train'
ROOT_FOLDER_PREFIX = '../../'

def plot_bounding_box(line):
    img = plt.imread('../' +line.form_img_path_y)

    fig, ax = plt.subplots(figsize=(20,15))
    fig.figsize=(20,10)
    ax.imshow(img, cmap='gray')

    ax.add_patch(
        patches.Rectangle(
            (line.x-8, line.y-8),
            line.w+16,
            line.h+16,
            fill=False,
            color = 'red'      
        ) ) 
    plt.axis('off')
    plt.show()

def get_img_hash(img_arr):
    return hashlib.sha256(img_arr).hexdigest()

def find_ymin_ymax_in_img(polygons):
    ymin = np.inf
    ymax = 0
    for polygon_data in polygons:
        for coord in polygon_data:
            y = coord[1]
            if y < ymin:
                ymin = y
            if ymax < y:
                ymax = y
    return (ymin, ymax)

def find_full_img_coords(clean_df):
    doctr_coords_json = {}

    model = ocr_predictor(
        det_arch = 'db_resnet50',    
        reco_arch = 'crnn_vgg16_bn', 
        pretrained = True
    )

    for form_img_path_y in list(set(clean_df.form_img_path_y)):
        form_img_path = '../' + form_img_path_y
        form_img_arr = plt.imread(form_img_path)

        doc = DocumentFile.from_images(form_img_path)
        det_res = model.det_predictor(doc)
        # det_res
        height = form_img_arr.shape[0]
        width = form_img_arr.shape[1]
        trans_coords = [[
            [arr[0] * width, arr[1] * height], #xmin #ymin
            [arr[2] * width, arr[1] * height], #xmax #ymin
            [arr[2] * width, arr[3] * height], #xmax #ymax
            [arr[0] * width, arr[3] * height], #xmin #ymax
        ] for arr in det_res[0]]
        doctr_coords_json[form_img_path] = trans_coords
        
    with open("doctr_coords.json", "w") as outfile:
        json.dump(doctr_coords_json, outfile, indent=4, sort_keys=False)

def split_in_train_val_img_dicts(clean_df, doctr_coords_json):
    train_img_dict = {}
    val_img_dict = {}
    i = 0
    nb_words = len(clean_df)
    for index, row in clean_df.iterrows():
        i += 1
        x = row.x
        y = row.y
        width = row.w
        height = row.h
        form_id = '../' + row.form_id
        form_img_path = '../' + row.form_img_path_y
        form_img_arr = plt.imread(form_img_path)

        is_train = i < int(nb_words * 0.8)

        box_coordinates = [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ]

        form_img_filename = row.form_id + '.png'
        form_img_filename = form_img_filename.split('/')[-1]
        
        if is_train:
            future_img_path = 'text_detection_train/'
            if not form_img_filename in train_img_dict:
                doctr_coords = doctr_coords_json[form_img_path]
                train_img_dict[form_img_filename] = {
                    IMG_DIM_ATTR_LABEL: form_img_arr.shape,
                    IMG_HASH_ATTR_LABEL: get_img_hash(form_img_arr),
                    POLYGON_ATTR_LABEL: doctr_coords,
                }
            train_img_dict[form_img_filename][POLYGON_ATTR_LABEL].append(box_coordinates)
        else: 
            future_img_path = 'text_detection_val/'
            if not form_img_filename in val_img_dict:
                doctr_coords = doctr_coords_json[form_img_path]
                val_img_dict[form_img_filename] = {
                    IMG_DIM_ATTR_LABEL: form_img_arr.shape,
                    IMG_HASH_ATTR_LABEL: get_img_hash(form_img_arr),
                    POLYGON_ATTR_LABEL: doctr_coords,
                }
            val_img_dict[form_img_filename][POLYGON_ATTR_LABEL].append(box_coordinates)

    with open("text_detection_train/labels.json", "w") as outfile:
        json.dump(train_img_dict, outfile, indent=4, sort_keys=False)
    with open("text_detection_val/labels.json", "w") as outfile:
        json.dump(val_img_dict, outfile, indent=4, sort_keys=False)

    return train_img_dict, val_img_dict


def show_boxes(model, img_path, should_print_res=False):
    doc = DocumentFile.from_images(img_path)
    result = model(doc)

    if should_print_res:
        print(result)
    text = ''
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                text += " ".join([word.value for word in line.words])
                text += "\n"
        
    if should_print_res:
        print(text)
    # Display the detection and recognition results on the image
    result.show(doc)

def plot_img_with_polygons(key, data):
    filepath = "text_detection_train/images/" + key

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(8)
    fig.set_dpi(142)
    plt.axis('off')
    for polygon_data in data['polygons']:
        polygon = Polygon(polygon_data)
        ax.add_patch(polygon)
    plt.imshow(plt.imread(filepath), cmap='gray')
    for x, y in data['polygons'][0]:
        plt.plot(x, y)
    plt.show()



def get_form_img_path_by_form_id(form_id):
    first_letter = form_id[0]
    if first_letter in ['a', 'b', 'c', 'd']:
        base_path =  "formsA-D/" 
    elif first_letter in ['e', 'f', 'g', 'h']:
        base_path = "formsE-H/" 
    else:
        base_path = "formsI-Z/" 
    return base_path + form_id +'.png'

def get_box_coordinates_from_geometry(geometry, img_arr = None):
    top_point, bottom_point = geometry
    if img_arr is None:
        return [
            [top_point[0], top_point[1]],
            [bottom_point[0], top_point[1]],
            [bottom_point[0], bottom_point[1]],
            [top_point[0], bottom_point[1]],
        ]
    return [
        [top_point[0] * img_arr.shape[1], top_point[1] * img_arr.shape[0]],
        [bottom_point[0] * img_arr.shape[1], top_point[1] * img_arr.shape[0]],
        [bottom_point[0] * img_arr.shape[1], bottom_point[1] * img_arr.shape[0]],
        [top_point[0] * img_arr.shape[1], bottom_point[1] * img_arr.shape[0]],
    ]

def get_all_found_box_coordinates(result, img_arr = None):
    return [
            get_box_coordinates_from_geometry(word.geometry, img_arr) 
            for page in result.pages for block in page.blocks for line in block.lines for word in line.words
        ]

def get_relative_img_path_from_form_id(form_id):
    return "../../data/" + get_form_img_path_by_form_id(form_id)

def get_box_coordinates_from_flat_arr_entry(entry, img_arr = None):
    top_point, bottom_point = (entry[2], entry[3]), (entry[0], entry[1])
    if img_arr is None:
        return [
            [top_point[0], top_point[1]],
            [bottom_point[0], top_point[1]],
            [bottom_point[0], bottom_point[1]],
            [top_point[0], bottom_point[1]],
        ]
    return [
        [top_point[0] * img_arr.shape[1], top_point[1] * img_arr.shape[0]],
        [bottom_point[0] * img_arr.shape[1], top_point[1] * img_arr.shape[0]],
        [bottom_point[0] * img_arr.shape[1], bottom_point[1] * img_arr.shape[0]],
        [top_point[0] * img_arr.shape[1], bottom_point[1] * img_arr.shape[0]],
    ]

def get_box_coordinates_from_flat_arr(flat_arr, img_arr = None):
    return [
            get_box_coordinates_from_flat_arr_entry(entry, img_arr) 
            for entry in flat_arr
        ]


def get_ground_truth_polygons_for_form(df: pd.DataFrame, form_id: str) -> List[box]:
    boxes = []
    rows = df[df['form_id'] == form_id]
    for index, row in rows.iterrows():
        boxes.append(box(row.x, row.y, row.x + row.w, row.h + row.y))
    return boxes

def get_ground_truth_polygon_boundaries(ground_truth_polygons: List[box], tolerance= 50):
    ymin = None
    ymax = None

    for b in ground_truth_polygons:
        for point in list(b.exterior.coords):
            if ymin is None:
                ymin = point[1]
            if ymax is None:
                ymax = point[1]

            if point[1] < ymin:
                ymin = point[1]
            if ymax < point[1]:
                ymax = point[1]
    return (ymin - tolerance, ymax + tolerance)

def get_predicted_polygons_for_form(model, form_id: str, boundaries = (700, 2700)) -> List[box]:
    img_path = get_relative_img_path_from_form_id(form_id)
    img_arr = plt.imread(img_path)
    boxes = []

    doc = DocumentFile.from_images(img_path)
    # Return an array of boxes' coordinates with normalized values
    res = model.det_predictor(doc)
    last_y_min = None
    for b in res[0]:
        xmin = img_arr.shape[1] * b[0]
        ymin = img_arr.shape[0] * b[1]
        xmax = img_arr.shape[1] * b[2]
        ymax = img_arr.shape[0] * b[3]
        if ymin > boundaries[0] and ymin < boundaries[1]:
        # if not is_filtering or (last_y_min and last_y_min + 250 > b[1]):
            boxes.append(box(
                xmin,
                ymin,
                xmax,
                ymax
                )
            )
        last_y_min = ymin
    return boxes

from shapely.geometry import Polygon, Point
import shapely
from shapely.ops import unary_union
import geopandas

# from shapely.ops import cascaded_union
# from shapely import union_all
# https://stackoverflow.com/questions/59308710/iou-of-multiple-boxes

def get_iou(ground_truth_polygons, predicted_polygons):
    ground_truth_union = Point(0, 0)
    for polygon in ground_truth_polygons:
        ground_truth_union = ground_truth_union.union(polygon)
    # ground_truth_union = unary_union(ground_truth_polygons)

    prediction_union = Point(0, 0)
    for polygon in predicted_polygons:
        prediction_union = prediction_union.union(polygon)
    # prediction_union = unary_union(prediction_polygons)
    
    area_of_union = ground_truth_union.union(prediction_union)
    area_of_intersection = ground_truth_union.intersection(prediction_union)

    return (geopandas.GeoSeries(area_of_intersection).area / geopandas.GeoSeries(area_of_union).area)[0]

def compute_iou(custom_model, clean_df, form_id):
    ground_truth_polygons = get_ground_truth_polygons_for_form(clean_df, form_id)
    boundaries = get_ground_truth_polygon_boundaries(ground_truth_polygons)
    predicted_polygons = get_predicted_polygons_for_form(custom_model, form_id, boundaries)
    return get_iou(ground_truth_polygons, predicted_polygons)

def show_img_with_prediction_and_iou(custom_model, clean_df, form_id):
    img_path = get_relative_img_path_from_form_id(form_id)

    ground_truth_polygons = get_ground_truth_polygons_for_form(clean_df, form_id)
    boundaries = get_ground_truth_polygon_boundaries(ground_truth_polygons)
    predicted_polygons = get_predicted_polygons_for_form(custom_model, form_id, boundaries)
    iou = get_iou(ground_truth_polygons, predicted_polygons)

    img_arr = plt.imread(img_path)
    plt.figure(figsize=(20,10))
    plt.imshow(img_arr)
    for blue in predicted_polygons:
        plt.plot(*blue.exterior.xy, color='b')
    for red in ground_truth_polygons:
        plt.plot(*red.exterior.xy, color='r')
    plt.title(f"Form {form_id} with IoU {iou:.2f}")


def show_bbox_from_file(model, filepath):
    custom_res = DocumentFile.from_images(filepath)
    custom_doc = model(custom_res)
    img_arr = plt.imread(filepath);

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(8)
    fig.set_dpi(142)
    plt.axis('off');

    box_coords = get_all_found_box_coordinates(custom_doc, img_arr)
    for box_coord in box_coords:
        polygon = Polygon(box_coord)
        ax.add_patch(polygon)
                    
    ax.imshow(img_arr, cmap='gray'); 
    return custom_doc, custom_res

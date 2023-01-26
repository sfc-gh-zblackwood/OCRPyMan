import streamlit as st
import pandas as pd
import numpy as np


title = "Text detection"
sidebar_name = "Text detection"


def run():

    st.title(title)

    st.markdown(
        """
        To do the text detection part, we have decided to make use of the Doctr
        API.

        Therefore, we neeeded at first to correctly format our dataset to match the requirements of this API.

        """
    )

    st.markdown(
        """
        We first run the doctr model to get the text coordinates.
        This additional step is run to later train our model with non-manuscript text as well,
        which is not labelised in our dataset.
        """
    )

    st.code(
        """
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
        """
    )


    st.markdown(
        """
        Once, we have all of the doctr result coordinates, we get our dataset coordinates in  
        the same format to combine them
        """
    )

    st.code(
        """
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
        """
    )

    st.image(
        "../images/text_detection_example.png"
    )

    st.code(
        """
        python doctr/references/detection/train_tensorflow.py text_detection_train text_detection_val  db_resnet50 --pretrained  
        """
    )



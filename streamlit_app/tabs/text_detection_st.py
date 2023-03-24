import streamlit as st
import matplotlib.pyplot as plt


def render_img_select():
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, figsize = (20,15))
    for i in range(nrows):
        for j in range(ncols):
            img_arr = plt.imread('assets/text_detection/a01-003.png')
            ax[i,j].imshow(img_arr, cmap='gray')
            ax[i,j].axis('off')
    st.pyplot(fig)

def show_comparison():
    st.header("Comparison")

    st.subheader("Using DocTR")
    st.image("../images/performance_fine_tuning_cropped.png", caption="Cropped fine tuning")
    st.image("../images/performance_fine_tuning_full.png", caption="Fine Tuning")
    st.markdown("""
    As we can see, the cropped model seems to have a better mean IoU (Intersect Over Union), 
    but is in general less performant than the model trained on the complete form images, the recall being
    significantly better (+6%) and precision (+3%).
    """)


    st.subheader("Using custom method")
    st.code("""
    def run_iou(custom_model, base_model = None, with_plot=False, is_saving=False):
        avg_iou = 0
        avg_base_iou = 0
        count = 0
        for form_id in list(set(df['form_id'])):
            form_df = df[df['form_id'] == form_id]
            all_form_writer_ids = list(set(form_df['writer_id']))
            for writer_id in all_form_writer_ids:
                tmp_df = form_df[form_df['writer_id'] == writer_id]
                iou, base_iou = compute_iou_from_form_df(custom_model, tmp_df, with_plot, is_saving, base_model=base_model)
                count = count + 1
                avg_iou = avg_iou + iou
                avg_base_iou = avg_base_iou + base_iou

        avg_iou = avg_iou / count
        avg_base_iou = avg_base_iou / count

        print("The average iou is {}".format(avg_iou))
        print("The average base iou is {}".format(avg_base_iou))
    """)
    st.markdown("""
    Using our custom method, we may think that the cropped model IoU is better, but our calculations
    are too vague. We do not take into account the IoU over each word but the **mean IoU of all text areas**.
    """)
    st.image("../images/iou.png")
    st.image("../images/iou_fine.png")

    st.markdown("""
    This however indicates us that we tend to have less precision in our model. It will eventually show bigger boxes
    that may englobed two differents words as one, which will be proven right experimentally.
    """)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

def show_theory():
    st.markdown(
        """
        To do the text detection part, we have decided to make use of the Doctr
        API.

        We needed first to correctly format our dataset to match the requirements of this API. 
        
        We have used two different approaches in our project:
        - using the whole form image
        - using a crop of the form image corresponding exclusively to the handwritten text.
        """
    )

    st.image(
        "assets/text_detection_model/text_detection_flow.png"
    )

    st.header("Using all document's text")

    st.markdown(
        """
        We first run the doctr model to get the text coordinates of the typed text.
        This additional step is mandatory as the model will be trained using 
        the full form image and the typed text is not labelised in our dataset.
        """
    )

    st.code(
        """
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

                ##########################
                FORMATING DB COORDINATES
                ##########################
                box_coordinates = [
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height]
                ]
                ##########################

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

                    ##########################
                    SPLITTING IN HIERARCHY
                    ##########################
            with open("text_detection_train/labels.json", "w") as outfile:
                json.dump(train_img_dict, outfile, indent=4, sort_keys=False)
            with open("text_detection_val/labels.json", "w") as outfile:
                json.dump(val_img_dict, outfile, indent=4, sort_keys=False)

            return train_img_dict, val_img_dict
        """
    )
    st.subheader("Colorized result of the detected boxes")

    st.image(
        "../images/text_detection_example.png"
    )

    st.markdown("""
    In all cases, we have to train the model using the given command coming from the API.
    """)

    st.code(
        """
        python doctr/references/detection/train_tensorflow.py text_detection_train 
        text_detection_val  db_resnet50 --pretrained  
        """
    )

    st.header("Using cropped image")

    st.markdown("""
    In order not to train the model using the model itself,
    We can also crop the image and adapt the coordinates coming from the database
    """)

    st.image(
        "assets/text_detection_model/cut_img_boxes.png"
    )


    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")



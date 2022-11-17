import ressources as rss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob  # pour la fonction get_files()
from lxml import etree  # pour la fonction get_words_from_xml_form()
import cv2


# Fonction rassemblant les étapes de preprocessing pour les formulaires
def prepro_form(form_df):
    # conversion de type
    form_df[['total_lines', 'correct_lines', 'total_words', 'correct_words']] = form_df[['total_lines', 'correct_lines', 'total_words', 'correct_words']].astype(int)
    # recupération ID text
    form_df['text_id'] = form_df.form_id.apply(lambda x: x[:3])
    # suppression des colonnes inutiles
    form_df = form_df[['form_id', 'writer_id', 'text_id', 'number_of_sentences', 'word_seg',
       'total_lines', 'correct_lines', 'total_words', 'correct_words']]
    # création du chemin d'accès à l'image correspondante au formulaire
    form_df['form_img_path'] = form_df['form_id'].apply(lambda x: get_form_img_path_by_form_id(x))

    return form_df

# Fonction rassemblant les étapes de preprocessing pour les mots
def prepro_word(word_df):
    word_df['x'] = word_df['x'].astype(int)
    word_df['y'] = word_df['y'].astype(int)
    word_df['w'] = word_df['w'].astype(int)
    word_df['h'] = word_df['h'].astype(int)
    word_df['seg_res'].replace(['ok', 'err'], [1, 0], inplace=True) 
    word_df['gray_level'] = word_df['gray_level'].astype(int)
    
    #on ne retient que les mots bien segmentés
    word_df = word_df[word_df['seg_res'] == 1]
    
    word_df['word_img_path'] = word_df['word_id'].apply(lambda x: get_word_image_path_by_word_id(x))
    word_df['form_img_path'] = word_df['word_id'].apply(lambda x: get_form_img_path_by_word_id(x))
    
    # LONG ! 8mn
    word_df = check_all_words_images(word_df)

    # LONG! 8mn
    word_df['michelson_contrast'] = word_df['word_img_path'].apply(lambda x: get_michelson_contrast(x))
    word_df = word_df[word_df['michelson_contrast'] > 0]
    
    # LONG! 3mn
    word_df['gray_level_mot'] = word_df.word_id.apply(get_img_mean)
    
    return word_df

def prepro_all(word_df, form_df):
    ### ajouter le form_id et le writer_id
    word_df['form_id'] = word_df.word_id.apply(lambda x: '-'.join(x.split(sep='-')[:2]))
    df = word_df.merge(right = form_df[['form_id','writer_id']], how='inner', on='form_id')
    df = pd.DataFrame(df.groupby(['transcription'])['writer_id'].value_counts())
    df = df.rename({'writer_id':'count'},axis=1).reset_index()
    
    return df
    


def parse_my_form_file(filename):
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            yield line.strip().split(' ')

def parse_my_word_file(filename):
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            yield line.strip().split(' ',len(rss.WORD_COLUMNS) - 1)


 
def check_all_forms_images(form_df):
    #Vérification de la possibilité d'ouvrir tous les fichiers des forms
    # list_corrupt=[]
    # for i in range(len(form_df)):
    #     try:
    #         plt.imread(form_df.form_img_path.iloc[i])
    #     except:
    #         print('Problème indice', str(i), form_df.form_img_path.iloc[i])
    #         list_corrupt.append(i)

    # if len(list_corrupt)==0: print('Tous les fichiers des forms sont accessibles')
    list_corrupt = check_all_files(form_df.form_img_path)
    if len(list_corrupt) != 0: 
        print('Certains fichiers des forms sont inaccessibles : on les retire du dataframe')
        form_df = form_df.drop(index = list_corrupt).reset_index()
    else:
        print('Tous les fichiers des forms sont accessibles')
        
    return form_df


def check_all_words_images(word_df):
    #Vérification de la possibilité d'ouvrir tous les fichiers des words
    list_corrupt = check_all_files(word_df.word_img_path)
    if len(list_corrupt) != 0: 
        print('Certains fichiers des mots sont inaccessibles : on les retire du dataframe')
        word_df = word_df.drop(index = list_corrupt).reset_index()
    else:
        print('Tous les fichiers des words sont accessibles')
        
    return word_df
    
def check_all_files(files_list):
    list_corrupt=[]
    for i in range(len(files_list)):
        try:
            plt.imread(files_list.iloc[i])
        except:
            print('Problème indice', str(i))
            list_corrupt.append(i)
    
    return list_corrupt
    
    
#Récupérer tous les mots d'un form
def get_words_from_xml_form(filepath):
    words = []
    tree = etree.parse(filepath)
    for word in tree.xpath("/form/handwritten-part/line/word"):
        words.append(word.get('text'))
    return words

# parcours de tous les fichiers d'un dossier et ses sous-dossiers, possibilité de spécifier une extension et la recherche parmis les sous dossiers
def get_files(path, ext='', sub=False):
    if sub == True:
        path = path + '/**'     
    if ext != '':
        path = path + '/*.' + ext
    else:
        path = path + '/*.*'  # le . filtre les dossiers  
    files = glob.glob(path, recursive=sub)
    return files
    

def get_form_img_path_by_form_id(form_id):
    first_letter = form_id[0]
    if first_letter in ['a', 'b', 'c', 'd']:
        base_path = rss.DATA_PATH + "formsA-D/" 
    elif first_letter in ['e', 'f', 'g', 'h']:
        base_path = rss.DATA_PATH + "formsE-H/" 
    else:
        base_path = rss.DATA_PATH + "formsI-Z/" 
    return base_path + form_id +'.png'


def get_word_image_path_by_word_id(word_id):
    path_parts = word_id.split('-')
    return rss.WORDS_IMG_PATH + '/' + path_parts[0] + '/' + "-".join(path_parts[0:2]) + '/' + word_id+ '.png'


def get_form_img_path_by_word_id(word_id):
    path_parts = word_id.split('-')
    first_letter = path_parts[0][0].lower()
    if first_letter in ['a', 'b', 'c', 'd']:
        base_path = "../data/formsA-D/" 
    elif first_letter in ['e', 'f', 'g', 'h']:
        base_path = "../data/formsE-H/" 
    else:
        base_path = "../data/formsI-Z/" 
    return base_path + "-".join(path_parts[0:2]) + '.png'


def get_michelson_contrast(img_path):
    """
        Retourne le contraste de Michelson pour un chemin d'image donné
        Les valeurs possibles sont comprises entre 0 et 1 et vaut -1 en cas d'erreur
    """
    try:
        img = cv2.imread(img_path)
        Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    except Exception:
        return -1
    min = int(np.min(Y))
    max = int(np.max(Y))
    if min == 0 and max == 0:
        return 0 
    return ((max - min) / (min + max))

def get_img_mean(id):
    image = plt.imread(get_word_image_path_by_word_id(id))
    return image.mean()

def get_letter_frequency_dict(text):
    dic = {}
    for l in text:
        if l in dic:
            dic[l] += 1
        else:
            dic[l] = 1
    return dic


def get_words_from_xml_files():
    #Chargement des informations depuis les fichiers xml (un xml par form)
    all_xml_files = get_files(rss.XML_FILES_PATH, ext='xml', sub=True)
    all_words = []
    for file in all_xml_files:
        all_words.extend(get_words_from_xml_form(file))
    all_letters = []
    for word in all_words:
        all_letters.extend(list(word))
    all_letters.sort()    

    # creation d'un dataframe avec tous les mots et leur longueur
    df_words = pd.DataFrame(all_words, columns=['words'])
    df_words['len'] = df_words['words'].apply(lambda w: len(w))

    #print('Mot pourri : ', df_words['words'][df_words['len'] == 53])
    #suppression du "faux" mot pour meilleure visu
    df_words = df_words.drop(index=107040)
    
    return df_words




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
import string
from fontpreview import FontPreview   # doc : https://fontpreview.readthedocs.io/en/latest/example.html
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import image 
import cv2
import numpy as np

import sys
sys.path.insert(1, '../')
import preprocessing as pp

generated_images_path = '../../data/generated/'

#########################################################################################################################
# Longueur dans l'usage : la longueur moyenne d'un mot communément acceptée est de 5 caractères ou lettres.

# En français, la longueur moyenne d'un mot est de 4,8 caractères.
# En anglais, la longueur moyenne d'un mot est de 4,7 caractères.


# Longueur dans le dictionnaire : la longueur moyenne d'un mot toutes langues confondues est d'environ 9.16 caractères

# En français, la longueur moyenne d'un mot est de 10,09 caractères.
# En anglais, la longueur moyenne d'un mot est de 8,23 caractères.
#########################################################################################################################

#Generation d'un mot aléatoire
def gen_word(font, dest):
    nb_letters = np.random.randint(2, 11)
    font_size = np.random.randint(30, 60)
    # coeff sur la taille
    size_coeff = np.random.randint(1, 50) / 100
    height = 100
    
    word = ''
    for i in range(0, nb_letters):
        random_letter = list(string.ascii_letters)[np.random.randint(0, len(string.ascii_letters)-1)].lower()  # rss.charList[np.random.randint(0, len(rss.charList)-1)]
        #random_letter = random_letter.strip().lower()
        word = word + random_letter

    # Majuscule en premiere lettre? 30% oui
    is_maj = np.random.randint(1, 10)
    if is_maj >= 7:
        word = word.capitalize() # converti la premiere lettre en majuscule
    
    
    fpp = FontPreview(font) 
    fpp.font_text = str(word)
    # fp.bg_color = (253, 194, 45)        # background color. RGB color: yellow
    fpp.dimension = (40*nb_letters, int(height + height*size_coeff))           # specify dimension in pixel: 300 x 250
    # fp.fg_color = (51, 153, 193)        # foreground or font color. RGB color: blue
    fpp.set_font_size(int(font_size + font_size*size_coeff))     
    fpp.set_text_position('center')        
    # before saving the image, you need to draw it
    fpp.draw()
    
    # print('word_images/' + filename + '.png')
    filename = font.split('\\')[-1][:-4] + '_' + word + '.png'   # exmple font = 'fonts\Amalfi Coast.ttf'
    fpp.save(dest + filename)
    
    #TODO trouver meilleur facon de faire...
    img = crop_image(image.imread(dest + filename))
    image.imsave(dest + filename, img)
    
    

# Pour générer x mots
def words_generator(nb_words):
    # Liste de toutes les fonts disponibles
    fonts = pp.get_files('fonts', ext='', sub=False)

    nb_img_per_font = nb_words // len(fonts)
    for font in fonts:
        for i in range(0,nb_img_per_font):
            gen_word(font, generated_images_path)

# crop une image numpy array
def crop_image(img):
    mask = img!=1
    mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    return img[np.ix_(mask1,mask0)]


def show_some_images():
    
    all_files = pp.get_files(generated_images_path, ext='png', sub=False)    
    print('Total files : ', len(all_files))

    j = 1
    plt.figure(figsize=(20, 30))
    for i in np.random.choice(len(all_files), size = 60):
        img = cv2.imread(all_files[i]) 
        # img = img.reshape(32, 128)
        
        plt.subplot(12, 5, j)
        j = j + 1
        plt.axis('on')
        plt.imshow(img, cmap=cm.binary, interpolation='None')
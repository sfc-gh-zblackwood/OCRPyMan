import string
from fontpreview import FontPreview
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import numpy as np


#########################################################################################################################
# Longueur dans l'usage : la longueur moyenne d'un mot communément acceptée est de 5 caractères ou lettres.

# En français, la longueur moyenne d'un mot est de 4,8 caractères.
# En anglais, la longueur moyenne d'un mot est de 4,7 caractères.


# Longueur dans le dictionnaire : la longueur moyenne d'un mot toutes langues confondues est d'environ 9.16 caractères

# En français, la longueur moyenne d'un mot est de 10,09 caractères.
# En anglais, la longueur moyenne d'un mot est de 8,23 caractères.
#########################################################################################################################

#Generation d'un mot aléatoire
def gen_word(font):
    nb_letters = np.random.randint(2, 14)
    font_size = np.random.randint(75, 150)
    # coeff sur la taille
    size_coeff = np.random.randint(1, 30) / 100
    height = 70
    
    word = ''
    for i in range(0, nb_letters):
        random_letter = list(string.ascii_letters)[np.random.randint(0, len(string.ascii_letters)-1)].lower()  # rss.charList[np.random.randint(0, len(rss.charList)-1)]
        #random_letter = random_letter.strip().lower()
        word = word + random_letter

    #TODO add upper first
    
    fpp = FontPreview(font) 
    fpp.font_text = str(word)
    # fp.bg_color = (253, 194, 45)        # background color. RGB color: yellow
    fpp.dimension = (15*nb_letters, int(height + height*size_coeff))           # specify dimension in pixel: 300 x 250
    # fp.fg_color = (51, 153, 193)        # foreground or font color. RGB color: blue
    fpp.set_font_size(int(font_size + font_size*size_coeff))       
    fpp.set_text_position('center')        
    # before saving the image, you need to draw it
    fpp.draw()
    
    # print('word_images/' + filename + '.png')
    filename = font.split('\\')[-1][:-4] + '_' + word + '.png'   # exmple font = 'fonts\Amalfi Coast.ttf'
    fpp.save('word_images/' + filename)


def show_some_images(all_files):
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
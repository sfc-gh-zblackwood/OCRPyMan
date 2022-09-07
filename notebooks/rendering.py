import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import preprocessing as pp


def show_img(path):
    img = plt.imread(path)
    plt.figure(figsize = (20,10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')


def show_word_image_by_line(line):
    show_img(line.word_img_path)


def show_form_img_by_word_id(line):
    show_img(line.form_img_path)


def plot_bounding_box(line):
    img = plt.imread(line.form_img_path)

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


def show_img_ax(path,ax):
    img = plt.imread(path)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img, cmap='gray')


def show_word_image_by_line_ax(line,ax):
    show_img_ax(line.word_img_path,ax)


def show_form_img_by_word_id_ax(line,ax):
    show_img_ax(line.form_img_path,ax)

   
def plot_bounding_box_ax(line,ax):
    img = plt.imread(line.form_img_path)

    ax.imshow(img, cmap='gray')

    ax.add_patch(
        patches.Rectangle(
            (line.x-8, line.y-8),
            line.w+16,
            line.h+16,
            fill=False,
            color = 'red'
        ) ) 
    ax.set_xticks([])
    ax.set_yticks([])


def plot_bounding_box_with_form(indice, df):    
    fig = plt.figure(figsize=(20,8))
    
    ax1 = fig.add_subplot(121)
    ax1.set_title('Mot identifié dans le Form : '+ str(df.word_id[indice][0:7]))
    plot_bounding_box_ax(df.iloc[indice],ax1)
    
    ax2 = fig.add_subplot(122)
    ax2.set_title('Mot : <'+ str(df.transcription[indice])+'>')
    show_word_image_by_line_ax(df.iloc[indice],ax2)

    fig.suptitle('Identification du mot d\'indice ' + str(indice) +'\nTranscription : <'+ str(df.transcription[indice])+'>', fontsize=20);




def show_6forms_from_writer000(form_df):
    form_df_000 = form_df[form_df['writer_id']=='000'].reset_index()

    n_lignes = 2
    n_colonnes = 3
    n_images = n_lignes * n_colonnes

    list_index = np.random.randint(0,form_df_000.shape[0], n_images)

    fig = plt.figure(figsize = (18,11))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        j = list_index[i]
        plt.imshow(plt.imread(form_df_000.form_img_path[j]), cmap = 'gray')
        plt.title(str(i+1)+') '+ form_df_000.form_id[j])
    
    fig.suptitle('Affichage de '+str(n_images)+' forms aléatoires du rédacteur 000', fontsize = 20);


def show_50_random_words(word_df):
    n_lignes = 5
    n_colonnes = 10
    n_images = n_lignes * n_colonnes

    rand_list = np.sort(np.random.randint(0, word_df.shape[0],n_images))

    fig = plt.figure(figsize = (20,10))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(word_df.word_id.iloc[rand_list[i]])), cmap = 'gray')
        plt.title('<'+str(word_df.transcription[rand_list[i]])+'>')

    fig.suptitle(t = 'Affichage de ' + str(n_images) + ' images aléatoires du dataset', fontsize = 20);

def show_images_with_contraste_0(word_df):    
    bad_contrast_df = word_df[word_df['michelson_contrast'] == 0].reset_index()
    
    n_lignes = 3
    n_colonnes = 5
    n_images = n_lignes * n_colonnes

    rand_list = np.sort(np.random.randint(0, bad_contrast_df.shape[0],n_images))

    fig = plt.figure(figsize = (20,8))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(bad_contrast_df.word_id.iloc[rand_list[i]])), cmap = 'gray')
        plt.title('<' + str(bad_contrast_df.transcription[rand_list[i]]) + '>')

    fig.suptitle(t = 'Affichage de ' + str(n_images) + ' images aléatoires du dataset qui présentent constraste = 0', fontsize = 20);


def show_100max_gray_level_images(word_df):
    n_lignes = 20
    n_colonnes = 5
    n_images = n_lignes * n_colonnes

    df_temp = word_df[word_df.seg_res==1].sort_values(by='gray_level', ascending = False, ignore_index=True)

    fig = plt.figure(figsize = (20,8))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(df_temp.word_id.iloc[i])), cmap = 'gray')
    
    fig.suptitle(t = 'Affichage des ' + str(n_images) + ' images de gray_level (ligne) le plus élevé', fontsize = 20);


def show_100min_gray_level_images(word_df):
    n_lignes = 20
    n_colonnes = 5
    n_images = n_lignes * n_colonnes

    df_temp = word_df[word_df.seg_res==1].sort_values(by='gray_level', ascending = True, ignore_index=True)

    fig = plt.figure(figsize = (20,8))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(df_temp.word_id.iloc[i])), cmap = 'gray')
    
    fig.suptitle(t = 'Affichage des ' + str(n_images) + ' images de gray_level (ligne) le plus bas', fontsize = 20);


def show_10_forms(form_df):
    list_index = np.random.randint(0,form_df.shape[0], 10)

    fig = plt.figure(figsize = (20,10))

    for i in range(10):
        fig.add_subplot(2,5,i+1)
        j = list_index[i]
        plt.imshow(plt.imread(form_df.form_img_path[j]), cmap = 'gray')
        plt.title(str(i+1)+') '+ form_df.form_id[j])

    fig.suptitle('Affichage de 10 forms aléatoires', fontsize = 20);


def show_75max_gray_level_word_images(word_df):
    n_lignes = 15
    n_colonnes = 5
    n_images = n_lignes * n_colonnes

    df_temp = word_df[word_df['seg_res']==1].sort_values(by='gray_level_mot', ascending = False, ignore_index=True)

    fig = plt.figure(figsize = (20,8))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(df_temp.word_id.iloc[i])), cmap = 'gray')
    
    fig.suptitle(t = 'Affichage des ' + str(n_images) + ' images de gray_level (mot) le plus élevé', fontsize = 20);


def show_75min_gray_level_word_images(word_df):
    n_lignes = 15
    n_colonnes = 5
    n_images = n_lignes * n_colonnes

    df_temp = word_df[(word_df['seg_res']==1) & (word_df['gray_level_mot']>0)].sort_values(by='gray_level_mot', ascending = True, ignore_index=True)

    fig = plt.figure(figsize = (20,8))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(df_temp.word_id.iloc[i])), cmap = 'gray')
    
    fig.suptitle(t = 'Affichage des ' + str(n_images) + ' images de gray_level (mot) le plus bas', fontsize = 20);


def show_45max_box_size_nosegmented_words_images(word_df):
    word_df['size'] = word_df.h * word_df.w

    n_lignes = 15
    n_colonnes = 3
    n_images = n_lignes * n_colonnes

    df_temp = word_df.sort_values(by='size', ascending = False, ignore_index=True)

    fig = plt.figure(figsize = (20,10))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(df_temp.word_id.iloc[i])), cmap = 'gray')
    
    fig.suptitle(t = 'Affichage des ' + str(n_images) + ' images de taille la plus élevée', fontsize = 20);

def show_45max_box_size_segmented_words_images(word_df):
    word_df['size'] = word_df.h * word_df.w

    n_lignes = 15
    n_colonnes = 3
    n_images = n_lignes * n_colonnes

    df_temp = word_df[word_df['seg_res']==1].sort_values(by='size', ascending = False, ignore_index=True)

    fig = plt.figure(figsize = (20,8))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(df_temp.word_id.iloc[i])), cmap = 'gray')
    
    fig.suptitle(t = 'Affichage des ' + str(n_images) + ' images de taille la plus élevée (segmentation ok)', fontsize = 20);


def show_100min_box_size_words_images(word_df):
    word_df['size'] = word_df.h * word_df.w

    n_lignes = 10
    n_colonnes = 10
    n_images = n_lignes * n_colonnes

    df_temp = word_df.sort_values(by='size', ascending = True, ignore_index=True)

    fig = plt.figure(figsize = (20,8))

    for i in range(n_images):
        fig.add_subplot(n_lignes,n_colonnes,i+1)
        plt.imshow(plt.imread(pp.get_word_image_path_by_word_id(df_temp.word_id.iloc[i])), cmap = 'gray')
    
    fig.suptitle(t = 'Affichage des ' + str(n_images) + ' images de taille la plus basse', fontsize = 20);





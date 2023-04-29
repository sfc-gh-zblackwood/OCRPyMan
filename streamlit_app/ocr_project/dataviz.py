import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ocr_project import preprocessing as pp
from ocr_project import rendering as rd
import pandas as pd

from collections import Counter
from functools import reduce


# Affiche le nombre de lignes et mots par forms
def show_lines_word_per_form(form_df):
    fig = plt.figure(figsize = (20,6))

    ax1 = fig.add_subplot(121)
    sns.histplot(x = 'total_lines' ,data = form_df, ax=ax1, color='red', label = 'total', bins=10)
    sns.histplot(x = 'correct_lines' ,data = form_df, ax=ax1, label = 'segmentés', bins=10)
    ax1.set_title('Histogramme du nombre de lignes dans les forms')
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(122)
    sns.histplot(x = 'total_words' ,data = form_df, ax=ax2, color = 'red', label = 'total', bins=10)
    sns.histplot(x = 'correct_words', data = form_df, ax=ax2, label = 'segmentés', bins=10)
    ax2.set_title('Histogramme du nombre de mots dans les forms')
    ax2.legend()
    ax2.grid()

    fig.suptitle('Distribution du nombre de lignes et mots par form', fontsize = 20);


def show_count_per_wirter(df):
    plt.figure(figsize=(15,5))
    sns.histplot(df)
    plt.xlabel('rédacteur')
    plt.title('Histogramme du nombre de textes par rédacteur')
    plt.grid();

def show_nb_words_having_some_contrast_0(word_df):
    bad_contrast_df = word_df[word_df['michelson_contrast'] == 0].reset_index()
    
    liste_mots_0 = bad_contrast_df.groupby('transcription').count().index
    group_mots_0 = word_df[word_df.transcription.isin(liste_mots_0)]

    plt.figure(figsize=(15,5))
    sns.countplot(x='transcription', data = group_mots_0, order = bad_contrast_df.transcription.value_counts().index)
    plt.title('Occurrence totale des mots dont certaines images ont un contraste égale = 0')
    plt.grid();
    
    
def show_contrast_distribution(word_df):
    fig = plt.figure(figsize=(20,8))

    ax1 = fig.add_subplot(121)
    ax1.set_title('Boxplot')
    sns.boxplot(y='michelson_contrast', data=word_df);

    ax2 = fig.add_subplot(122)
    sns.histplot(data = word_df, x= 'michelson_contrast', kde = True)
    ax2.set_title('Histogramme')
    plt.grid()

    fig.suptitle('Distribution de michelson_contrast des mots du dataset', fontsize=15);


def show_min_max_contrast_images(word_df):
    fig = plt.figure(figsize=(20,5))

    ax1 = fig.add_subplot(121)
    ax1.set_title('Mot de contraste le plus faible')
    ind_min = word_df.michelson_contrast.sort_values(ascending=True).index[0]
    rd.show_word_image_by_line_ax(word_df.iloc[ind_min], ax1)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Mot de contraste le plus élevé')
    ind_max = word_df.michelson_contrast.sort_values(ascending=False).index[0]
    rd.show_word_image_by_line_ax(word_df.iloc[ind_max], ax2)

    fig.suptitle('Images de michelson_contrast le plus élevé et le plus bas', fontsize=15);


def show_gray_level_distribution(word_df):
    fig = plt.figure(figsize=(20,8))

    ax1 = fig.add_subplot(121)
    ax1.set_title('Boxplot')
    sns.boxplot(y='gray_level', data=word_df);

    ax2 = fig.add_subplot(122)
    sns.histplot(data = word_df, x= 'gray_level', kde = True)
    ax2.set_title('Histogramme')
    plt.grid()

    fig.suptitle('Distribution de gray_level des mots du dataset\n(gray_level de la ligne à laquelle le mot appartient) ', fontsize=15);


def show_min_max_gray_level_images(word_df):
    min_gray_level = word_df['gray_level'].min()
    max_gray_level = word_df['gray_level'].max()

    low_gray_level_line_index = word_df[word_df['gray_level'] == min_gray_level].index.tolist()[0]
    high_gray_level_line_index = word_df[word_df['gray_level'] == max_gray_level].index.tolist()[0]

    low_gray_form_img_path =  word_df.iloc[low_gray_level_line_index].form_img_path
    low_gray_letter_img_path = word_df.iloc[low_gray_level_line_index].word_img_path

    high_gray_form_img_path = word_df.iloc[high_gray_level_line_index].form_img_path
    high_gray_letter_img_path = word_df.iloc[high_gray_level_line_index].word_img_path

    low_gray_form_img = plt.imread(low_gray_form_img_path)
    low_gray_letter_img = plt.imread(low_gray_letter_img_path)

    high_gray_form_img = plt.imread(high_gray_form_img_path)
    high_gray_letter_img = plt.imread(high_gray_letter_img_path)


    fig,ax = plt.subplots(1, 2, figsize = (20,10))

    ax = ax.ravel()
    ax[0].imshow(low_gray_letter_img, cmap='gray')
    ax[1].imshow(high_gray_letter_img, cmap='gray')
    plt.show()

    fig,ax = plt.subplots(1, 2, figsize = (20,10))
    ax[0].imshow(low_gray_form_img, cmap='gray')
    ax[1].imshow(high_gray_form_img, cmap='gray')
    plt.show()


def show_gray_level_per_word(word_df):
    fig = plt.figure(figsize=(20,8))

    ax1 = fig.add_subplot(121)
    ax1.set_title('Boxplot')
    sns.boxplot(y='gray_level_mot', data=word_df);

    ax2 = fig.add_subplot(122)
    sns.histplot(data = word_df, x= 'gray_level_mot', kde = True)
    ax2.set_title('Histogramme')
    plt.grid()

    fig.suptitle('Distribution de gray_level des mots du dataset', fontsize=15);


def show_letters_occurences(df_all):
    labels, counts = get_letters_count(df_all)

    plt.figure(figsize=(20,8))
    plt.title("Représentation du nombre d'occurences de chaque caractère dans le dataset", fontsize=15)
    plt.bar(labels, counts)
    plt.grid()
    plt.show()


def show_lower_letters_frequency(df_all):
    labels, counts = get_letters_count(df_all)
        
    alphabet_start = labels.index('a')

    alphabet_labels = labels[alphabet_start:]
    alphabet_counts = counts[alphabet_start:]
    total_alphabet_occurences = reduce(lambda a, b: a + b,  alphabet_counts)
    colors = [('green' if p > 1000 else 'red') for p in alphabet_counts]

    plt.figure(figsize=(20,10))
    plt.title("Fréquences des lettres minuscules dans le corpus", fontsize=15)
    plt.bar(alphabet_labels, [count / total_alphabet_occurences for count in alphabet_counts], color=colors)
    plt.grid()
    plt.show()


def show_upper_letters_frequency(df_all):
    labels, counts = get_letters_count(df_all)
    
    alphabet_start = labels.index('A')
    alphabet_end = labels.index('Z')

    alphabet_labels = labels[alphabet_start:alphabet_end+1]
    alphabet_counts = counts[alphabet_start:alphabet_end+1]
    colors = [('green' if p > 100 else 'red') for p in alphabet_counts]

    plt.figure(figsize=(20,10))
    plt.title("Occurences des lettres majuscules dans le corpus")
    plt.bar(alphabet_labels, [count for count in alphabet_counts], color=colors)
    plt.show()


def get_letters_count(df_all):
    corpus = "".join([trans for trans in df_all['transcription']])
    freq_dict = pp.get_letter_frequency_dict(corpus)
    
    keys = list(freq_dict.keys())
    values = list(freq_dict.values())
    sorted_indexes = np.argsort(keys)
    labels = []
    counts = []
    for index in sorted_indexes:
        labels.append(keys[index])
        counts.append(values[index])
    
    return labels, counts


def show_letters_repartition_with_english(df_all):
    labels, counts = get_letters_count(df_all)        
    alphabet_start = labels.index('a')
    alphabet_labels = labels[alphabet_start:]
    alphabet_counts = counts[alphabet_start:]
    
    df_alphabet = pd.DataFrame(list(zip(alphabet_labels,alphabet_counts)), columns = ['label','data_count'])
    df_alphabet['data_percentage'] = df_alphabet['data_count'] / df_alphabet['data_count'].sum()

    english_labels = ['E','A','R','I','O','T','N','S','L','C','U','D','P','M','H','G','B','F','Y','W','K','V','X','Z','J','Q']
    english_counts= [0.111607,0.084966,0.075809,0.075448,0.071635,0.069509,0.066544,0.057351,0.054893,0.045388,0.036308,0.033844,0.031671,0.030129,0.030034,0.024705,0.02072,0.018121,0.017779,0.012899,0.011016,0.010074,0.002902,0.002722,0.001965,0.001962]

    df_alphabet_english = pd.DataFrame(list(zip(english_labels,english_counts)), columns =['label', 'english_percentage'])
    df_alphabet_english.label = df_alphabet_english.label.apply(lambda x: x.lower())

    df_alphabet = df_alphabet.merge(right = df_alphabet_english, on='label')

    plt.figure(figsize = (9,5))
    sns.scatterplot(data=df_alphabet, x='english_percentage', y='data_percentage')
    plt.title('Fréquence des lettres dans le corpus vs la langue anglaise')
    plt.plot([0,.14],[0,.14], ls='--', color='gray')
    plt.xlim(0,.14)
    plt.ylim(0,.14)
    plt.grid()

    j=0
    for i in range(len(df_alphabet)):
        if df_alphabet.data_percentage.iloc[i] < .005:
            plt.text(x = .005, y=.002 - j * .005,s = df_alphabet.label.iloc[i])
            j+=1
        elif df_alphabet.data_percentage.iloc[i] >= df_alphabet.english_percentage.iloc[i]:
            plt.text(x = (df_alphabet.english_percentage.iloc[i] - .002), y=(df_alphabet.data_percentage.iloc[i] + .002), s = df_alphabet.label.iloc[i])
        else :
            plt.text(x = (df_alphabet.english_percentage.iloc[i] + .002), y=(df_alphabet.data_percentage.iloc[i] - .002), s = df_alphabet.label.iloc[i])


    plt.fill_between(x=[0,.14], y1=[0,0], y2=[0,.14],alpha = .1, color='red')
    plt.fill_between(x=[0,.14], y1=[0,.14], y2=[.14,.14],alpha = .1, color='green')

    plt.text(x=.07, y=.02,s='Lettres sous-représentées dans le dataset');
    plt.text(x=.02,y=.11,s='Lettres sur-représentées dans le dataset');


def show_text_position_over_dataset(df_all):
    arr = np.zeros((3542, 2479))
    for index, row in df_all.iterrows():
        x = row.x
        y = row.y
        arr[y][x] += 1
        for i in range(row.h):
            for j in range(row.w):
                arr[y + i][x + j] += 1

    plt.figure(figsize=(20,10))
    plt.imshow(arr);


def show_bounding_boxes_size(word_df):
    word_df['size'] = word_df.h * word_df.w

    plt.figure(figsize = (15,5))
    sns.histplot(data = word_df[word_df.seg_res==1], x= 'size', kde = True)
    plt.title('Distribution de la surface des bounding box des mots (segmantation ok)')
    plt.grid()

    a = plt.axes([.3, .35, .3, .4])
    sns.histplot(data = word_df[word_df.seg_res==1], x= 'size', kde = True)
    plt.title('Zoom')
    plt.xlim((100,20000))
    plt.ylim((0,3000))
    plt.grid();

def show_words_repartition():
    #Répartition des mots
    df_words = pp.get_words_from_xml_files()

    df_words_count = pd.DataFrame(Counter(df_words['words']).keys(), columns=['words'])
    df_words_count['count'] = Counter(df_words['words']).values()
    df_words_count.sort_values(by='count', inplace=True, ascending=False)

    print('Nombre de mots différents : ', len(df_words['words'].unique()));
    print('Nombre de mots vues une seule fois : ', len(df_words_count['count'][df_words_count['count'] == 1]));
    print('Nombre de mots vues 1000+ fois : ', len(df_words_count['count'][df_words_count['count'] >= 1000]));

    plt.plot(df_words_count['words'][df_words_count['count'] > 100], df_words_count['count'][df_words_count['count'] > 100])
    plt.xticks(np.arange(0, 100, 10), ['1', '10', '20', '30', '40', '50', '60', '70', '80', '90'])
    plt.title('Répartition des mots vus 100+ fois');


def show_letters_per_word():
    df_words = pp.get_words_from_xml_files()
    
    # Nombre de lettres par mot
    sns.displot(df_words['len'], kind = 'hist')
    plt.title('Nombre de lettres par mot');

    print('Nombre total de lettres : ', df_words['len'].sum())
    print('Nombre moyen de lettres par mot: ', df_words['len'].mean())

def show_caracters_distribution_per_writer(df_all):
    letter_writers_dict = {}
    for index, row in df_all.iterrows():
        writer_id = str(row['writer_id'])
        for letter in row['transcription']:
            if letter != ' ':
                if letter not in letter_writers_dict:
                    letter_writers_dict[letter] = [writer_id]
                else:
                    if writer_id not in letter_writers_dict[letter]:
                        letter_writers_dict[letter].append(writer_id)

    keys = list(letter_writers_dict.keys())
    sorted_indexes = np.argsort(keys)
    values = [len(letter_writers_dict[key]) for key in keys]
    labels = []
    counts = []
    for index in sorted_indexes:
        labels.append(keys[index])
        counts.append(values[index])

    plt.figure(figsize=(20,10))
    plt.title("Distribution des caractères par nombre de rédacteurs")
    plt.bar(x=labels, height=counts)
    plt.xlabel('Caractère')
    plt.ylabel("Nombre de de rédacteurs")
    plt.show()



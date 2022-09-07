

def init():
    global DATA_PATH
    DATA_PATH = "../data/"

    global WORDS_META_FILENAME
    WORDS_META_FILENAME = DATA_PATH + "ascii/words.txt"

    global FORMS_META_FILENAME
    FORMS_META_FILENAME = DATA_PATH + "ascii/forms.txt"

    global XML_FILES_PATH
    XML_FILES_PATH = DATA_PATH + "xml"

    global WORDS_IMG_PATH
    WORDS_IMG_PATH = DATA_PATH + 'words'   # BASE_IMG_PATH
    #FORMS_IMG_PATH = DATA_PATH + 'forms'

    global WORD_COLUMNS
    WORD_COLUMNS = ['word_id', 'seg_res', 'gray_level', 'x', 'y', 'w', 'h', 'tag', 'transcription']
    global FORMS_COLUMNS
    FORMS_COLUMNS = ['form_id', 'writer_id', 'number_of_sentences', 'word_seg', 'total_lines', 'correct_lines', 'total_words', 'correct_words']



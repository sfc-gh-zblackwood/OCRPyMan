import pandas as pd
import re
from collections import Counter

import letter_detection_utils as ld_util
import ressources as rss

WORDS = Counter(words(open('../ressources/english_words/big.txt').read()))


# Fonction de metrique des prédictions

def metric(y_test, predicted_transcriptions, nb_correction):

    lev = []
    cer = []
    acc = []
    nb_correction_arr = []

    for i in range(len(y_test)):
        lev.append(ld_util.levenshtein_distance(y_test[i], predicted_transcriptions[i]))
        cer.append(lev[i]/len(y_test[i]))
        if predicted_transcriptions[i] == y_test[i]:
            acc.append(1)
        else : acc.append(0)
        nb_correction_arr.append(nb_correction)

    metric_df = pd.DataFrame(list(zip(y_test, predicted_transcriptions, nb_correction_arr, lev, cer, acc)), columns =['test','prediction','nb_correction','levenshtein', 'cer', 'acc']) 

    return metric_df

# GROUPE de fonctions pour la correction orthographique

def words(text): return re.findall(r'\w+', text.lower())

def P(word, N=sum(WORDS.values())): return WORDS[word] / N

def edits1(word):
    "All edits that are one edit away from `word`."
    # QUESTION A INSTRUIRE SI ON DOIT LAISSER LA PONCTUATION
    letters    = ''.join(rss.charList)
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def known(words): return set(w for w in words if w in WORDS)

def candidates(word, nb_correction=2): 
    # ORIGINAL :
    #return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
    if len(word) == 1: return word
    elif len(word) ==2: return known([word]) or known(edits1(word)) or [word]
    elif nb_correction == 1 : return known([word]) or known(edits1(word)) or [word]
    elif nb_correction == 2 : return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
    else : return [word]

def correction(word, nb_correction):
    correction = max(candidates(word, nb_correction), key=P)
    if not word.islower() and not word.isupper(): correction = correction.capitalize()
    elif word.isupper() : correction = correction.upper()
    return correction

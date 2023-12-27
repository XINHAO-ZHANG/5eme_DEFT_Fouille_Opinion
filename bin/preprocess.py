#!/usr/bin/env python3

import nltk
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# installer les packages nltk nécessaires pour la préparation des données 
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text, lang):
    '''
    prétraiter le texte en appliquant les étapes suivantes:
    - convertir le texte en minuscules
    - supprimer les nombres
    - supprimer les apostrophes
    - supprimer les mots de longueur inférieure ou égale à 2
    - supprimer les stopwords et la ponctuation
    '''
    # assurer que le texte est de type str
    if not isinstance(text, str):
        return str(text)  # convertir en str

    # charger les stopwords selon la langue
    if lang == 'en':
        stop_words = set(stopwords.words('english'))
    elif lang == 'fr':
        stop_words = set(stopwords.words('french'))
    elif lang == 'it':
        stop_words = set(stopwords.words('italian'))

    # enlever les symboles ennuyeux et les nombres 
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r"['’‘]", ' ', text)  
    text = re.sub(r'\b\w{1,2}\b', '', text)  

    words = word_tokenize(text.lower())

    # enlever les stopwords et la ponctuation
    words = [word for word in words if word not in stop_words and word not in string.punctuation]

    cleaned_text = ' '.join(words)
    return cleaned_text

def preprocess_csv(csv_path, lang,output_csv_path):
    '''
    lecture du fichier csv contenant les textes et leurs labels
    et prétraitement des textes
    '''
    # lire le fichier csv sauvegardé précédemment
    df = pd.read_csv(csv_path)

    # appliquer la fonction de prétraitement sur la colonne texte
    df['texte'] = df['texte'].apply(lambda x: preprocess_text(x, lang))

    df.to_csv(output_csv_path, index=False, encoding='utf-8')
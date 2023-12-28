#!/usr/bin/env python3

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def tfidfVectorize(CSV_train, CSV_test):
    '''
    vectoriser les textes en utilisant la méthode TF-IDF
    '''
    train_df = pd.read_csv(CSV_train)
    test_df = pd.read_csv(CSV_test)

    # initialiser le vectorizer avec les paramètres désirés
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.5,
        use_idf=True,
        sublinear_tf=True
        max_features=10000
    )

    # vectoriser les textes d'entraînement et de test 
    print("Vectorizing train data...")
    X_train = vectorizer.fit_transform(tqdm(train_df['texte']))

    print("Vectorizing test data...")
    X_test = vectorizer.transform(tqdm(test_df['texte']))

    # récupérer les labels correspondants
    y_train = train_df['parti'].astype(str)
    y_test = test_df['parti'].astype(str)

    return X_train, X_test, y_train, y_test

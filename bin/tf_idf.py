#!/usr/bin/env python3

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidfVectorize(CSV_train, CSV_test):
    '''
    vectoriser les textes en utilisant la méthode TF-IDF
    '''
    train_df = pd.read_csv(CSV_train)
    test_df = pd.read_csv(CSV_test)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.5,
        use_idf=True,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df['texte'])
    X_test = vectorizer.transform(test_df['texte'])

    # récupérer les labels correspondants
    y_train = train_df['label']
    y_test = test_df['label']

    return X_train, X_test, y_train, y_test

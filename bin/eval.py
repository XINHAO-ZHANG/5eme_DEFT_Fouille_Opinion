#!/usr/bin/env python3
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tf_idf import *

def evaluate(X_train, X_test, y_train, y_test, model):
    '''
    Évaluer les performances du modèle choisi
    en calculant l'accuracy, le kappa et la matrice de confusion
    '''
    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédire les labels sur les données de test
    y_pred = model.predict(X_test)

    return y_test, y_pred

def print_scores(y_test, y_pred):
    '''
    Afficher les scores d'évaluation du modèle choisi
    '''
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Kappa Score:", kappa)
    print("Classification Report:\n", report)

def plot_confusion_matrix(y_test,y_pred, model):
    '''
    Sauvagarder la visualisation de la matrice de confusion du modèle choisi
    '''
    # obtenir les labels prédits par le modèle
    unique_labels = sorted(set(y_test) | set(y_pred))

    # générer la matrice de confusion en spécifiant les labels
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # utiliser seaborn pour afficher la matrice de confusion sous forme de heatmap
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greys', 
                xticklabels=unique_labels, yticklabels=unique_labels)

    plt.xlabel('Parti Prédit')
    plt.ylabel('Parti Réel')
    plt.title('Matrice de Confusion')
    plt.show()

    # sauvegarder la figure dans le dossier result
    plt.savefig(f'../result/Confusion_Matrix_{model}.png', format='png', dpi=300)
# Détermination du Parti Politique des Parlementaires Européens 🇪🇺

Ce projet de recherche implémente une classification de texte pour identifier l'affiliation politique des parlementaires européens à partir de leurs discours, en se focalisant sur le corpus en français. Différents classificateurs ont été appliqués et évalués pour déterminer le plus performant.



## 📚 Traitement du corpus

Le corpus comprend des discours du Parlement européen en français. Les données ont été extraites de fichiers XML, normalisées, et traitées pour la classification. Le prétraitement comprend la normalisation de la casse, la suppression des mots vides, des caractères spéciaux et des chiffres.





## 💡 Feature Engineering

#### TF-IDF (Term Frequency-Inverse Document Frequency)

- **Processus :** Utilisation de TF-IDF pour convertir les discours en vecteurs, en tenant compte de la fréquence des mots tout en équilibrant leur importance à travers différents documents.
- **Configuration :** Paramètre `max_df` réglé à 0.6 pour filtrer les mots qui apparaissent fréquemment, améliorant ainsi la distinction entre les catégories.

### 

## 🚀 Méthodologies

Plusieurs modèles de classification ont été explorés :

### 1. Classificateurs Linéaires

#### Régression Logistique

- **Application :** Testé avec et sans rééquilibrage des classes. Amélioration notable de la performance avec rééquilibrage.

#### Multinomial Naive Bayes

- **Particularités :** Paramètre d'apprentissage ajusté pour gérer les déséquilibres de catégories.

### 2. Méthodes Ensemblistes

#### Random Forest

#### Gradient Boosting Decision Tree (GBDT)

- **XGBoost**
- **LightGBM**



## 📊Résultats et évaluation

Des expérimentations ont été menées pour chaque modèle, et les performances ont été évaluées en termes de précision, rappel, score F1, et support. Le projet a mis en lumière des différences significatives dans les performances des divers classificateurs.



## 👥 Auteurs

- Xinhao Zhang
- Yingzi Liu
- Xiaohua Cui

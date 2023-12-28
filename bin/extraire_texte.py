#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import pandas as pd


def extract_data_brut(file_path,output_csv_path):
    '''
    extraire les textes et leurs parities à partir des corpus d'apprentissage bruts
    et les conserver dans un fichier csv pour chaque langue
    '''
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for doc in root.findall('.//doc'):
        # obtenir tous les paragraphes du texte et les joindre en un seul
        text_elements = doc.findall('.//texte//p')
        texte = ' '.join([elem.text.strip() for elem in text_elements if elem.text])
        partis = [parti.get('valeur') for parti in doc.findall('.//PARTI')]
        for parti in partis:
            data.append({'texte': texte, 'parti': parti})

    df = pd.DataFrame(data)

    df.to_csv(output_csv_path, index=False, encoding='utf-8')

def extract_and_combine_text_label(xml_file_path, label_file_path, output_csv_path):
    '''
    extraire les textes tests dans les fichiers xml de et leurs labels des fichiers textes 
    et les combiner dans un seul fichier csv pour chaque langue
    '''   
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    test_data = []
    for doc in root.findall('.//doc'):
        # obtenir tous les paragraphes du texte et les joindre en un seul
        text_elements = doc.findall('.//texte//p')
        texte = ' '.join([elem.text.strip() for elem in text_elements if elem.text])
        test_data.append({'texte': texte})

    texte_df = pd.DataFrame(test_data)

    # lecture des labels à partir du fichier texte et les combiner avec les textes dans un même dataframe
    labels_df = pd.read_csv(label_file_path, sep='\t', header=None, names=['id', 'label'])
    labels_df.set_index('id', inplace=True)

    # ajouter une colonne id pour pouvoir joindre les deux dataframes
    texte_df['id'] = range(1, len(texte_df) + 1)
    texte_df.set_index('id', inplace=True)

    # joindre les deux dataframes en utilisant l'index id comme clé de jointure 
    combined_df = texte_df.join(labels_df, how='left')
    
    combined_df.to_csv(output_csv_path, index=False)



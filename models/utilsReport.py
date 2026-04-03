#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:51:24 2025

@author: kishan
"""


import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import  Draw, AllChem, DataStructs
import umap
import matplotlib.pyplot as plt
from IPython.display import display

# Function to convert SMILES to fingerprints
def smiles_to_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits=2048) for mol in mols]
    return fingerprints

# Function to calculate pairwise Tanimoto similarities
def calculate_pairwise_tanimoto(fingerprints):
    num_fps = len(fingerprints)
    similarities = []
    for i in range(num_fps):
        for j in range(i+1, num_fps):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(similarity)
    return similarities
    
    
def plot_and_save_combined_tanimoto_histogram(similarities1, similarities2, filename, dpi=500, font_size=12):
    plt.figure(figsize=(5, 5))
    plt.hist(similarities1, bins=50, alpha=0.5, color='black', label='Training', density=True)
    plt.hist(similarities2, bins=50, alpha=0.5, color='green', label='Generated', density=True)
    plt.xlabel('Tanimoto Similarity', fontsize=16)
    plt.ylabel('Normalized Frequency', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(filename, dpi=dpi, format='jpg', bbox_inches='tight')
    plt.show()    
    
def plot_smiles(smiles_list, num_top=1,num_bottom=1):
    # Create RDKit molecule objects
    #smiles_list = novel_generated_smiles['can_smiles']
    mol_top = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[:num_top]]
    mol_bottom = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[-num_bottom:]]
    
    # Draw the molecules
    img1 = Draw.MolsToGridImage(mol_top, molsPerRow=5, subImgSize=(500,500))
    img2 = Draw.MolsToGridImage(mol_bottom, molsPerRow=4, subImgSize=(500,500))
    display(img1, img2)
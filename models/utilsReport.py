#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:51:24 2025

@author: kishan
"""


import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs, rdFingerprintGenerator
import matplotlib.pyplot as plt
from IPython.display import display


_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Function to convert SMILES to fingerprints
def smiles_to_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    fingerprints = [_morgan_generator.GetFingerprint(mol) for mol in mols if mol is not None]
    return fingerprints

# Function to calculate pairwise Tanimoto similarities
def calculate_pairwise_tanimoto(fingerprints, max_pairs=200000, random_seed=0):
    num_fps = len(fingerprints)
    if num_fps < 2:
        return []

    total_pairs = num_fps * (num_fps - 1) // 2
    if max_pairs is None or total_pairs <= max_pairs:
        similarities = []
        for i in range(num_fps - 1):
            similarities.extend(
                DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
            )
        return similarities

    # A histogram does not require every O(n^2) pair. Sample ordered pairs
    # uniformly without constructing the full pair-index matrix.
    rng = np.random.default_rng(random_seed)
    left = rng.integers(0, num_fps, size=int(max_pairs))
    right = rng.integers(0, num_fps - 1, size=int(max_pairs))
    right += right >= left
    return [
        DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        for i, j in zip(left, right)
    ]
    
    
def plot_and_save_combined_tanimoto_histogram(
    similarities1,
    similarities2,
    filename,
    dpi=500,
    font_size=12,
    show=False,
):
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.hist(similarities1, bins=50, alpha=0.5, color='black', label='Training', density=True)
    axis.hist(similarities2, bins=50, alpha=0.5, color='green', label='Generated', density=True)
    axis.set_xlabel('Tanimoto Similarity', fontsize=16)
    axis.set_ylabel('Normalized Frequency', fontsize=16)
    axis.legend(fontsize=16)
    axis.tick_params(axis='both', labelsize=16)
    figure.savefig(filename, dpi=dpi, format='jpg', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(figure)
    
def plot_smiles(smiles_list, num_top=1,num_bottom=1):
    # Create RDKit molecule objects
    #smiles_list = novel_generated_smiles['can_smiles']
    mol_top = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[:num_top]]
    mol_bottom = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[-num_bottom:]]
    
    # Draw the molecules
    img1 = Draw.MolsToGridImage(mol_top, molsPerRow=5, subImgSize=(500,500))
    img2 = Draw.MolsToGridImage(mol_bottom, molsPerRow=4, subImgSize=(500,500))
    display(img1, img2)

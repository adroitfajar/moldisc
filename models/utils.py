#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:39:39 2025

@author: kishan
"""


import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from rdkit import Chem, RDLogger
import logging  # alternative?



#from rdkit import RDLogger 
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# This avoids printing of exception error when calling Chem.MolFromSmiles
RDLogger.DisableLog('rdApp.*')

def remove_residual(smiles):
    smiles=Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    return smiles

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        #print(e)
        return False

# Create a custom Dataset #  Adroit's code
class SmilesDataset(Dataset):
    def __init__(self, encoded_inputs):
        self.input_ids = encoded_inputs['input_ids']
        self.attention_mask = encoded_inputs['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx]
        }
    
# validate SMILES   #  Adroit's code    
def validate_smiles(smiles):
    """ Validate SMILES string using RDKit """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Chem.SanitizeMol(mol)
            return smiles
        else:
            return False
    except Exception as e:
        logger.debug(f"Invalid SMILES: {smiles}, Error: {e}")
    return False


    
# Augment SMILES   #  Adroit's code
def augment_smiles(smiles, num_augmentations=5):
    """ Generate different representations of a SMILES string """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]
    
    augmented_smiles = set()
    augmented_smiles.add(Chem.MolToSmiles(mol, canonical=True))
    
    for _ in range(num_augmentations):
        augmented_smiles.add(Chem.MolToSmiles(mol, canonical=False))
    
    return list(augmented_smiles)




'''
# Function to plot and save combined normalized Tanimoto similarity histogram
def plot_and_save_combined_tanimoto_histogram(similarities1, similarities2, filename, dpi=500):
    plt.figure(figsize=(5, 5))
    plt.hist(similarities1, bins=50, alpha=0.5, color='black', label='Training ILs', density=True)
    plt.hist(similarities2, bins=50, alpha=0.5, color='green', label='Generated ILs', density=True)
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.savefig(filename, dpi=dpi, format='jpg', bbox_inches='tight')
    plt.show()
'''
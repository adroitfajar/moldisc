#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:47:03 2026

@author: kishan
"""

import numpy as np
import time
from moldisc import *


project_name = "data_1"

# Defining the hyperparameters bounds for SMILESX 
embed_bounds = [8]  #[8, 16, 32, 64, 128, 256, 512] # embedding size
lstm_bounds = [8]  #[8, 16, 32, 64, 128, 256, 512] # number of units in the LSTM layer
tdense_bounds = [8]  #[8, 16, 32, 64, 128, 256, 512] # number of units in the dense layer
bs_bounds = [8]  #[8, 16, 32, 64, 128, 256, 512] # batch size
lr_bounds = [2.]  #[2., 2.5, 3., 3.5, 4.0, 4.5] # learning rate



moldisc(project_name = "data_1",
         input_file_labeled = 'labeled.csv',
         input_file_unlabeled = 'unlabeled.csv',
         project_folder = 'projects', 
          data_err=None, # Error on the property
          data_name='test',
          data_units='units',
          data_label='Test label',
          smiles_concat=True,
          #outdir =  '/outputs/' + str(0) +"/",
          geomopt_mode='off', # Zero-cost geometry optimization
          bayopt_mode='off', # Bayesian optimization
          train_mode='on', # Train
          model_type = 'regression', # 'regression' or 'classification'
          scale_output = True,
          bs_bounds=[64],
          lr_bounds=[2.0],
          embed_bounds=[64],
          lstm_bounds=[64],
          tdense_bounds=[64],
          k_fold_number=2, # Number of cross-validation splits
          n_runs=5 ,# Number of runs per fold
          check_smiles=True, # Verify SMILES validity via RDKit
          augmentation=True, # Augment the data or not
          bayopt_n_rounds=5,
          bayopt_n_epochs=25,
          bayopt_n_runs=25,
          n_gpus=1,
          n_epochs=3,
          log_verbose=True, # To send print outs both to the file and console
          train_verbose=True,
          gpt_pretrained_model='gpt2', 
          gpt_augmentation=2,
          gpt_data_split=0.8,
          #gpt_output_dir ="./output/",
          gpt_tr_epochs=100,
          gpt_tr_batch_size=2,
          gpt_eval_batch_size=2,
          gpt_warmup_steps=10,
          gpt_decay =0.01,
          #gpt_log_dir ="./log/",
          gpt_patience= 25,
          gpt_device = 2,
          gpt_num_generation=100,
          gpt_remove_ionic = 'all',
          gpt_num_attempts =5,
          max_generation=100,
          cycles = 1,
          sa_score=5,
          cutoff=0.5,
          option_save=True,
          option_show=True,
          num_mol_top=5,
          num_mol_bottom=5,
          cuda=0,
          patience=100,
          remove_tmp=True)

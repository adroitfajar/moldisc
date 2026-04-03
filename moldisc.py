#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:56:10 2026

@author: kishan
"""

import os
import numpy as np
import pandas as pd
import pickle  
from SMILESX import main
from SMILESX import loadmodel
from SMILESX import inference
import time
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import subprocess
import json
from models.utilsReport import calculate_pairwise_tanimoto, smiles_to_fingerprints, plot_and_save_combined_tanimoto_histogram, plot_smiles
from  sascore import SAscore
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display
from PIL import Image




def moldisc(project_name = "data_1",
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
          gpt_num_generation=10000,
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
          remove_tmp=True):


    
    
    project_dir =  project_folder + "/" +  project_name # pass this to SMILES-X 
    smilesX_output_dir = project_dir + "/" + "SMILESX/"
    gpt_output_dir = project_dir + "/" + "GPT/"
    gpt_log_dir = project_dir + "/" + "GPT/logs"
    tmp_output_dir = project_dir + "/" + "tmp/"
    final_output_dir = project_dir + "/" + "final/"
    data_dir =  "data/" +  project_name 
    
    try:
        os.makedirs(project_dir) 
        os.makedirs(tmp_output_dir) 
        os.makedirs(gpt_output_dir) 
        os.makedirs(gpt_log_dir) 
        os.makedirs(final_output_dir)
        
    except:
        print("Error in creating a project folder.")
        exit(-1)
        
        
    ## Loading data   
    try:
        input_file_labeled = pd.read_csv(data_dir  + "/"  +  input_file_labeled)
        print(input_file_labeled['smiles'].values.tolist())
    except pd.errors.EmptyDataError:
        print("Error in reading input file with SMILES and labels.")
        exit(-1)
    
    try:
        input_file_unlabeled = pd.read_csv(data_dir + "/"  +  input_file_unlabeled)
        print(input_file_unlabeled['smiles'].values.tolist())
    except pd.errors.EmptyDataError:
        print("Error in reading input file with SMILES without labels.")
        exit(-1)    
      
    
    
    
    # Storing the inout smiles for reporting ....
    input_file_labeled_original = input_file_labeled
    input_file_unlabeled_original = input_file_unlabeled
    
    
    
    # Train SMILESX onces with labeled data
    main.main(data_smiles=input_file_labeled[['smiles']], # SMILES input
              data_prop=input_file_labeled[['property']], # Property of interest
              data_err=None, # Error on the property
              data_name= data_name,
              data_units= data_units,
              data_label= data_label,
              smiles_concat=True,
              outdir = smilesX_output_dir + '/outputs/' + str(0) +"/",
              geomopt_mode= geomopt_mode, # Zero-cost geometry optimization
              bayopt_mode= bayopt_mode, # Bayesian optimization
              train_mode= train_mode, # Train
              model_type =  model_type, # or 'classification'
              scale_output =  scale_output,
              bs_bounds= bs_bounds,
              lr_bounds= lr_bounds,
              embed_bounds= embed_bounds,
              lstm_bounds= lstm_bounds,
              tdense_bounds= tdense_bounds,
              k_fold_number= k_fold_number, # Number of cross-validation splits
              n_runs= n_runs,# Number of runs per fold
              check_smiles=True, # Verify SMILES validity via RDKit
              augmentation=True, # Augment the data or not
              bayopt_n_rounds= bayopt_n_rounds,
              bayopt_n_epochs= bayopt_n_epochs,
              bayopt_n_runs= bayopt_n_runs,
              n_gpus= n_gpus,
              n_epochs= n_epochs,
              log_verbose=True, # To send print outs both to the file and console
              train_verbose=True)
    
    model_SMILESX = loadmodel.LoadModel(data_name= data_name,
                            outdir = smilesX_output_dir + '/outputs/' + str(0) +"/",
                            augment=True,
                            gpu_ind=0,
                            return_attention=True)
    
    sum_molecular_generations = 0
    extracted_smiles = []
    ##for iter in range( cycles):
    iter = 0
    while True:
        # First generation
        if iter == 0: 
           
            all_SMILES = input_file_labeled['smiles'].values.tolist() + input_file_unlabeled['smiles'].values.tolist()
            print(len(all_SMILES))
            
            ## Subprocess calling for GPT2  ####################################################################
            ##
            tmp_input_file =  tmp_output_dir + "/smiles_in_" + str(iter) + '.pkl'
            
            with open(tmp_input_file, 'wb') as file:
                pickle.dump(all_SMILES, file)
                file.close()
            
            gpt_args = [ str(iter),
                         gpt_pretrained_model, 
                         str( gpt_augmentation),
                         str( gpt_data_split),
                         gpt_output_dir,
                         str( gpt_tr_epochs),
                         str( gpt_tr_batch_size),
                         str( gpt_eval_batch_size),
                         str( gpt_warmup_steps),
                         str( gpt_decay),
                         gpt_log_dir,
                         str( gpt_patience),
                         str( gpt_device),
                         str( gpt_num_generation),
                         gpt_remove_ionic,
                         str( gpt_num_attempts),
                         str( project_folder),
                         str( project_name)
                        ]
            
           
            try:
                with open(tmp_output_dir + "args" + str(iter)  , 'wb') as argfile:
                    pickle.dump(gpt_args, argfile)
                    argfile.close()
            except:
                print("Error in writing argument to GPT subprocess...")
            
            
            # Subprocess to run the GPT2
            subprocess.run(['conda', 'run', '-n', 'subGPT2', 'python', 'mainGPT.py', '--iter' , str(iter) , '--args_project_folder' ,  project_folder , '--args_project_name' ,   project_name])
            
            print("Back to main....")
            
            
            #Remove tmeporary files
            #if remove_tmp:
            #    os.remove(tmp_input_file)
            
           
            #new_smiles = generator.generate_smiles( num_generation)
            new_smiles = []
            tmp_input_file =  tmp_output_dir + "/smiles_new_" + str(iter) + '.pkl'
            with open(tmp_input_file, 'rb') as file:
                new_smiles = pickle.load( file)
                file.close()
             
            
            #Remove tmeporary files
            #if remove_tmp:
            #    os.remove(tmp_input_file)
             
                
            # Computing SA scores on all generated molecules. If the the average is > 5 exit the loop
            list_sa_scores = []
            list_new_smies = list(new_smiles)
            len_gen_mols = len(list_new_smies)
            if len_gen_mols >0:
                for sm in range(len_gen_mols):
                    list_sa_scores.append(SAscore(list_new_smies[sm]))
                if sum(list_sa_scores) / len(list_sa_scores)> 5:
                    print("Average SA score of generated molecues is above 5. Exiting the generation at cycle: " + str(iter+1))
                    break
                
            preds_ = inference.infer(model=model_SMILESX,
                            data_smiles=new_smiles,
                            augment=True,
                            check_smiles=True,
                            log_verbose=True)
            
            ##  
            ####################################################################################################
    
            if  model_type == 'regression':     
                new_smiles_tabular = {'smiles': list(new_smiles),  'property': list(preds_['mean']) }
                df = pd.DataFrame(new_smiles_tabular)
        
                # Sort by ascending order 
                sorted_new_smiles_tabular = df.sort_values(by='property' , ascending=[False])
                
                #max_value = sorted_new_smiles_tabular.iloc[0]
                len_predcited_smiles = len(sorted_new_smiles_tabular)
                
                selelcted_len_predcited_smiles = int(np.ceil(len_predcited_smiles* cutoff))
                sum_molecular_generations = selelcted_len_predcited_smiles
                extracted_smiles = sorted_new_smiles_tabular.iloc[0:selelcted_len_predcited_smiles]
            else:
                '''
                new_smiles_tabular = {'smiles': list(new_smiles),  'property': list((preds_['mean'] > 0.5).astype("int8")) }
                print(new_smiles_tabular)
                df = pd.DataFrame(new_smiles_tabular)
                df[df.property > 0.5]
                extracted_smiles = new_smiles_tabular[new_smiles_tabular['property'] > 0]
                '''
                new_smiles_tabular = {'smiles': list(new_smiles),  'property': list(preds_['mean']) }
                df = pd.DataFrame(new_smiles_tabular)
                extracted_smiles = df[df.property > 0.5]
                print(extracted_smiles)
            
            # Termination conditions
            
            
            new_dataframe = pd.DataFrame({'smiles': extracted_smiles['smiles'].tolist(), 'property': extracted_smiles['property'].tolist() })
            input_file_labeled = pd.concat([input_file_labeled, new_dataframe], ignore_index=True)    
            
            tmp_input_file =  tmp_output_dir + "/smiles_new_dump" + str(iter) + '.csv'
            new_dataframe.to_csv(tmp_input_file)
            
            tmp_input_file =  tmp_output_dir + "/smiles_new_dump_conc" + str(iter) + '.csv'
            input_file_labeled.to_csv(tmp_input_file)
            
            all_generated_smiles = new_dataframe
      
        else:
            all_SMILES = input_file_labeled['smiles'].values.tolist() + input_file_unlabeled['smiles'].values.tolist()
    
            ## Subprocess calling for GPT2  ####################################################################
            ##
            tmp_input_file =  tmp_output_dir + "/smiles_in_" + str(iter) + '.pkl'
            
            with open(tmp_input_file, 'wb') as file:
                pickle.dump(all_SMILES, file)
                file.close()
                
            gpt_args = [ str(iter),
                         gpt_pretrained_model, 
                         str( gpt_augmentation),
                         str( gpt_data_split),
                         gpt_output_dir,
                         str( gpt_tr_epochs),
                         str( gpt_tr_batch_size),
                         str( gpt_eval_batch_size),
                         str( gpt_warmup_steps),
                         str( gpt_decay),
                         gpt_log_dir,
                         str( gpt_patience),
                         str( gpt_device),
                         str( gpt_num_generation),
                         gpt_remove_ionic,
                         str( gpt_num_attempts),
                         str( project_folder),
                         str( project_name)
                        ]
            
           
            try:
                with open(tmp_output_dir + "args" + str(iter)  , 'wb') as argfile:
                    pickle.dump(gpt_args, argfile)
                    argfile.close()
            except:
                print("Error in writing argument to GPT subprocess...")
            
    
            subprocess.run(['conda', 'run', '-n', 'subGPT2', 'python', 'mainGPT.py', '--iter' , str(iter) , '--args_project_folder' ,  project_folder , '--args_project_name' ,   project_name])
            
            #subprocess.run(['conda', 'run', '-n', 'subGPT2', 'python', 'mainGPT_.py', '--iter' , str(iter)  , '--args_project_name' ,   project_name]) #, '--input_folder' , project_dir
    
            
            print("Back to main....")
            
            
            #Remove tmeporary files
            #if remove_tmp:
            #    os.remove(tmp_input_file)
           
            #new_smiles = generator.generate_smiles( num_generation)
            new_smiles = []
            tmp_input_file =  tmp_output_dir + "/smiles_new_" + str(iter) + '.pkl'
            with open(tmp_input_file, 'rb') as file:
                new_smiles = pickle.load( file)
                file.close()
             
            #Remove tmeporary files
            #if remove_tmp:
            #    os.remove(tmp_input_file)
                
            # Computing SA scores on all generated molecules. If the the average is > 5 exit the loop
            list_sa_scores = []
            list_new_smies = list(new_smiles)
            len_gen_mols = len(list_new_smies)
            if len_gen_mols >0:
                for sm in range(len_gen_mols):
                    list_sa_scores.append(SAscore(list_new_smies[sm]))
                if sum(list_sa_scores) / len(list_sa_scores)> 5:
                    print("Average SA score of generated molecues is above 5. Exiting the generation at cycle: " + str(iter+1))
                    break
            
            preds_ = inference.infer(model=model_SMILESX,
                            data_smiles=new_smiles,
                            augment=True,
                            check_smiles=True,
                            log_verbose=True)
            
            
            tf.keras.backend.clear_session() 
            ##  
            ####################################################################################################
    
    
            if  model_type == 'regression':     
                new_smiles_tabular = {'smiles': list(new_smiles),  'property': list(preds_['mean']) }
                df = pd.DataFrame(new_smiles_tabular)
        
                # Sort by ascending order 
                sorted_new_smiles_tabular = df.sort_values(by='property' , ascending=[False])
                
                #max_value = sorted_new_smiles_tabular.iloc[0]
                len_predcited_smiles = len(sorted_new_smiles_tabular)
                
                selelcted_len_predcited_smiles = int(np.ceil(len_predcited_smiles* cutoff))
                sum_molecular_generations = selelcted_len_predcited_smiles
                extracted_smiles = sorted_new_smiles_tabular.iloc[0:selelcted_len_predcited_smiles]
            else:
                '''
                new_smiles_tabular = {'smiles': list(new_smiles),  'property': list((preds_['mean'] > 0.5).astype("int8")) }
                #new_smiles_tabular = {'smiles': list(new_smiles),  'property': list(preds_['mean']) }
                df = pd.DataFrame(new_smiles_tabular)
                extracted_smiles = new_smiles_tabular[new_smiles_tabular['property'] > 0]
                '''
                new_smiles_tabular = {'smiles': list(new_smiles),  'property': list(preds_['mean']) }
                df = pd.DataFrame(new_smiles_tabular)
                extracted_smiles = df[df.property > 0.5]
                print(extracted_smiles)
            
            
            
            new_dataframe = pd.DataFrame({'smiles': extracted_smiles['smiles'].tolist(), 'property': extracted_smiles['property'].tolist() })
             
            all_generated_smiles = pd.concat([all_generated_smiles, new_dataframe], ignore_index=True)
            input_file_labeled = pd.concat([input_file_labeled, new_dataframe], ignore_index=True) 
            
            
            tmp_input_file =  tmp_output_dir + "/smiles_new_dump" + str(iter) + '.csv'
            new_dataframe.to_csv(tmp_input_file)
            
            tmp_input_file =  tmp_output_dir + "/smiles_new_dump_conc" + str(iter) + '.csv'
            input_file_labeled.to_csv(tmp_input_file)
            
        
        
        
        #Termination condition
        if  max_generation != -1:
            if sum_molecular_generations >  max_generation:
                print("Maximum number of molecular generations have been reached.")
                break
             
        if  cycles != -1:
            if  cycles == iter + 1:
                print("Maximum molecular generation cycles have been reached.")
                break       
           
        iter = iter +1
        #computing 
    
    
    ## save generated smiles
    ## check new_dataframe not null or empty
    gen_smiles_csv = final_output_dir + "/generated_smiles.csv"
    all_generated_smiles = all_generated_smiles.sort_values(by='property' , ascending=[False])
    all_generated_smiles.to_csv(gen_smiles_csv) 
    
    
    ## TO DO: plotting of Molecues
    gen_smiles_list = all_generated_smiles['smiles'].tolist()
    len_gen_smiles_list = len(gen_smiles_list)
    
        
    
    ## PLotting and summarizing
    ## plotting molecues
    # Create RDKit molecule objects
    plot_top = np.min((len_gen_smiles_list, num_mol_top))
    plot_bottom = np.min((len_gen_smiles_list, num_mol_bottom))
    smiles_list = all_generated_smiles['smiles']
    #mol_top = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[:plot_top]]
    #mol_bottom = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[-plot_bottom:]]
    top_mol_img_file = final_output_dir + "/top_" + str(plot_top)  + "_molecules.png"
    bottom_mol_img_file = final_output_dir + "/bottom_" + str(plot_bottom)  + "_molecules.png"
    img_top = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles) for smiles in smiles_list[:plot_top]])
    img_bottom = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles) for smiles in smiles_list[-plot_bottom:]])
    img_top.save(top_mol_img_file) 
    img_bottom.save(bottom_mol_img_file) 

    
    # Draw the molecules
    #img1 = Draw.MolsToGridImage(mol_top, molsPerRow=5, subImgSize=(500,500))
    #img2 = Draw.MolsToGridImage(mol_bottom, molsPerRow=4, subImgSize=(500,500))
    #display(img1, img2)
        
    
    ## plotting tanimoteo
    try:
        tanimoto_jpg = final_output_dir + "/tanimoto.jpg"
        imput_fingerprints = smiles_to_fingerprints(input_file_labeled_original['smiles'])
        gen_fingerprints = smiles_to_fingerprints(new_dataframe['smiles'])
            
        input_similarities = calculate_pairwise_tanimoto(imput_fingerprints)
        gen_similarities = calculate_pairwise_tanimoto(gen_fingerprints)
        
        plot_and_save_combined_tanimoto_histogram(input_similarities, gen_similarities, tanimoto_jpg, 500)
    except:
        print("Error in computing Tanimoto histograms.")    
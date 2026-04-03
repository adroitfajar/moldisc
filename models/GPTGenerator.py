#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:39:39 2025

@author: kishan
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from .utils import augment_smiles, SmilesDataset, validate_smiles, remove_residual
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback
#import logger
import logging
import re
from transformers.utils import logging as logtr


logtr.get_logger("transformers").setLevel(logging.ERROR)



# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GPTGenerator():
    
    def __init__(self, gpt_pretrained_model='gpt2' , augmentation=0, data_split = 0.8, out_dir = 'results', tr_epochs = 50,tr_batch_size = 2,eval_batch_size =2, warmup_steps = 10, decay =  0.01, log_dir = 'logs' ,  patience =1 , device=0):
        super(GPTGenerator, self).__init__()

        self.augmentation = augmentation
        self.gpt_pretrained_model = gpt_pretrained_model
        self.data_split = data_split
        self.out_dir = out_dir
        self.tr_epochs = tr_epochs
        self.tr_batch_size = tr_batch_size
        self.eval_batch_size = eval_batch_size
        self.warmup_steps = warmup_steps
        self.decay = decay
        self.log_dir = log_dir
        self.patience = patience

        self.model = ''
        self.tokenizer = ''
        if torch.cuda.is_available():
            current_device_id = torch.cuda.current_device()
            self.device = torch.device("cuda:" + str(current_device_id))
            #self.device = torch.device("cuda:" + str(device))
        else:
            self.device = torch.device("cpu")   


    def save_models(self, save_model_path ='pretrained_models/', save_model_name ='pt0001' ):
        try:
            self.model.save_pretrained(save_model_path + save_model_name)
            print("Pretrained model " + save_model_path + ' saved at ' + save_model_name +'.')
        except:
            print("Saving pretrained model failed.")  

    def save_tokenizer(self,  save_tokenizer_path='tokenizer/',  save_tokenizer_name='tk0001' ):
        try:
            self.tokenizer.save_pretrained(save_tokenizer_path + save_tokenizer_name)
            print("Pretrained model " + save_tokenizer_path + ' saved at ' + save_tokenizer_name +'.')
        except:
            print("Saving pretrained model failed.")  
                  


    def build_dataset(self, input_smiles_data):
        smiles_list = input_smiles_data # input_smiles_data['smiles'].tolist()
        
        # Apply augmentation
        augmented_smiles_list = []
        if self.augmentation > 0:
            for smiles in smiles_list:
                augmented_smiles_list.extend(augment_smiles(smiles,self.augmentation))
        else:
            augmented_smiles_list= smiles_list
        
        
        print(self.gpt_pretrained_model)
        # Tokenization using GPT2 Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_pretrained_model) #.to(self.device)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        length_tokenizer = len(self.tokenizer)
        print("load pretrained....")
        
        # Encode the SMILES strings
        encoded_inputs = self.tokenizer(augmented_smiles_list, padding=True, truncation=True, return_tensors="pt")   
        
        dataset = SmilesDataset(encoded_inputs)

        # Split the data
        train_size = int(self.data_split * len(dataset))
        test_size = len(dataset) - train_size
        print(train_size)
        print(test_size)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset, length_tokenizer
        

    def train(self, train_dataset, test_dataset, length_tokenizer):
        
        #  Build and Train the Model
        self.model = GPT2LMHeadModel.from_pretrained(self.gpt_pretrained_model) #.to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        '''
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,  # Max number of epochs
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=2,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_first_step = True,
            logging_strategy = "epoch",
            evaluation_strategy="epoch",  # Evaluate every epoch
            save_strategy="epoch",
            load_best_model_at_end=True,  # Load the best model at the end of training
            
        )
        '''
        '''
        print(self.out_dir)
        print(self.log_dir)
        print(self.tr_batch_size)
        print(self.eval_batch_size)
        print(self.warmup_steps)
        print(self.decay)
        print(self.tr_epochs)
        
        training_args = TrainingArguments(
            output_dir= "./" + str(self.out_dir),
            num_train_epochs=10,  # Max number of epochs
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir= "./" + str(self.log_dir),
            logging_first_step = True,
            logging_strategy = "epoch",
            evaluation_strategy="epoch",  # Evaluate every epoch
            save_strategy="epoch",
            load_best_model_at_end=True,  # Load the best model at the end of training
        )
        '''
        
        training_args = TrainingArguments(
            output_dir= "./" + str(self.out_dir),
            num_train_epochs=int(self.tr_epochs),  # Max number of epochs
            per_device_train_batch_size=int(self.tr_batch_size),
            per_device_eval_batch_size=int(self.eval_batch_size),
            warmup_steps=int(self.warmup_steps),
            weight_decay=float(self.decay),
            logging_dir= "./" + str(self.log_dir),
            logging_first_step = True,
            logging_strategy = "epoch",
            evaluation_strategy="epoch",  # Evaluate every epoch
            save_strategy="epoch",
            load_best_model_at_end=True,  # Load the best model at the end of training
        )
        
        
        # Adding EarlyStoppingCallback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=self.patience)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            callbacks=[early_stopping],  # Add early stopping callback
        )
        
        trainer.train()        
        
    # remove_ionics - 'all' , '+'. '-'   
    def generate_smiles(self, target_num_samples, remove_ionics='all', num_attempts = 100):
        generated_smiles = set()
        attempts = 0
        
        
        while len(generated_smiles) < target_num_samples or attempts < target_num_samples * num_attempts:
            input_ids = torch.tensor(self.tokenizer.encode('[PAD]', add_special_tokens=False)).unsqueeze(0).to(self.device)
            attention_mask = torch.ones(input_ids.shape , device=self.device)  # Ensure attention mask is set
            try:
                sample_output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.9,  # Increase temperature for more diversity
                    do_sample=True
                )
                smiles = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
                print(smiles)
                print(len(smiles))

                smiles = remove_residual(smiles)
                valid_smiles = validate_smiles(smiles)
                
                #num_valid_smiles = len(valid_smiles)
                
                if valid_smiles and valid_smiles not in generated_smiles:
                    
                    if remove_ionics == 'all': # neutral
                        if (not re.search(r'[\.\-\+]', valid_smiles)):
                            generated_smiles.add(valid_smiles)
                    elif remove_ionics == '+':
                        if (not re.search(r'[\.\+]', valid_smiles)):
                            generated_smiles.add(valid_smiles)
                    elif remove_ionics == '-':
                        if (not re.search(r'[\.\-]', valid_smiles)):
                            generated_smiles.add(valid_smiles)
                    else:
                        generated_smiles.add(valid_smiles)
                        #if (not re.search(r'[.]', valid_smiles)):
                        #    generated_smiles.add(valid_smiles)
                    print(generated_smiles) 
                    
                    #generated_smiles.add(valid_smiles)
                    #print(generated_smiles) 
                    logger.info(f"Generated valid SMILES: {valid_smiles}")
                else:
                    logger.debug(f"Invalid or duplicate SMILES: {smiles}")
                
                '''
                for i in range(len(smiles)):
                    valid_smiles = validate_smiles(smiles[i])
                    if  valid_smiles: 
                        if remove_ionics == 'all':
                            filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.\-\+]')]
                        elif remove_ionics == '+':
                            filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.\+]')]
                        elif remove_ionics == '-':
                            filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.\-]')]
                        else:
                            filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.]')]
                            
                        if filtered_smiles and filtered_smiles not in generated_smiles:
                            generated_smiles.add(filtered_smiles)
                            logger.info(f"Generated valid SMILES: {valid_smiles}")
                        else:
                            logger.debug(f"Invalid or duplicate SMILES: {smiles}")
                '''
                
                '''
                valid_smiles = validate_smiles(smiles)
                print(valid_smiles)
                if  valid_smiles: 
                    valid_smiles['smiles'] = valid_smiles['smiles'].astype(str)
                    # Removing ionics and radicals
                    if remove_ionics == 'all':
                        filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.\-\+]')]
                    elif remove_ionics == '+':
                        filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.\+]')]
                    elif remove_ionics == '-':
                        filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.\-]')]
                    else:
                        filtered_smiles = valid_smiles[~valid_smiles['smiles'].str.contains(r'[.]')]
                        
                    if filtered_smiles and filtered_smiles not in generated_smiles:
                        generated_smiles.add(filtered_smiles)
                        logger.info(f"Generated valid SMILES: {valid_smiles}")
                    else:
                        logger.debug(f"Invalid or duplicate SMILES: {smiles}")
                '''    
            except Exception as e:
                logger.error(f"Error during SMILES generation: {e}")
            attempts += 1
        if len(generated_smiles) < target_num_samples:
            logger.warning(f"Only {len(generated_smiles)} valid SMILES were generated after {attempts} attempts.")
        return list(generated_smiles)
        
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 09:25:40 2025

@author: kishan
"""

import pickle
from models.GPTGenerator import GPTGenerator 
#import json
import argparse                                                                               
import shutil
from pathlib import Path
import os 


parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=0, help='Outer iteration from main.')
parser.add_argument('--args_project_folder', type=str, default='./project', help='Enter a unique project folder.')
parser.add_argument('--args_project_name', type=str, default='test_project', help='Enter a unique project name. New folder will be created inside the project folder.')
args_parser  = parser.parse_args()

print("inside the GPT2 main.....")
#args_file = "projects/" + args_parser.args_project_name + "/tmp/args" + str(args_parser.iter) + ".txt"
args_file = str(args_parser.args_project_folder) + "/" + args_parser.args_project_name + "/tmp/args" + str(args_parser.iter)
print(args_file)


args = []
try:
    with open(args_file, 'rb') as argfile:
        args = pickle.load(argfile)
        print(args)
        argfile.close()
except:
    print("Error in reading argument from main process...")

print("inside the GPT2 main.....")

out_iter = args[0]
gpt_pretrained_model = args[1]
gpt_augmentation= int(args[2])
gpt_data_split= float(args[3])
output_folder= args[4]
gpt_tr_epochs= int(args[5])
gpt_tr_batch_size= args[6]
gpt_eval_batch_size= int(args[7])
gpt_warmup_steps= int(args[8])
gpt_decay= float(args[9])
gpt_log_dir= args[10]
gpt_patience= int(args[11])
gpt_device= int(args[12])
gpt_num_generation= int(args[13])
gpt_remove_ionic= args[14]
gpt_num_attempts= int(args[15]) 
project_folder= str(args[16]) 
project_name= str(args[17]) 
#out_iter= int(args[18]) 

tmp_input_file = project_folder + "/" + project_name + '/tmp/smiles_in_' + out_iter + '.pkl'
out_dir = project_folder + "/" + project_name +  "/" + "GPT/"

all_SMILES = []
with open(tmp_input_file, 'rb') as file:
    all_SMILES = pickle.load(file)
    file.close()
    


generator = GPTGenerator( gpt_pretrained_model=gpt_pretrained_model, augmentation=gpt_augmentation, data_split=gpt_data_split, out_dir=output_folder, tr_epochs=gpt_tr_epochs, tr_batch_size=gpt_tr_batch_size, eval_batch_size=gpt_eval_batch_size, warmup_steps=gpt_warmup_steps, decay=gpt_decay, log_dir=gpt_log_dir ,  patience=gpt_patience , device=gpt_device)


train_dataset, test_dataset, length_tokenizer = generator.build_dataset(all_SMILES)


generator.train(train_dataset, test_dataset, length_tokenizer)


new_smiles = generator.generate_smiles( target_num_samples= gpt_num_generation, remove_ionics=gpt_remove_ionic, num_attempts = gpt_num_attempts)





tmp_output_file =  project_folder + "/" + project_name + '/tmp/smiles_new_' + out_iter + '.pkl'

tmp_output_file_ = project_folder + "/" + project_name + '/tmp/smiles_new_' + out_iter + '__.pkl'

#print(tmp_output_file)

with open(tmp_output_file, 'wb') as file1:
    pickle.dump(new_smiles, file1)
    file1.close()
    

# cleaning intermediary GPT files
parent_folder_path = project_folder + "/" + project_name + '/GPT/' 
p = Path(parent_folder_path)

if not p.is_dir():
   print(f"Error: '{parent_folder_path}' is not a valid directory.")
#return

for item in p.iterdir():
   if item.is_dir():
      try:
         shutil.rmtree(item)
         print(f"Removed directory: {item}")
      except OSError as e:
         print(f"Error removing directory {item}: {e}")


    
    
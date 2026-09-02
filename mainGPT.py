#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 09:25:40 2025

@author: authors
"""

import json
import pickle
import random
from models.GPTGenerator import GPTGenerator 
import argparse                                                                               
import shutil
from pathlib import Path
import sys

import numpy as np
import torch

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
gpt_generation_batch_size = int(args[18]) if len(args) > 18 else 128
gpt_max_new_tokens = int(args[19]) if len(args) > 19 else 100
random_seed = int(args[20]) if len(args) > 20 else 42
excluded_smiles_path = Path(args[21]) if len(args) > 21 and args[21] else None
gpt_min_carbon_atoms = int(args[22]) if len(args) > 22 else 1
gpt_min_heavy_atoms = int(args[23]) if len(args) > 23 else 2
gpt_max_heavy_atoms = (
    None if len(args) <= 24 or args[24] in (None, "", "None") else int(args[24])
)
gpt_max_molecular_weight = (
    None if len(args) <= 25 or args[25] in (None, "", "None") else float(args[25])
)
gpt_max_logp = (
    None if len(args) <= 26 or args[26] in (None, "", "None") else float(args[26])
)
gpt_allowed_elements = args[27] if len(args) > 27 else None
#out_iter= int(args[18]) 

cycle_seed = random_seed + int(out_iter)
random.seed(cycle_seed)
np.random.seed(cycle_seed)
torch.manual_seed(cycle_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cycle_seed)



tmp_input_file = project_folder + "/" + project_name + '/tmp/smiles_in_' + out_iter + '.pkl'
out_dir = project_folder + "/" + project_name +  "/" + "GPT/"

all_SMILES = []
excluded_smiles = set()

try:
    with open(tmp_input_file, 'rb') as file:
        all_SMILES = pickle.load(file)
        file.close()
except FileNotFoundError:
    print("The file smiles_in_" + out_iter + ".pkl was not found.")
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit()

if excluded_smiles_path is not None and excluded_smiles_path.exists():
    with open(excluded_smiles_path, 'rb') as file:
        excluded_smiles = set(pickle.load(file))

#with open(tmp_input_file, 'rb') as file:
#    all_SMILES = pickle.load(file)
#    file.close()
    


latest_model_path = Path(project_folder) / project_name / "GPT" / "latest_model"
model_source = gpt_pretrained_model
if args_parser.iter > 0 and (latest_model_path / "config.json").exists():
    model_source = str(latest_model_path)
    print(f"Continuing GPT fine-tuning from {latest_model_path}")

generator = GPTGenerator( gpt_pretrained_model=model_source, augmentation=gpt_augmentation, data_split=gpt_data_split, out_dir=output_folder, tr_epochs=gpt_tr_epochs, tr_batch_size=gpt_tr_batch_size, eval_batch_size=gpt_eval_batch_size, warmup_steps=gpt_warmup_steps, decay=gpt_decay, log_dir=gpt_log_dir ,  patience=gpt_patience , device=gpt_device, random_seed=cycle_seed)


train_dataset, test_dataset, length_tokenizer = generator.build_dataset(all_SMILES)


generator.train(train_dataset, test_dataset, length_tokenizer)


new_smiles = generator.generate_smiles(
    target_num_samples=gpt_num_generation,
    remove_ionics=gpt_remove_ionic,
    num_attempts=gpt_num_attempts,
    generation_batch_size=gpt_generation_batch_size,
    max_new_tokens=gpt_max_new_tokens,
    excluded_smiles=excluded_smiles,
    min_carbon_atoms=gpt_min_carbon_atoms,
    min_heavy_atoms=gpt_min_heavy_atoms,
    max_heavy_atoms=gpt_max_heavy_atoms,
    max_molecular_weight=gpt_max_molecular_weight,
    max_logp=gpt_max_logp,
    allowed_elements=gpt_allowed_elements,
)

latest_model_path.mkdir(parents=True, exist_ok=True)
generator.model.save_pretrained(str(latest_model_path))
generator.tokenizer.save_pretrained(str(latest_model_path))

generation_stats = dict(generator.last_generation_stats)
generation_stats.update({
    "cycle": int(out_iter),
    "random_seed": cycle_seed,
    "pretrained_model": str(model_source),
    "training_molecules": len(all_SMILES),
    "excluded_molecules": len(excluded_smiles),
    "training_metrics": generator.training_metrics,
    "requested_pretrained_model": gpt_pretrained_model,
    "resolved_model_revision": generator.model_revision,
    "resolved_tokenizer_revision": generator.tokenizer_revision,
})
stats_output_file = Path(project_folder) / project_name / 'tmp' / f'generation_stats_{out_iter}.json'
with open(stats_output_file, 'w', encoding='utf-8') as stats_file:
    json.dump(generation_stats, stats_file, indent=2, default=float)

model_provenance_path = Path(project_folder) / project_name / 'GPT' / 'model_provenance.json'
if model_provenance_path.exists():
    try:
        model_provenance = json.loads(model_provenance_path.read_text(encoding='utf-8'))
    except Exception:
        model_provenance = {"cycles": []}
else:
    model_provenance = {"cycles": []}
model_provenance.setdefault("cycles", []).append({
    "cycle": int(out_iter),
    "requested_pretrained_model": gpt_pretrained_model,
    "loaded_model_source": str(model_source),
    "resolved_model_revision": generator.model_revision,
    "resolved_tokenizer_revision": generator.tokenizer_revision,
})
model_provenance_path.write_text(
    json.dumps(model_provenance, indent=2, default=str), encoding='utf-8'
)





tmp_output_file =  project_folder + "/" + project_name + '/tmp/smiles_new_' + out_iter + '.pkl'

#tmp_output_file_ = project_folder + "/" + project_name + '/tmp/smiles_new_' + out_iter + '__.pkl'

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
   if item.is_dir() and item.name.startswith("checkpoint-"):
      try:
         shutil.rmtree(item)
         print(f"Removed directory: {item}")
      except OSError as e:
         print(f"Error removing directory {item}: {e}")


    
    

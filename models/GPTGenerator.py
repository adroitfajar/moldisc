#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:39:39 2025

@author: kishan
"""
import random
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors
from .utils import augment_smiles, SmilesDataset, canonicalize_smiles
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback
import logging
from transformers.utils import logging as logtr


logtr.get_logger("transformers").setLevel(logging.ERROR)



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustCheckpointTrainer(Trainer):
    """Trainer with a narrowly scoped retry for transient Windows renames."""

    @staticmethod
    def _checkpoint_paths(run_dir, global_step):
        destination = Path(run_dir) / f"checkpoint-{global_step}"
        source = Path(run_dir) / f"tmp-checkpoint-{global_step}"
        return source, destination

    @staticmethod
    def _retryable_windows_rename(exc, source: Path, destination: Path) -> bool:
        if os.name != "nt" or not isinstance(exc, OSError):
            return False
        if getattr(exc, "winerror", None) not in {5, 32, 33}:
            return False
        if source.parent.resolve() != destination.parent.resolve():
            return False
        return (
            source.name.startswith("tmp-checkpoint-")
            and destination.name.startswith("checkpoint-")
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        try:
            return super()._save_checkpoint(model, trial, metrics)
        except OSError as exc:
            run_dir = self._get_output_dir(trial=trial)
            source, destination = self._checkpoint_paths(run_dir, self.state.global_step)
            if not self._retryable_windows_rename(exc, source, destination):
                raise
            if destination.exists() and not source.exists():
                logger.warning("Checkpoint rename completed despite a transient Windows error.")
            else:
                if destination.exists() or not source.exists():
                    raise
                for attempt in range(5):
                    try:
                        time.sleep(0.2 * (2**attempt))
                        os.rename(source, destination)
                        logger.warning(
                            "Recovered transient Windows checkpoint rename after %s retry/retries.",
                            attempt + 1,
                        )
                        break
                    except OSError as retry_exc:
                        if not self._retryable_windows_rename(
                            retry_exc, source, destination
                        ) or attempt == 4:
                            raise
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
            self.args.distributed_state.wait_for_everyone()

class GPTGenerator():
    
    def __init__(self, gpt_pretrained_model='gpt2' , augmentation=0, data_split = 0.8, out_dir = 'results', tr_epochs = 50,tr_batch_size = 2,eval_batch_size =2, warmup_steps = 10, decay =  0.01, log_dir = 'logs' ,  patience =1 , device=0, random_seed=42):
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
        self.random_seed = int(random_seed)
        self.last_generation_stats = {}
        self.training_metrics = {}
        self.model_revision = None
        self.tokenizer_revision = None

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        rdBase.SeedRandomNumberGenerator(self.random_seed)

        self.model = ''
        self.tokenizer = ''
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            requested_device = int(device)
            if requested_device < 0 or requested_device >= device_count:
                logger.warning(
                    "Requested CUDA device %s is unavailable; using cuda:0 from %s visible device(s).",
                    requested_device,
                    device_count,
                )
                requested_device = 0
            torch.cuda.set_device(requested_device)
            self.device = torch.device("cuda", requested_device)
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
        smiles_list = list(input_smiles_data)
        if len(smiles_list) < 2:
            raise ValueError("At least two SMILES are required to create train and evaluation sets.")

        # Split molecules before augmentation so alternative representations of
        # the same molecule cannot leak into both train and evaluation sets.
        train_size = min(max(1, int(self.data_split * len(smiles_list))), len(smiles_list) - 1)
        test_size = len(smiles_list) - train_size
        split_generator = torch.Generator().manual_seed(self.random_seed)
        train_subset, test_subset = random_split(
            smiles_list,
            [train_size, test_size],
            generator=split_generator,
        )

        def expand_smiles(smiles_subset):
            expanded = []
            for smiles in smiles_subset:
                if self.augmentation > 0:
                    expanded.extend(augment_smiles(smiles, self.augmentation))
                else:
                    expanded.append(smiles)
            return expanded

        train_smiles = expand_smiles(train_subset)
        test_smiles = expand_smiles(test_subset)

        logger.info("Loading tokenizer from %s", self.gpt_pretrained_model)
        # Tokenization using GPT2 Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_pretrained_model)
        self.tokenizer_revision = getattr(self.tokenizer, "_commit_hash", None)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        length_tokenizer = len(self.tokenizer)

        # GPT-2 does not add sequence delimiters automatically. Training with an
        # explicit end token lets generation stop before the maximum length.
        bos_token = self.tokenizer.bos_token or self.tokenizer.eos_token or ""
        eos_token = self.tokenizer.eos_token or ""
        train_smiles = [f"{bos_token}{smiles}{eos_token}" for smiles in train_smiles]
        test_smiles = [f"{bos_token}{smiles}{eos_token}" for smiles in test_smiles]

        train_inputs = self.tokenizer(train_smiles, padding=True, truncation=True, return_tensors="pt")
        test_inputs = self.tokenizer(test_smiles, padding=True, truncation=True, return_tensors="pt")
        train_dataset = SmilesDataset(train_inputs)
        test_dataset = SmilesDataset(test_inputs)
        logger.info(
            "GPT dataset split: %s molecules (%s representations) train / "
            "%s molecules (%s representations) evaluation",
            train_size,
            len(train_dataset),
            test_size,
            len(test_dataset),
        )
        return train_dataset, test_dataset, length_tokenizer
        

    def train(self, train_dataset, test_dataset, length_tokenizer):
        
        #  Build and Train the Model
        self.model = GPT2LMHeadModel.from_pretrained(self.gpt_pretrained_model)
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        training_args = TrainingArguments(
            output_dir=str(self.out_dir),
            num_train_epochs=int(self.tr_epochs),  # Max number of epochs
            per_device_train_batch_size=int(self.tr_batch_size),
            per_device_eval_batch_size=int(self.eval_batch_size),
            warmup_steps=int(self.warmup_steps),
            weight_decay=float(self.decay),
            logging_dir=str(self.log_dir),
            logging_first_step = True,
            logging_strategy = "epoch",
            evaluation_strategy="epoch",  # Evaluate every epoch
            save_strategy="epoch",
            load_best_model_at_end=True,  # Load the best model at the end of training
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to=[],
            seed=self.random_seed,
            data_seed=self.random_seed,
        )
        
        
        # Adding EarlyStoppingCallback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=self.patience)
        
        trainer = RobustCheckpointTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            callbacks=[early_stopping],  # Add early stopping callback
        )
        
        train_result = trainer.train()
        self.training_metrics = dict(train_result.metrics)
        self.model = trainer.model
        self.model.to(self.device)
        self.model.eval()
        
    @staticmethod
    def _passes_ionic_filter(mol, remove_ionics):
        if remove_ionics not in {'all', '+', '-'}:
            return True
        if len(Chem.GetMolFrags(mol)) != 1:
            return False

        atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        if remove_ionics == 'all':
            return all(charge == 0 for charge in atom_charges)
        if remove_ionics == '+':
            return all(charge <= 0 for charge in atom_charges)
        return all(charge >= 0 for charge in atom_charges)

    # remove_ionics: 'all', '+', '-', or any other value to keep all molecules.
    def generate_smiles(
        self,
        target_num_samples,
        remove_ionics='all',
        num_attempts=100,
        generation_batch_size=128,
        max_new_tokens=100,
        excluded_smiles=None,
        min_carbon_atoms=1,
        min_heavy_atoms=2,
        max_heavy_atoms=None,
        max_molecular_weight=None,
        max_logp=None,
        allowed_elements=None,
    ):
        target_num_samples = max(0, int(target_num_samples))
        if target_num_samples == 0:
            self.last_generation_stats = {
                "target": 0,
                "attempt_budget": 0,
                "attempts": 0,
                "accepted": 0,
                "yield_rate": 1.0,
            }
            return []

        generation_batch_size = max(1, int(generation_batch_size))
        max_attempts = target_num_samples * max(1, int(num_attempts))
        excluded_smiles = set(excluded_smiles or ())
        generated_smiles = set()
        unique_valid_smiles = set()
        unique_known_smiles = set()
        attempts = 0
        valid_decodes = 0
        ionic_filtered = 0
        min_carbon_filtered = 0
        min_heavy_atom_filtered = 0
        max_heavy_atom_filtered = 0
        max_molecular_weight_filtered = 0
        max_logp_filtered = 0
        element_filtered = 0
        disallowed_element_counts = {}
        allowed_element_set = set(allowed_elements) if allowed_elements is not None else None
        started_at = time.perf_counter()

        # Sampling is deterministic for a given seed, model, and device class.
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        self.model.to(self.device)
        self.model.eval()

        start_token_id = self.tokenizer.bos_token_id
        if start_token_id is None:
            start_token_id = self.tokenizer.eos_token_id
        if start_token_id is None:
            start_token_id = self.tokenizer.pad_token_id
        input_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        next_progress_report = min(1000, max_attempts)

        # Both conditions are required: reach the target or stop at the budget.
        while len(generated_smiles) < target_num_samples and attempts < max_attempts:
            current_batch_size = min(generation_batch_size, max_attempts - attempts)
            try:
                with torch.inference_mode():
                    sample_output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(max_new_tokens),
                        num_return_sequences=current_batch_size,
                        temperature=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and generation_batch_size > 1:
                    generation_batch_size = max(1, generation_batch_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning(
                        "Generation batch did not fit in memory; retrying with batch size %s.",
                        generation_batch_size,
                    )
                    continue
                attempts += current_batch_size
                logger.error("Error during SMILES generation: %s", exc)
                continue
            except Exception as exc:
                attempts += current_batch_size
                logger.error("Error during SMILES generation: %s", exc)
                continue

            attempts += len(sample_output)
            decoded_smiles = self.tokenizer.batch_decode(
                sample_output, skip_special_tokens=True
            )
            for candidate in decoded_smiles:
                canonical_smiles, mol = canonicalize_smiles(candidate.strip())
                if canonical_smiles is None:
                    continue
                valid_decodes += 1
                unique_valid_smiles.add(canonical_smiles)
                if canonical_smiles in excluded_smiles:
                    unique_known_smiles.add(canonical_smiles)
                    continue
                if not self._passes_ionic_filter(mol, remove_ionics):
                    ionic_filtered += 1
                    continue
                if allowed_element_set is not None:
                    disallowed = sorted(
                        {
                            atom.GetSymbol()
                            for atom in mol.GetAtoms()
                            if atom.GetSymbol() not in allowed_element_set
                        }
                    )
                    if disallowed:
                        element_filtered += 1
                        for symbol in disallowed:
                            disallowed_element_counts[symbol] = (
                                disallowed_element_counts.get(symbol, 0) + 1
                            )
                        continue
                carbon_atoms = sum(
                    atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()
                )
                if carbon_atoms < max(0, int(min_carbon_atoms)):
                    min_carbon_filtered += 1
                    continue
                if mol.GetNumHeavyAtoms() < max(0, int(min_heavy_atoms)):
                    min_heavy_atom_filtered += 1
                    continue
                if (
                    max_heavy_atoms is not None
                    and mol.GetNumHeavyAtoms() > int(max_heavy_atoms)
                ):
                    max_heavy_atom_filtered += 1
                    continue
                if (
                    max_molecular_weight is not None
                    and Descriptors.MolWt(mol) > float(max_molecular_weight)
                ):
                    max_molecular_weight_filtered += 1
                    continue
                if max_logp is not None and Crippen.MolLogP(mol) > float(max_logp):
                    max_logp_filtered += 1
                    continue
                if canonical_smiles in generated_smiles:
                    continue
                generated_smiles.add(canonical_smiles)
                if len(generated_smiles) >= target_num_samples:
                    break

            if attempts >= next_progress_report or len(generated_smiles) >= target_num_samples:
                logger.info(
                    "Generated %s/%s valid unique SMILES after %s/%s attempts.",
                    len(generated_smiles),
                    target_num_samples,
                    attempts,
                    max_attempts,
                )
                next_progress_report += 1000

        elapsed = time.perf_counter() - started_at
        unique_valid_count = len(unique_valid_smiles)
        novel_unique_count = unique_valid_count - len(unique_known_smiles)
        self.last_generation_stats = {
            "target": target_num_samples,
            "attempt_budget": max_attempts,
            "attempts": attempts,
            "valid_decodes": valid_decodes,
            "unique_valid": unique_valid_count,
            "unique_known": len(unique_known_smiles),
            "ionic_filtered": ionic_filtered,
            "allowed_elements": sorted(allowed_element_set) if allowed_element_set is not None else None,
            "element_filtered": element_filtered,
            "disallowed_element_counts": dict(sorted(disallowed_element_counts.items())),
            "min_carbon_filtered": min_carbon_filtered,
            "min_heavy_atom_filtered": min_heavy_atom_filtered,
            "max_heavy_atom_filtered": max_heavy_atom_filtered,
            "max_molecular_weight_filtered": max_molecular_weight_filtered,
            "max_logp_filtered": max_logp_filtered,
            "accepted": len(generated_smiles),
            "validity_rate": valid_decodes / attempts if attempts else 0.0,
            "uniqueness_rate": unique_valid_count / valid_decodes if valid_decodes else 0.0,
            "novelty_rate": novel_unique_count / unique_valid_count if unique_valid_count else 0.0,
            "yield_rate": len(generated_smiles) / target_num_samples if target_num_samples else 1.0,
            "elapsed_seconds": elapsed,
            "attempts_per_second": attempts / elapsed if elapsed else 0.0,
            "accepted_per_second": len(generated_smiles) / elapsed if elapsed else 0.0,
        }

        if len(generated_smiles) < target_num_samples:
            logger.warning(
                "Only %s/%s valid unique SMILES were generated after %s attempts.",
                len(generated_smiles),
                target_num_samples,
                attempts,
            )
        return sorted(generated_smiles)
        
        
        

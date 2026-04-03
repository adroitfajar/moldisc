#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:03:48 2025

@author: kishan
"""

import os
import pandas as pd
import torch

class SmilesxWrapper():
    
    def __init__(self, data_smiles,
         data_prop,
         data_err = None,
         data_extra = None,
         data_name: str = 'Test',
         data_units: str = '',
         data_label: str  = '',
         smiles_concat: bool = False,
         outdir: str = './outputs',
         geomopt_mode: str ='off',
         bayopt_mode: str = 'off',
         train_mode: str = 'on',
         pretrained_data_name: str = '',
         pretrained_augm: str = False,
         model_type = 'regression', 
         scale_output = True, 
         embed_bounds: Optional[List[int]] = None,
         lstm_bounds: Optional[List[int]] = None,
         tdense_bounds: Optional[List[int]] = None,
         nonlin_bounds: Optional[List[int]] = None,
         bs_bounds: Optional[List[int]] = None,
         lr_bounds: Optional[List[float]] = None,
         embed_ref: Optional[int] = 512,
         lstm_ref: Optional[int] = 128,
         tdense_ref: Optional[int] = 128,
         dense_depth: Optional[int] = 0,
         bs_ref: int = 16,
         lr_ref: float = 3.9,
         k_fold_number: Optional[int] = 5,
         k_fold_index: Optional[List[int]] = None,
         run_index: Optional[List[int]] = None,
         n_runs: Optional[int] = None,
         check_smiles: bool = True,
         augmentation: bool = False,
         geom_sample_size: int = 32,
         bayopt_n_rounds: int = 25,
         bayopt_n_epochs: int = 30,
         bayopt_n_runs: int = 3,
         n_gpus: int = 1,
         gpus_list: Optional[List[int]] = None,
         gpus_debug: bool = False,
         patience: int = 25,
         n_epochs: int = 100,
         batchsize_pergpu: Optional[int] = None,
         lr_schedule: Optional[str] = None,
         bs_increase: bool = False,
         ignore_first_epochs: int = 0,
         lr_min: float = 1e-5,
         lr_max: float = 1e-2,
         prec: int = 4,
         log_verbose: bool = True,
         train_verbose: bool = True,
         device=0):
        #super(SmilesxWrapper, self).__init__()
        
        self.data_smiles = ''
        self.model = ''
        self.predictions = []
        self.scores_folds = []
        
    def load_datsaset(self, data_smiles,  
             data_prop,
             data_err = None,
             data_extra = None,
             data_name: str = 'Test',
             data_units: str = '',
             data_label: str  = '',
             smiles_concat: bool = False):
        
        self.data_smiles = data_smiles
        # Reading the data
        header = []
        data_smiles = data_smiles.replace([np.nan, None], ["", ""]).values
        if data_smiles.ndim==1:
            data_smiles = data_smiles.reshape(-1,1)
            header.extend(["SMILES"])
        elif data_smiles.shape[1]==1:
            data_smiles = data_smiles.reshape(-1,1)
            header.extend(["SMILES"])
        else:
            for i in range(data_smiles.shape[1]):
                header.extend(["SMILES_{}".format(i+1)])
        data_prop = data_prop.values
        header.extend([data_label])
        if data_err is not None:
            if data_err.ndim==1:
                data_err = data_err.reshape(-1,1)
            if data_err.shape[1] == 1:
                header.extend(["Standard deviation"])
                err_bars = 'std'
            elif data_err.shape[1] == 2:
                header.extend(["Minimum", "Maximum"])
                err_bars = 'minmax'
            data_err = data_err.values
        else:
            err_bars = None
        if data_extra is not None:
            header.extend(data_extra.columns)
            data_extra = data_extra.values
            extra_dim = data_extra.shape[1]
        else:
            extra_dim = None
        if data_label=='':
            data_label = data_name    
    
        # Initialize Predictions.txt and Scores.csv files
        predictions = np.concatenate([arr for arr in (data_smiles, data_prop.reshape(-1,1), data_err, data_extra) if arr is not None], axis=1)
        self.predictions = pd.DataFrame(predictions)
        self.predictions.columns = header
        

    def build_dataset(self,):
        
        
        # Setting up GPUs
        #strategy, gpus = utils.set_gpuoptions(n_gpus=n_gpus,
        #                                      gpus_list=gpus_list,
        #                                      gpus_debug=gpus_debug)

    
        # Setting up the scores summary
        self.scores_summary = {'train': [],
                          'valid': [],
                          'test': []}
    
        if ignore_first_epochs >= n_epochs:
                logging.error("ERROR:")
                logging.error("The number of ignored epochs `ignore_first_epochs` should be less than")
                logging.error("the total number of training epochs `n_epochs`.")
                logging.error("")
                logging.error("*** SMILES-X EXECUTION ABORTED ***")
                raise utils.StopExecution
    
        # Retrieve the models for training in case of transfer learning
        if train_mode == 'finetune':
            if len(pretrained_data_name) == 0:
                logging.error("ERROR:")
                logging.error("Cannot determine the pretrained model path.")
                logging.error("Please, specify the name of the data used for the pretraining (`pretrained_data_name`)")
                logging.error("")
                logging.error("*** SMILES-X EXECUTION ABORTED ***")
                raise utils.StopExecution
            if k_fold_number is None:
                # If the dataset is too small to transfer the number of kfolds
                if model.k_fold_number > data.shape[0]:
                    k_fold_number = data.shape[0]
                    logging.info("The number of cross-validation folds (`k_fold_number`) is not defined.")
                    logging.info("Borrowing it from the pretrained model...")
                    logging.info("Number of folds `k_fold_number` is set to {}". format(k_fold_number))
                else:
                    k_fold_number = model.k_fold_number
                    logging.info("The number of cross-validation folds (`k_fold_number`)")
                    logging.info("used for the pretrained model is too large to be used with current data:")
                    logging.info("size of the data is too small ({} > {})".format(model.k_fold_number, data.shape[0]))
                    logging.info("The number of folds is set to the length of the data ({})". format(k_fold_number))
            if n_runs is None:
                logging.info("The number of runs per fold (`n_runs`) is not defined.")
                logging.info("Borrowing it from the pretrained model...")
                logging.info("Number of runs `n_runs` is set to {}". format(model.n_runs))
            logging.info("Fine tuning has been requested, loading pretrained model...")
            pretrained_model = loadmodel.LoadModel(data_name = pretrained_data_name,
                                                   outdir = outdir,
                                                   augmentation = pretrained_augm,
                                                   gpu_name = gpus[0].name,
                                                   strategy = strategy, 
                                                   return_attention=False, # no need to return attention for transfer learning
                                                   model_type = model_type,
                                                   extra = (data_extra!=None),
                                                   scale_output=scale_output, 
                                                   k_fold_number = k_fold_number)
        else:
            if k_fold_number is None:
                logging.error("ERROR:")
                logging.error("The number of cross-validation folds (`k_fold_number`) is not defined.")
                logging.error("")
                logging.error("*** SMILES-X EXECUTION ABORTED ***")
                raise utils.StopExecution
            if n_runs is None:
                logging.error("ERROR:")
                logging.error("The number of runs per fold (`n_runs`) is not defined.")
                logging.error("")
                logging.error("*** SMILES-X EXECUTION ABORTED ***")
                raise utils.StopExecution
            pretrained_model = None
    
        # Setting up the cross-validation according to the model_type
        # For regression
        # Splitting is done based on groups of the provided SMILES data
        # This is done for the cases where the same SMILES has multiple entries with
        # varying additional parameters (molecular weight, proportion, processing time, etc.)
        # For classification
        # Splitting is done based on the provided property (e.g. class) data
        if model_type == 'regression':
            groups = pd.DataFrame(data_smiles).groupby(by=0).ngroup().values.tolist()
            kf = GroupKFold(n_splits=k_fold_number)
            kf.get_n_splits(X=data_smiles, groups=groups)
            kf_splits = kf.split(X=data_smiles, groups=groups)
            model_loss = 'mse'
            model_metrics = [metrics.mae, metrics.mse]
        elif model_type == 'classification':
            scale_output = False
            kf = StratifiedKFold(n_splits=k_fold_number, shuffle=True, random_state=42)
            kf.get_n_splits(X=data_smiles, y=data_prop)
            kf_splits = kf.split(X=data_smiles, y=data_prop)
            model_loss = 'binary_crossentropy'
            model_metrics = ['accuracy']
         
        # Individual counter for the folds of interest in case of k_fold_index
        nfold = 0
        for ifold, (train_val_idx, test_idx) in enumerate(kf_splits):
            start_fold = time.time()
    
            # In case only some of the folds are requested for training
            if k_fold_index is not None:
                k_fold_number = len(k_fold_index)
                if ifold not in k_fold_index:
                    continue
            
            # Keep track of the fold number for every data point
            predictions.loc[test_idx, 'Fold'] = ifold
    
            # Estimate remaining training duration based on the first fold duration
            if nfold > 0:
                if nfold == 1:
                    onefold_time = time.time() - start_time # First fold's duration
                elif nfold < (k_fold_number - 1):
                    logging.info("Remaining time: {:.2f} h. Processing fold #{} of data..."\
                                 .format((k_fold_number - nfold) * onefold_time/3600., ifold))
                elif nfold == (k_fold_number - 1):
                    logging.info("Remaining time: {:.2f} h. Processing the last fold of data..."\
                                 .format(onefold_time/3600.))
    
            logging.info("")
            logging.info("***Fold #{} initiated...***".format(ifold))
            logging.info("")
            
            logging.info("***Splitting and standardization of the dataset.***")
            logging.info("")
            x_train, x_valid, x_test, \
            extra_train, extra_valid, extra_test, \
            y_train, y_valid, y_test, \
            y_err_train, y_err_valid, y_err_test = utils.rand_split(smiles_input = data_smiles,
                                                                    prop_input = data_prop,
                                                                    extra_input = data_extra,
                                                                    err_input = data_err,
                                                                    train_val_idx = train_val_idx,
                                                                    test_idx = test_idx)
            # Scale the outputs
            if scale_output:
                scaler_out_file = '{}/{}_Scaler_Outputs'.format(scaler_dir, data_name)
                y_train_scaled, y_valid_scaled, y_test_scaled, scaler = utils.robust_scaler(train=y_train,
                                                                                            valid=y_valid,
                                                                                            test=y_test,
                                                                                            file_name=scaler_out_file,
                                                                                            ifold=ifold)
            else:
                y_train_scaled, y_valid_scaled, y_test_scaled, scaler = y_train, y_valid, y_test, None
            # Scale the auxiliary numeric inputs (if given) 
            if data_extra is not None:
                scaler_extra_file = '{}/{}_Scaler_Extra'.format(scaler_dir, data_name)
                extra_train, extra_valid, extra_test, scaler_extra = utils.robust_scaler(train=extra_train,
                                                                                         valid=extra_valid,
                                                                                         test=extra_test,
                                                                                         file_name=scaler_extra_file,
                                                                                         ifold=ifold)
    
            # Check/augment the data if requested
            train_augm = augm.augmentation(x_train,
                                           train_val_idx,
                                           extra_train,
                                           y_train_scaled,
                                           check_smiles,
                                           augmentation)
    
            valid_augm = augm.augmentation(x_valid,
                                           train_val_idx,
                                           extra_valid,
                                           y_valid_scaled,
                                           check_smiles,
                                           augmentation)
    
            test_augm = augm.augmentation(x_test,
                                          test_idx,
                                          extra_test,
                                          y_test_scaled,
                                          check_smiles,
                                          augmentation)
            
            x_train_enum, extra_train_enum, y_train_enum, y_train_clean, x_train_enum_card, _ = train_augm
            x_valid_enum, extra_valid_enum, y_valid_enum, y_valid_clean, x_valid_enum_card, _ = valid_augm
            x_test_enum, extra_test_enum, y_test_enum, y_test_clean, x_test_enum_card, test_idx_clean = test_augm
                    
            # Concatenate multiple SMILES into one via 'j' joint
            if smiles_concat:
                x_train_enum = utils.smiles_concat(x_train_enum)
                x_valid_enum = utils.smiles_concat(x_valid_enum)
                x_test_enum = utils.smiles_concat(x_test_enum)
            
            logging.info("Enumerated SMILES:")
            logging.info("\tTraining set: {}".format(len(x_train_enum)))
            logging.info("\tValidation set: {}".format(len(x_valid_enum)))
            logging.info("\tTest set: {}".format(len(x_test_enum)))
            logging.info("")
    
            logging.info("***Tokenization of SMILES.***")
            logging.info("")
    
            # Tokenize SMILES per dataset
            x_train_enum_tokens = token.get_tokens(x_train_enum)
            x_valid_enum_tokens = token.get_tokens(x_valid_enum)
            x_test_enum_tokens = token.get_tokens(x_test_enum)
    
            logging.info("Examples of tokenized SMILES from a training set:")
            logging.info("{}".format(x_train_enum_tokens[:5]))
            logging.info("")
    
            # Vocabulary size computation
            all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens
    
            # Check if the vocabulary for current dataset exists already
            vocab_file = '{}/Other/{}_Vocabulary.txt'.format(save_dir, data_name)
            if os.path.exists(vocab_file):
                tokens = token.get_vocab(vocab_file)
            else:
                tokens = token.extract_vocab(all_smiles_tokens)
                token.save_vocab(tokens, vocab_file)
                tokens = token.get_vocab(vocab_file)
    
            # TODO(kathya): add info on how much previous model vocabs differ from the current data train/val/test vocabs
            #               (for transfer learning)
            train_unique_tokens = token.extract_vocab(x_train_enum_tokens)
            logging.info("Number of tokens only present in training set: {}".format(len(train_unique_tokens)))
            logging.info("")
    
            valid_unique_tokens = token.extract_vocab(x_valid_enum_tokens)
            logging.info("Number of tokens only present in validation set: {}".format(len(valid_unique_tokens)))
            if valid_unique_tokens.issubset(train_unique_tokens):
                logging.info("Validation set contains no new tokens comparing to training set tokens")
            else:
                logging.info("Validation set contains the following new tokens comparing to training set tokens:")
                logging.info(valid_unique_tokens.difference(train_unique_tokens))
                logging.info("")
    
            test_unique_tokens = token.extract_vocab(x_test_enum_tokens)
            logging.info("Number of tokens only present in a test set: {}".format(len(test_unique_tokens)))
            if test_unique_tokens.issubset(train_unique_tokens):
                logging.info("Test set contains no new tokens comparing to the training set tokens")
            else:
                logging.info("Test set contains the following new tokens comparing to the training set tokens:")
                logging.info(test_unique_tokens.difference(train_unique_tokens))
    
            if test_unique_tokens.issubset(valid_unique_tokens):
                logging.info("Test set contains no new tokens comparing to the validation set tokens")
            else:
                logging.info("Test set contains the following new tokens comparing to the validation set tokens:")
                logging.info(test_unique_tokens.difference(valid_unique_tokens))
                logging.info("")
    
            # Add 'pad' (padding), 'unk' (unknown) tokens to the existing list
            tokens.insert(0,'unk')
            tokens.insert(0,'pad')
    
            logging.info("Full vocabulary: {}".format(tokens))
            logging.info("Vocabulary size: {}".format(len(tokens)))
            logging.info("")
    
            # Maximum of length of SMILES to process
            max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
            logging.info("Maximum length of tokenized SMILES: {} tokens (termination spaces included)".format(max_length))
            logging.info("")
    
            # predict and compare for the training, validation and test sets
            x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_train_enum_tokens,
                                                                max_length=max_length + 1,
                                                                vocab=tokens)
            x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_valid_enum_tokens,
                                                                max_length=max_length + 1,
                                                                vocab=tokens)
            x_test_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_test_enum_tokens,
                                                               max_length=max_length + 1,
                                                               vocab=tokens)
            # Hyperparameters optimisation
            if nfold==0:
                logging.info("*** HYPERPARAMETERS OPTIMISATION ***")
                logging.info("")
    
            # Dictionary to store optimized hyperparameters
            # Initialize at reference values, update gradually
                hyper_opt = {'Embedding': embed_ref,
                             'LSTM': lstm_ref,
                             'TD dense': tdense_ref,
                             'Batch size': bs_ref,
                             'Learning rate': lr_ref}
                hyper_bounds = {'Embedding': embed_bounds,
                                'LSTM': lstm_bounds,
                                'TD dense': tdense_bounds,
                                'Batch size': bs_bounds,
                                'Learning rate': lr_bounds}
                
                # Geometry optimisation
                if geomopt_mode == 'on':
                    geom_file = '{}/Other/{}_GeomScores.csv'.format(save_dir, data_name)
                    # Do not optimize the architecture in case of transfer learning
                    if train_mode=='finetune':
                        logging.info("Transfer learning is requested together with geometry optimisation,")
                        logging.info("but the architecture is already fixed in the original model.")
                        logging.info("Only batch size and learning rate can be tuned.")
                        logging.info("Skipping geometry optimisation...")
                        logging.info("")
                    else:
                        hyper_opt, hyper_bounds = \
                        geomopt.geom_search(data_token=x_train_enum_tokens_tointvec,
                                            data_extra=extra_train_enum,
                                            subsample_size=geom_sample_size,
                                            hyper_bounds=hyper_bounds,
                                            hyper_opt=hyper_opt,
                                            dense_depth=dense_depth,
                                            vocab_size=len(tokens),
                                            max_length=max_length,
                                            geom_file=geom_file,
                                            strategy=strategy, 
                                            model_type=model_type)
                else:
                    logging.info("Trainless geometry optimisation is not requested.")
                    logging.info("")
    
                 # Bayesian optimisation
                if bayopt_mode == 'on':
                    if geomopt_mode == 'on':
                        logging.info("*Note: Geometry-related hyperparameters will not be updated during the Bayesian optimisation.")
                        logging.info("")
                        if not any([bs_bounds, lr_bounds]):
                            logging.info("Batch size bounds and learning rate bounds are not defined.")
                            logging.info("Bayesian optimisation has no parameters to optimize.")
                            logging.info("Skipping...")
                            logging.info("")
                    hyper_opt = bayopt.bayopt_run(smiles=data_smiles,
                                                  prop=data_prop,
                                                  extra=data_extra,
                                                  train_val_idx=train_val_idx,
                                                  smiles_concat=smiles_concat,
                                                  tokens=tokens,
                                                  max_length=max_length,
                                                  check_smiles=check_smiles,
                                                  augmentation=augmentation,
                                                  hyper_bounds=hyper_bounds,
                                                  hyper_opt=hyper_opt,
                                                  dense_depth=dense_depth,
                                                  bo_rounds=bayopt_n_rounds,
                                                  bo_epochs=bayopt_n_epochs,
                                                  bo_runs=bayopt_n_runs,
                                                  strategy=strategy,
                                                  model_type=model_type, 
                                                  scale_output=scale_output, 
                                                  pretrained_model=pretrained_model)
                else:
                    logging.info("Optuna-based Bayesian optimisation is not requested.")
                    logging.info("")
                    if geomopt == 'off':
                        logging.info("Using reference values for training.")
                        logging.info("")
    
                hyper_df = pd.DataFrame([hyper_opt.values()], columns = hyper_opt.keys())
                hyper_file = "{}/Other/{}_Hyperparameters.csv".format(save_dir, data_name)
                hyper_df.to_csv(hyper_file, index=False)
    
                logging.info("*** HYPERPARAMETERS OPTIMISATION COMPLETED ***")
                logging.info("")
                
                logging.info("The following hyperparameters will be used for training:")
                for key in hyper_opt.keys():
                    if key == "Learning rate":
                        logging.info("    - {}: 10^-{}".format(key, hyper_opt[key]))
                    else:
                        logging.info("    - {}: {}".format(key, hyper_opt[key]))
                logging.info("")
                logging.info("File containing the list of used hyperparameters:")
                logging.info("    {}".format(hyper_file))
                logging.info("")
    
                logging.info("*** TRAINING ***")
                logging.info("")
            start_train = time.time()
            prediction_train_bag = np.zeros((y_train_enum.shape[0], n_runs))
            prediction_valid_bag = np.zeros((y_valid_enum.shape[0], n_runs))
            prediction_test_bag = np.zeros((y_test_enum.shape[0], n_runs))        


    def train(self, ):
        
        
    def inference(self, ):
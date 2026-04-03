"""Add main docstring discription

"""

import time
import math
import logging
import datetime

import numpy as np

import optuna
from optuna.samplers import TPESampler

from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from SMILESX import utils, augm, token, model, trainutils

def bayopt_run(smiles, prop, extra, train_val_idx, smiles_concat, tokens, max_length, check_smiles, augmentation, hyper_bounds, hyper_opt, dense_depth, bo_rounds, bo_epochs, bo_runs, strategy, model_type, scale_output, pretrained_model=None):
    '''Bayesian optimization of hyperparameters.

    Parameters
    ----------
    smiles: np.array
        Input SMILES.
    prop: np.array
        Input property values.
    extra: np.array
        Additional data input.
    train_val_idx: list
        List of indices for training and validation data for the current fold.
    tokens: list
        List of tokens contained within the dataset.
    max_length: int
        Maximum length of SMILES in training and validation data.
    check_smiles: bool
        Whether to check SMILES validity with RDKit.
    augmentation: bool
        Whether to perform data augmentation during bayesian optimization process.
    hyper_bounds: dict
        A dictionary of bounds {"param":[bounds]}, where parameter `"param"` can be
        embedding, LSTM, time-distributed dense layer units, batch size or learning
        rate, and `[bounds]` is a list of possible values to be tested during
        Bayesian optimization for a given parameter.
    hyper_opt: dict
        A dictionary of bounds {"param":val}, where parameter `"param"` can be
        embedding, LSTM, time-distributed dense layer units, batch size or learning
        rate, and `val` is default value for a given parameter.
    dense_depth: int
        Number of additional dense layers to be appended after attention layer.
    bo_rounds: int
        Number of rounds to be used during Bayesian optimization.
    bo_epochs: int
        Number of epochs required for training within the optimization frame.
    bo_runs: int
        Number of training repetitions with random train/val split.
    strategy:
        GPU memory growth strategy.
    model_type: str
        Type of the model to be used. Can be either 'regression' or 'classification'.
    scale_output: bool
        Whether to scale the output property values or not. For binary classification tasks, it is recommended not to scale 
        the categorical (e.g. 0, 1) output values. For regression tasks, this is preferable to guarantee quicker 
        training convergence.
    pretrained_model:
        Pretrained model in case of the transfer learning (`train_mode='finetune'`).
        (Default: None)
            
    Returns
    -------
    hyper_opt: dictdata_prop
        Dictionary with hyperparameters updated with optimized values
    '''
    # Get the logger for smooth logging
    logger = logging.getLogger()

    logging.info("*** Bayesian optimization ***")
    logging.info("")

    # Identify which parameters to optimize via Bayesian optimisation
    if not any(hyper_bounds.values()):
        logging.warning("ATTENTION! Bayesian optimisation is requested, but no bounds are given.")
        logging.info("")
        logging.warning("Specify at least one of the following:")
        logging.warning("      - embed_bounds")
        logging.warning("      - lstm_bounds")
        logging.warning("      - tdense_bounds")
        logging.warning("      - bs_bounds")
        logging.warning("      - lr_bounds")
        logging.info("")
        logging.warning("If no Bayesian optimisation is desired, set `bayopt_mode='off'`.")
        logging.info("")
        logging.warning("The SMILES-X execution is aborted.")
        raise utils.StopExecution
    search_space = {key: bounds for key, bounds in hyper_bounds.items() if bounds is not None}
    logging.info('Bayesian optimisation is requested for:')
    for key in search_space.keys():
        logging.info('      - {}'.format(key))
    logging.info('*Note: selected hyperparameters are sampled jointly via Optuna.')
    logging.info("")

    # Set up Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(multivariate=True, seed=42)
    ordered_params = list(search_space.keys())
    extra_dim = extra.shape[1] if extra is not None else None

    def objective(trial):
        """Optuna objective: returns mean validation score for sampled hyperparameters."""
        trial_params = hyper_opt.copy()
        chosen_values = []
        for key in ordered_params:
            suggestion = trial.suggest_categorical(key, search_space[key])
            trial_params[key] = suggestion
            chosen_values.append(suggestion)
        logging.info("Trial #%d hyperparameters: %s", trial.number, chosen_values)

        score_valids = []
        for irun in range(bo_runs):
            # Random train/val splitting for every run to assure better generalizability
            x_train, x_valid, extra_train, extra_valid, y_train, y_valid = utils.rand_split(smiles_input=smiles,
                                                                                            prop_input=prop,
                                                                                            extra_input=extra,
                                                                                            err_input=None,
                                                                                            train_val_idx=train_val_idx,
                                                                                            test_idx=None,
                                                                                            bayopt=True)
            # Scale the outputs when required
            if scale_output:
                y_train_scaled, y_valid_scaled, _, _ = utils.robust_scaler(train=y_train,
                                                                           valid=y_valid,
                                                                           test=None,
                                                                           file_name=None,
                                                                           ifold=None)
            else:
                y_train_scaled, y_valid_scaled = y_train, y_valid
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

            x_train_enum, extra_train_enum, y_train_enum, y_train_clean, x_train_enum_card, _ = train_augm
            x_valid_enum, extra_valid_enum, y_valid_enum, y_valid_clean, x_valid_enum_card, _ = valid_augm

            # Concatenate multiple SMILES into one via 'j' joint
            if smiles_concat:
                x_train_enum = utils.smiles_concat(x_train_enum)
                x_valid_enum = utils.smiles_concat(x_valid_enum)

            x_train_enum_tokens = token.get_tokens(x_train_enum)
            x_valid_enum_tokens = token.get_tokens(x_valid_enum)
            x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_train_enum_tokens,
                                                                max_length=max_length + 1,
                                                                vocab=tokens)
            x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_valid_enum_tokens,
                                                                max_length=max_length + 1,
                                                                vocab=tokens)

            K.clear_session()
            #TODO(Guillaume): Check pretraining case
            if pretrained_model is not None:
                # Load the pretrained model
                model_train = pretrained_model.model_dic['Fold_{}'.format(ifold)][run]
                # Freeze encoding layers
                #TODO(Guillaume): Check if this is the best way to freeze the layers as layers' name may differ
                for layer in model_train.layers:
                    if layer.name in ['embedding', 'bidirectional', 'time_distributed']:
                        layer.trainable = False

                logging.info("Retrieved model summary:")
                model_train.summary(print_fn=logging.info)
                logging.info("\n")
            else:
                with strategy.scope():
                    model_opt = model.LSTMAttModel.create(input_tokens=max_length + 1,
                                                          extra_dim=extra_dim,
                                                          vocab_size=len(tokens),
                                                          embed_units=trial_params['Embedding'],
                                                          lstm_units=trial_params['LSTM'],
                                                          tdense_units=trial_params['TD dense'],
                                                          dense_depth=dense_depth,
                                                          model_type=model_type)

            if model_type == 'regression':
                model_loss = 'mse'
                model_metrics = [metrics.mae, metrics.mse]
                hist_val_name = 'val_mean_squared_error'
            elif model_type == 'classification':
                model_loss = 'binary_crossentropy'
                model_metrics = ['accuracy']
                hist_val_name = 'val_loss'

            with strategy.scope():
                batch_size = int(trial_params['Batch size']) * strategy.num_replicas_in_sync
                batch_size_val = min(len(x_train_enum_tokens_tointvec), batch_size)
                custom_adam = Adam(lr=math.pow(10, -float(trial_params['Learning rate'])))
                model_opt.compile(loss=model_loss, optimizer=custom_adam, metrics=model_metrics)

                history = model_opt.fit_generator(generator=\
                                                  trainutils.DataSequence(x_train_enum_tokens_tointvec,
                                                                          extra_train_enum,
                                                                          y_train_enum,
                                                                          batch_size),
                                                  validation_data=\
                                                  trainutils.DataSequence(x_valid_enum_tokens_tointvec,
                                                                          extra_valid_enum,
                                                                          y_valid_enum,
                                                                          batch_size_val),
                                                  epochs=bo_epochs,
                                                  shuffle=True,
                                                  initial_epoch=0,
                                                  verbose=0)

            # Skip the first half of epochs during evaluation to ignore the burn-in period
            best_epoch = np.argmin(history.history['val_loss'][int(bo_epochs//2):])
            score_valid = history.history[hist_val_name][best_epoch + int(bo_epochs//2)]

            if math.isnan(score_valid):
                score_valid = math.inf
            score_valids.append(score_valid)

        mean_score = float(np.mean(score_valids))
        logging.info("Trial #%d average best validation score: %.4f", trial.number, mean_score)
        return mean_score

    start_bo = time.time()

    study = optuna.create_study(direction="minimize", sampler=sampler)
    n_trials = max(int(bo_rounds), 1)
    study.optimize(objective, n_trials=n_trials)

    for key, value in study.best_params.items():
        hyper_opt[key] = value

    elapsed_bo = time.time() - start_bo

    logging.info("")
    logging.info("*** Bayesian hyperparameters optimization is completed ***")
    logging.info("")
    logging.info("Bayesian optimisation duration: {}".format(str(datetime.timedelta(seconds=elapsed_bo))))
    for key in search_space.keys():
        logging.info("    - {}: {}".format(key, hyper_opt[key]))
    logging.info("")

    return hyper_opt
##
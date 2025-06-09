"""
Main script to train and evaluate deep learning models for fake news detection using K-Fold cross-validation.
This script sets up the training pipeline, loads the dataset, initializes the model and trainer,
and performs training and evaluation across multiple folds.
"""

from loguru import logger
import os
import argparse
from typing import *
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from deep_learning.utils.config import MODEL_PARAMS, TRAINER_CONFIG_PARAMS
from deep_learning.utils.load_dataset import load_dataset
from trainer.dl_trainer import DeepLearningTrainer


def run_single_fold(fold_id: int, train_idx: np.ndarray, val_idx: np.ndarray,
                    all_texts: np.ndarray, all_labels: np.ndarray,
                    all_raw_num_feat: Optional[np.ndarray],
                    model_config: Dict[str, Any],
                    trainer_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run a single fold of training and evaluation.
    Args:
        fold_id (int): The ID of the current fold.
        train_idx (np.ndarray): Indices for training data.
        val_idx (np.ndarray): Indices for validation data.
        all_texts (np.ndarray): All texts in the dataset.
        all_labels (np.ndarray): All labels in the dataset.
        all_raw_num_feat (Optional[np.ndarray]): All raw numerical features, if available.
        model_config (Dict[str, Any]): Configuration parameters for the model.
        trainer_config (Dict[str, Any]): Configuration parameters for the trainer.
    Returns:
        Optional[Dict[str, Any]]: Evaluation metrics for the fold, or None if an error occurs.
    """

    model_name = f"{model_config['rnn_type']}_fold_{fold_id}"
    fold_output_dir = os.path.join(trainer_config['output_dir_base'], f"{model_config['rnn_type']}/fold_{fold_id}")
    os.makedirs(fold_output_dir, exist_ok=True)
    logger.info(f"--- Starting Fold {fold_id}/{trainer_config['n_splits']} ---")
    logger.info(f"Output directory for this fold: {fold_output_dir}")

    texts_train_fold, labels_train_fold = all_texts[train_idx], all_labels[train_idx]
    texts_val_fold, labels_val_fold = all_texts[val_idx], all_labels[val_idx]

    num_feat_train_fold_scaled, num_feat_val_fold_scaled = None, None
    fold_scaler = None

    if all_raw_num_feat is not None and model_config.get('num_numerical_features', 0) > 0:
        num_feat_train_fold_raw = all_raw_num_feat[train_idx]
        num_feat_val_fold_raw = all_raw_num_feat[val_idx]

        fold_scaler = StandardScaler()
        logger.info(f"Fold {fold_id}: Fitting StandardScaler on training numerical features of this fold.")
        num_feat_train_fold_scaled = fold_scaler.fit_transform(num_feat_train_fold_raw)
        logger.info(f"Fold {fold_id}: Transforming validation numerical features of this fold.")
        num_feat_val_fold_scaled = fold_scaler.transform(num_feat_val_fold_raw)
    
    current_model_params = model_config.copy()

    trainer = DeepLearningTrainer(
        model_name=model_name,
        model_params=current_model_params,
        tokenizer_name=trainer_config['tokenizer_name'],
        lr=trainer_config['lr'],
        num_epochs=trainer_config['num_epochs'],
        batch_size=trainer_config['batch_size'],
        max_len=trainer_config['max_len'],
        early_stopping_patience=trainer_config['early_stopping_patience'],
        metric_for_best_model=trainer_config['metric_for_best_model'],
        output_dir=fold_output_dir,
        seed=trainer_config['seed_base'] + fold_id,
        device=trainer_config['device'],
        optimizer_name=trainer_config.get('optimizer_name', 'adamw'),
        lr_scheduler_name=trainer_config.get('lr_scheduler_name', 'linear_warmup'),
        warmup_steps_ratio=trainer_config.get('warmup_steps_ratio', 0.001),
        criterion_name=trainer_config.get('criterion_name', 'cross_entropy')
    )
    

    try:
        trainer.fit(
            train_texts=texts_train_fold.tolist(),
            train_labels=labels_train_fold.tolist(),
            train_numerical_features=num_feat_train_fold_scaled.tolist() if num_feat_train_fold_scaled is not None else None,
            val_texts=texts_val_fold.tolist(),
            val_labels=labels_val_fold.tolist(),
            val_numerical_features=num_feat_val_fold_scaled.tolist() if num_feat_val_fold_scaled is not None else None
        )
    except Exception as e_fit:
        logger.error(f"Error during training in fold {fold_id}: {e_fit}")
        return None

        
    try:
        val_fold_results = trainer.evaluate(
            texts=texts_val_fold.tolist(),
            labels=labels_val_fold.tolist(),
            numerical_features=num_feat_val_fold_scaled.tolist() if num_feat_val_fold_scaled is not None else None,
            model_path=os.path.join(fold_output_dir, f"{model_name}.pth"),
            stage=f"Fold_{fold_id}_Validation"
        )
        logger.info(f"--- Finished Fold {fold_id} ---")
        return val_fold_results['metrics']
    except Exception as e_eval:
        logger.error(f"Error during evaluation in fold {fold_id} (after training): {e_eval}")
        if hasattr(trainer, 'history') and trainer.history['val_f1']:
            best_val_epoch_idx = -1
            if trainer.metric_for_best_model == 'val_loss':
                if trainer.history['val_loss']: best_val_epoch_idx = np.argmin(trainer.history['val_loss'])
            else:
                metric_key = 'val_' + trainer.metric_for_best_model
                if metric_key in trainer.history and trainer.history[metric_key]:
                     best_val_epoch_idx = np.argmax(trainer.history[metric_key])
            
            if best_val_epoch_idx != -1:
                logger.warning(f"Evaluation failed for fold {fold_id}, returning metrics from best epoch in history ({best_val_epoch_idx+1}).")
                return {
                    'accuracy': trainer.history['val_accuracy'][best_val_epoch_idx],
                    'precision': trainer.history['val_precision'][best_val_epoch_idx],
                    'recall': trainer.history['val_recall'][best_val_epoch_idx],
                    'f1': trainer.history['val_f1'][best_val_epoch_idx],
                }
        return None


def main():
    """
    Main function to run the training and evaluation pipeline.
    It sets up the argument parser, loads the dataset, initializes the model and trainer,
    and performs K-Fold cross-validation.
    """
    logger.info("--- Starting the training and evaluation pipeline for fake news detection ---")

    parser = argparse.ArgumentParser(description="Train and evaluate deep learning models for fake news detection.")

    # --- MODEL_PARAMS Arguments ---
    parser.add_argument('--cnn_out_channels_per_kernel', type=int, default=MODEL_PARAMS['cnn_out_channels_per_kernel'], help='CNN output channels per kernel.')
    parser.add_argument('--cnn_kernel_sizes', type=int, nargs='+', default=MODEL_PARAMS['cnn_kernel_sizes'], help='List of CNN kernel sizes.')
    parser.add_argument('--cnn_dropout_prob', type=float, default=MODEL_PARAMS['cnn_dropout_prob'], help='CNN dropout probability.')
    parser.add_argument('--rnn_hidden_dim', type=int, default=MODEL_PARAMS['rnn_hidden_dim'], help='RNN hidden dimension.')
    parser.add_argument('--n_rnn_layers', type=int, default=MODEL_PARAMS['n_rnn_layers'], help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout_prob', type=float, default=MODEL_PARAMS['rnn_dropout_prob'], help='RNN dropout probability.')
    parser.add_argument('--rnn_type', type=str, choices=['rnn', 'lstm', 'gru'], default=MODEL_PARAMS['rnn_type'], help='Type of RNN to use.')
    parser.add_argument('--fc_dropout_prob', type=float, default=MODEL_PARAMS['fc_dropout_prob'], help='FC dropout probability.')
    parser.add_argument('--feature_integration_method', type=str, default=MODEL_PARAMS['feature_integration_method'], help='Feature integration method.')
    parser.add_argument('--freeze_bert_completely', type=bool, default=MODEL_PARAMS['freeze_bert_completely'], help='Whether to freeze BERT completely.')


    # --- TRAINER_CONFIG_PARAMS Arguments ---
    parser.add_argument('--lr', type=float, default=TRAINER_CONFIG_PARAMS['lr'], help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=TRAINER_CONFIG_PARAMS['num_epochs'], help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=TRAINER_CONFIG_PARAMS['batch_size'], help='Batch size.')
    parser.add_argument('--max_len', type=int, default=TRAINER_CONFIG_PARAMS['max_len'], help='Max sequence length.')
    parser.add_argument('--early_stopping_patience', type=int, default=TRAINER_CONFIG_PARAMS['early_stopping_patience'], help='Early stopping patience.')
    parser.add_argument('--metric_for_best_model', type=str, default=TRAINER_CONFIG_PARAMS['metric_for_best_model'], help='Metric for best model selection.')
    parser.add_argument('--n_splits', type=int, default=TRAINER_CONFIG_PARAMS['n_splits'], help='Number of K-Fold splits.')
    parser.add_argument('--seed_base', type=int, default=TRAINER_CONFIG_PARAMS['seed_base'], help='Base random seed.')
    parser.add_argument('--output_dir_base', type=str, default=TRAINER_CONFIG_PARAMS['output_dir_base'], help='Base output directory.')
    parser.add_argument('--optimizer_name', type=str, default=TRAINER_CONFIG_PARAMS['optimizer_name'], help='Optimizer name.')
    parser.add_argument('--lr_scheduler_name', type=str, default=TRAINER_CONFIG_PARAMS['lr_scheduler_name'], help='LR scheduler name.')
    parser.add_argument('--warmup_steps_ratio', type=float, default=TRAINER_CONFIG_PARAMS['warmup_steps_ratio'], help='Warmup steps ratio.')
    parser.add_argument('--criterion_name', type=str, default=TRAINER_CONFIG_PARAMS['criterion_name'], help='Criterion name.')

    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data CSV file.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the testing data CSV file.')

    args = parser.parse_args()

    # --- Update MODEL_PARAMS and TRAINER_CONFIG_PARAMS from CLI args ---
    # Create copies to modify
    current_model_params = MODEL_PARAMS.copy()
    current_trainer_config_params = TRAINER_CONFIG_PARAMS.copy()

    # Update based on CLI arguments, respecting non-overridable params
    for arg_name, arg_value in vars(args).items():
        if arg_name in current_model_params and arg_name not in ['bert_model_name', 'output_dim', 'num_numerical_features']:
            if arg_name == 'cnn_kernel_sizes' and isinstance(arg_value, str):
                current_model_params[arg_name] = [int(x) for x in arg_value.split(',')]
            else:
                current_model_params[arg_name] = arg_value
        elif arg_name in current_trainer_config_params and arg_name not in ['tokenizer_name', 'device']:
            current_trainer_config_params[arg_name] = arg_value
    

    # --- Load and parse configurations ---
    logger.info("--- Loaded and parsed configurations ---")
    logger.info(f"MODEL_PARAMS (effective): {current_model_params}")
    logger.info(f"TRAINER_CONFIG_PARAMS (effective): {current_trainer_config_params}")

    # --- Load Dataset ---
    logger.info(f"Loading dataset from: {args.train_data_path} and {args.test_data_path}")
    data = load_dataset(args.train_data_path, args.test_data_path)

    all_texts_train_full_np = data['train']['texts']
    all_labels_train_full_np = data['train']['labels']
    all_numerical_features_train_full_raw = data['train']['numerical_features']

    texts_test_final_np = data['test']['texts']
    labels_test_final_np = data['test']['labels']
    numerical_features_test_final_raw = data['test']['numerical_features']

    # --- Log dataset information ---
    logger.info(f"MODEL_PARAMS (final after data update): {current_model_params}")
    logger.info(f"Num samples (train): {len(all_texts_train_full_np)}, (test): {len(texts_test_final_np)}")
    logger.info(f"Num numerical features: {current_model_params['num_numerical_features']}")
    logger.info(f"Output dimension: {current_model_params['output_dim']}")

    # --- K-Fold Cross-Validation ---
    skf = StratifiedKFold(n_splits=current_trainer_config_params['n_splits'],
                          shuffle=True,
                          random_state=current_trainer_config_params['seed_base'])
    
    all_fold_val_metrics = [] # To store metrics from each fold's validation set

    logger.info(f"Starting {current_trainer_config_params['n_splits']}-Fold Cross-Validation for RNN Type: {current_model_params['rnn_type'].upper()}...")
    
    for fold_num, (train_index, val_index) in enumerate(skf.split(all_texts_train_full_np, all_labels_train_full_np)):
        logger.info(f"================== FOLD {fold_num + 1}/{current_trainer_config_params['n_splits']} (RNN: {current_model_params['rnn_type'].upper()}) ==================")
        
        metrics_from_fold = run_single_fold(
            fold_id=fold_num + 1,
            train_idx=train_index, val_idx=val_index,
            all_texts=all_texts_train_full_np,
            all_labels=all_labels_train_full_np,
            all_raw_num_feat=all_numerical_features_train_full_raw,
            model_config=current_model_params,
            trainer_config=current_trainer_config_params
        )
        if metrics_from_fold:
            all_fold_val_metrics.append(metrics_from_fold)
        else:
            logger.warning(f"Fold {fold_num + 1} did not return metrics, possibly due to an error.")

    # --- Aggregate K-Fold Validation Results ---
    if not all_fold_val_metrics:
        logger.error("No successful K-Fold runs to aggregate validation metrics from.")
    else:
        df_val_metrics = pd.DataFrame(all_fold_val_metrics)
        logger.info("\n--- Validation Metrics from all successful folds: ---")
        print(df_val_metrics.to_string())

        mean_val_metrics = df_val_metrics.mean()
        std_val_metrics = df_val_metrics.std()

        logger.info("\n--- Aggregated Cross-Validation Metrics (Mean +/- Std on Validation Sets) ---")
        for metric_name in mean_val_metrics.index:
            mean_val = mean_val_metrics[metric_name]
            std_val = std_val_metrics[metric_name]
            logger.info(f"{metric_name.capitalize()}: {mean_val:.4f} +/- {std_val:.4f}")

        cv_summary_path = os.path.join(current_trainer_config_params['output_dir_base'],
                                     f"{current_model_params['rnn_type'].upper()}_cross_validation_summary_metrics.csv")
        results_summary_cv = pd.DataFrame({'mean_validation': mean_val_metrics, 'std_validation': std_val_metrics})
        results_summary_cv.to_csv(cv_summary_path)
        logger.info(f"Aggregated CV metrics saved to {cv_summary_path}")

    # --- Evaluate Final Model on Held-Out Test Set ---
    logger.info(f"\n--- Evaluating final model (RNN: {current_model_params['rnn_type'].upper()}) on the held-out TEST set ---")
    
    all_test_metrics_from_folds_list = []

    # 1. Prepare Scaler for the test data (fit on the full original training data)
    test_set_scaler = None
    if all_numerical_features_train_full_raw is not None and all_numerical_features_train_full_raw.size > 0:
        logger.info("Fitting StandardScaler on the ENTIRE original training data for test set scaling.")
        test_set_scaler = StandardScaler()
        test_set_scaler.fit(all_numerical_features_train_full_raw)
    else:
        logger.warning("No numerical features provided or training data for scaler. Test set will not be scaled if numerical features are present.")

    # 2. Process numerical features of the test set using the common scaler
    numerical_features_test_scaled = None
    if numerical_features_test_final_raw is not None and numerical_features_test_final_raw.size > 0:
        if test_set_scaler is not None:
            numerical_features_test_scaled = test_set_scaler.transform(numerical_features_test_final_raw).tolist()
        else:
            numerical_features_test_scaled = numerical_features_test_final_raw.tolist()
    else:
        numerical_features_test_scaled = None

    # 3. Initialize a reusable Trainer instance for evaluation
    eval_trainer_reusable = DeepLearningTrainer(
        model_name=f"test_{current_model_params['rnn_type']}_eval",
        model_params=current_model_params,
        tokenizer_name=current_trainer_config_params['tokenizer_name'],
        output_dir=os.path.join(current_trainer_config_params['output_dir_base'], "test_evals_each_fold"),
        device=current_trainer_config_params['device'],
        lr=0, num_epochs=0, early_stopping_patience=0
    )
    os.makedirs(eval_trainer_reusable.output_dir, exist_ok=True)

    # 4. Iterate through each fold and evaluate its best model on the test set
    current_rnn_type = current_model_params['rnn_type']

    for fold_num in range(1, current_trainer_config_params['n_splits'] + 1):
        model_path_current_fold = os.path.join(
            current_trainer_config_params['output_dir_base'],
            f"{current_rnn_type}/fold_{fold_num}/{current_rnn_type}_fold_{fold_num}.pth"
        )

        if not os.path.exists(model_path_current_fold):
            logger.warning(f"Model file not found for fold {fold_num} (RNN: {current_rnn_type}) at {model_path_current_fold}. Skipping test evaluation for this fold.")
            all_test_metrics_from_folds_list.append(None)
            continue

        logger.info(f"--- Evaluating model from Fold {fold_num} (RNN: {current_rnn_type}) on Test Set ---")

        try:        
            test_results_current_fold = eval_trainer_reusable.evaluate(
                texts=texts_test_final_np.tolist(),
                labels=labels_test_final_np.tolist(),
                numerical_features=numerical_features_test_scaled,
                model_path=model_path_current_fold,
                stage=f"Test_Eval_RNN_{current_rnn_type}_Fold_{fold_num}"
            )
            if test_results_current_fold and 'metrics' in test_results_current_fold:
                all_test_metrics_from_folds_list.append(test_results_current_fold['metrics'])
                logger.info(f"Test Metrics for Fold {fold_num} (RNN: {current_rnn_type}): {test_results_current_fold['metrics']}")
            else:
                logger.warning(f"Evaluation for Fold {fold_num} (RNN: {current_rnn_type}) did not return valid metrics.")
                all_test_metrics_from_folds_list.append(None)
        except Exception as e_eval_fold:
            logger.error(f"Error evaluating model from Fold {fold_num} (RNN: {current_rnn_type}) on test set: {e_eval_fold}")
            all_test_metrics_from_folds_list.append(None)

    # 5. Calculate Mean and Std of Test Metrics
    valid_test_metrics_agg = [m for m in all_test_metrics_from_folds_list if m is not None]

    if not valid_test_metrics_agg:
        logger.error("No successful test evaluations to aggregate metrics from.")
    else:
        df_test_metrics_agg = pd.DataFrame(valid_test_metrics_agg)
        logger.info(f"\n--- Test Metrics from K-Fold Models ({current_rnn_type.upper()}) ---")
        print(df_test_metrics_agg.to_string())

        mean_test_metrics_agg = df_test_metrics_agg.mean()
        std_test_metrics_agg = df_test_metrics_agg.std()

        logger.info("\n--- Aggregated Test Metrics (Mean +/- Std) ---")
        for metric_name in mean_test_metrics_agg.index:
            mean_val = mean_test_metrics_agg[metric_name]
            std_val = std_test_metrics_agg[metric_name]
            logger.info(f"{metric_name.capitalize()}: {mean_val:.4f} +/- {std_val:.4f}")

        # Save aggregated results to CSV
        test_summary_path = os.path.join(current_trainer_config_params['output_dir_base'],
                                        f"{current_rnn_type.upper()}_test_summary_metrics.csv")
        results_summary_test_df = pd.DataFrame({'mean_test': mean_test_metrics_agg, 'std_test': std_test_metrics_agg})
        results_summary_test_df.to_csv(test_summary_path)
        logger.info(f"Aggregated test metrics saved to: {test_summary_path}")

    logger.info("--- Test Evaluation Completed ---")

if __name__ == "__main__":
    main()

"""
This script is designed to load a trained deep learning model and make predictions on new data.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from deep_learning.utils.config import MODEL_PARAMS, TRAINER_CONFIG_PARAMS
from trainer.dl_trainer import DeepLearningTrainer

logger.add("predict_pipeline.log", rotation="500 MB", level="INFO")
logger.info("Starting the prediction pipeline.")

def predict():
    parser = argparse.ArgumentParser(description="Make predictions using a trained deep learning model.")
    parser.add_argument('--predict_data_path', type=str, required=True,
                        help='Path to the CSV file containing data for prediction.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .pth file of the trained model (e.g., best_dl_model.pth).')
    parser.add_argument('--scaler_path', type=str, 
                        help='Path to the .joblib file of the fitted scaler. Required if numerical features were used.')
    parser.add_argument('--rnn_type', type=str, choices=['rnn', 'lstm', 'gru'], 
                        default=MODEL_PARAMS['rnn_type'],
                        help='RNN type used in the trained model (e.g., "lstm", "gru"). Essential for model reconstruction.')

    args = parser.parse_args()

    # --- Load Configuration ---
    prediction_model_params = MODEL_PARAMS.copy()
    prediction_trainer_config_params = TRAINER_CONFIG_PARAMS.copy()

    # --- Load Data for Prediction ---
    logger.info(f"Loading data for prediction from: {args.predict_data_path}")
    try:
        predict_data_df = pd.read_csv(args.predict_data_path)
        if 'Unnamed: 0' in predict_data_df.columns:
            predict_data_df = predict_data_df.drop('Unnamed: 0', axis=1)

        predict_texts = predict_data_df['text'].values
        predict_labels_dummy = np.zeros(len(predict_texts), dtype=int)
        predict_features_df_raw = predict_data_df.drop(columns=['text', 'label'], errors='ignore') # Drop 'label' if it exists
        predict_numerical_features_raw = predict_features_df_raw.values if not predict_features_df_raw.empty else None

    except Exception as e:
        logger.error(f"Error loading prediction data: {e}")
        return

    # --- Update prediction_model_params based on CLI args/data, ensuring consistency ---
    prediction_model_params['rnn_type'] = args.rnn_type
    
    if predict_numerical_features_raw is not None:
        if prediction_model_params['num_numerical_features'] == 0:
            prediction_model_params['num_numerical_features'] = predict_numerical_features_raw.shape[1]
            logger.warning(f"MODEL_PARAMS['num_numerical_features'] was 0, but prediction data has {predict_numerical_features_raw.shape[1]} features. Updated MODEL_PARAMS dynamically.")
        elif prediction_model_params['num_numerical_features'] != predict_numerical_features_raw.shape[1]:
            logger.error(f"Mismatch: Model expects {prediction_model_params['num_numerical_features']} numerical features, but prediction data has {predict_numerical_features_raw.shape[1]}. Aborting.")
            return
    else:
        if prediction_model_params['num_numerical_features'] > 0:
            logger.error(f"Mismatch: Model expects {prediction_model_params['num_numerical_features']} numerical features, but no numerical features found in prediction data. Aborting.")
            return
        prediction_model_params['num_numerical_features'] = 0
        prediction_model_params['feature_integration_method'] = 'none'

    # --- Prepare Scaler for Prediction ---
    loaded_scaler = None
    if prediction_model_params['num_numerical_features'] > 0:
        if args.scaler_path and os.path.exists(args.scaler_path):
            try:
                loaded_scaler = joblib.load(args.scaler_path)
                logger.info(f"Loaded scaler from {args.scaler_path}")
            except Exception as e_scaler:
                logger.error(f"Error loading scaler from {args.scaler_path}: {e_scaler}. Cannot scale numerical features.")
                loaded_scaler = None
        else:
            logger.warning(f"Scaler path not provided or scaler file not found at {args.scaler_path}. Numerical features will not be scaled. This may lead to incorrect predictions.")
    
    numerical_features_scaled_for_prediction = None
    if predict_numerical_features_raw is not None and prediction_model_params['num_numerical_features'] > 0:
        if loaded_scaler is not None:
            numerical_features_scaled_for_prediction = loaded_scaler.transform(predict_numerical_features_raw).tolist()
        else:
            numerical_features_scaled_for_prediction = predict_numerical_features_raw.tolist() # Use raw if no scaler

    # --- Initialize DeepLearningTrainer for Prediction ---
    predict_output_dir = os.path.join(prediction_trainer_config_params['output_dir_base'], "predictions")
    os.makedirs(predict_output_dir, exist_ok=True)

    try:
        prediction_trainer = DeepLearningTrainer(
            model_name="prediction_model",
            model_params=prediction_model_params,
            tokenizer_name=prediction_trainer_config_params['tokenizer_name'],
            output_dir=predict_output_dir,
            device=prediction_trainer_config_params['device'],
            lr=0, num_epochs=1, batch_size=prediction_trainer_config_params['batch_size'],
            early_stopping_patience=0, metric_for_best_model='f1'
        )
    except Exception as e_trainer_init:
        logger.error(f"Error initializing DeepLearningTrainer for prediction: {e_trainer_init}")
        return

    # --- Make Predictions ---
    logger.info(f"Making predictions using model from: {args.model_path}")
    try:
        predicted_labels, probabilities, logits = prediction_trainer.predict(
            texts=predict_texts.tolist(),
            numerical_features=numerical_features_scaled_for_prediction,
            model_path=args.model_path # Path to the model .pth file
        )

        # --- Output Results ---
        results_df = pd.DataFrame({
            'text': predict_texts,
            'predicted_label': predicted_labels,
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1],
            'raw_logits_0': logits[:, 0],
            'raw_logits_1': logits[:, 1]
        })
        
        # Save predictions to CSV
        output_csv_path = os.path.join(predict_output_dir, "predictions.csv")
        results_df.to_csv(output_csv_path, index=False)
        logger.info(f"Predictions saved to: {output_csv_path}")
        logger.info("\n--- Sample Predictions ---")
        print(results_df.head())

    except Exception as e_predict:
        logger.error(f"Error during prediction: {e_predict}")

    logger.info("--- Prediction pipeline completed ---")

if __name__ == "__main__":
    predict()

"""
Utility function to load dataset from CSV files for training and testing.
This function reads the training and testing data from specified CSV files,
prepares the text and label data, and extracts numerical features if available.
It returns a dictionary containing the processed data ready for model training and evaluation.
"""

from typing import *
import pandas as pd
from deep_learning.utils.config import MODEL_PARAMS

def load_dataset(
        train_data_path: str = "~/nlp-project/dataset/stratify_train.csv",
        test_data_path: str = "~/nlp-project/dataset/stratify_test.csv"
    ) -> Dict[str, Any]:
    """
    Load dataset from CSV files and prepare the data for training and testing.
    Args:
        train_data_path (str): Path to the training data CSV file.
        test_data_path (str): Path to the testing data CSV file.
    Returns:
        Dict[str, Any]: A dictionary containing training and testing data.
    """

    train_data_df = pd.read_csv(train_data_path)
    if 'Unnamed: 0' in train_data_df.columns:
        train_data_df = train_data_df.drop('Unnamed: 0', axis=1)

    test_data_df = pd.read_csv(test_data_path)
    if 'Unnamed: 0' in test_data_df.columns:
        test_data_df = test_data_df.drop('Unnamed: 0', axis=1)

    texts_train = train_data_df['text'].values
    labels_train = train_data_df['label'].values

    features_train_df = train_data_df.drop(columns=['text', 'label'], errors='ignore')
    if not features_train_df.empty:
        numerical_features_train_raw = features_train_df.values
        MODEL_PARAMS['num_numerical_features'] = numerical_features_train_raw.shape[1]
    else:
        numerical_features_train_raw = None
        MODEL_PARAMS['num_numerical_features'] = 0

    if MODEL_PARAMS['num_numerical_features'] == 0:
        MODEL_PARAMS['feature_integration_method'] = 'none'

    texts_test = test_data_df['text'].values
    labels_test = test_data_df['label'].values
    features_test_df = test_data_df.drop(columns=['text', 'label'], errors='ignore')
    if not features_test_df.empty:
        numerical_features_test_raw = features_test_df.values
    else:
        numerical_features_test_raw = None

    data_dict = {
        "train": {
            "texts": texts_train,
            "labels": labels_train,
            "numerical_features": numerical_features_train_raw
        },
        "test": {
            "texts": texts_test,
            "labels": labels_test,
            "numerical_features": numerical_features_test_raw
        }
    }

    return data_dict

"""
This module defines a custom PyTorch Dataset for handling fake news data with additional numerical features.
It allows for the integration of both text data and numerical features, suitable for training models that require both types of inputs.
"""

from typing import *
import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
    """
    A custom dataset for fake news classification that includes both text and numerical features.
    This dataset is designed to work with a tokenizer for text processing and can handle additional numerical features.
    Args:
        texts (list of str): List of text samples.
        numerical_features (list of list of float): List of numerical feature vectors corresponding to each text sample.
        labels (list of int): List of labels corresponding to each text sample.
        tokenizer: A tokenizer instance for processing text data.
        max_len (int): Maximum length for tokenized sequences.
    """
    def __init__(self, texts: List[str], numerical_features: Optional[List[List[float]]] = None,
                 labels: List[int] = None, tokenizer=None, max_len: int = 512):
        if labels is None:
            raise ValueError("Labels must be provided for the dataset.")
        if tokenizer is None:
            raise ValueError("A tokenizer must be provided for the dataset.")
        
        self.texts = texts
        if numerical_features is not None:
            self.numerical_features = torch.FloatTensor(numerical_features)
        else:
            self.numerical_features = None
        self.labels = torch.LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }
        if self.numerical_features is not None:
            item['numerical_features'] = self.numerical_features[idx]
        return item

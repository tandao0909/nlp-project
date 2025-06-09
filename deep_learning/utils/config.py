"""
Configuration parameters for the model and trainer.
This module defines the parameters required for the model architecture and training process.
"""

import torch

MODEL_PARAMS = {
    'bert_model_name': 'bert-base-uncased',
    'cnn_out_channels_per_kernel': 32,
    'cnn_kernel_sizes': [3, 5, 7],
    'cnn_dropout_prob': 0.2,
    'rnn_hidden_dim': 50,
    'n_rnn_layers': 1,
    'rnn_dropout_prob': 0.2,
    'rnn_type': 'lstm',
    'fc_dropout_prob': 0.2,
    'output_dim': 2,
    'num_numerical_features': 0, # update later
    'feature_integration_method': 'concat',
    'freeze_bert_completely': True
}


TRAINER_CONFIG_PARAMS = {
    'tokenizer_name': 'bert-base-uncased',
    'lr': 1e-4,
    'num_epochs': 100,
    'batch_size': 256,
    'max_len': 256,
    'early_stopping_patience': 3,
    'metric_for_best_model': 'f1',
    'n_splits': 5,
    'seed_base': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir_base': "~/nlp-project/deep_learning/output",
    'optimizer_name': 'adamw',
    'lr_scheduler_name': 'linear_warmup',
    'warmup_steps_ratio': 0.001,
    'criterion_name': 'cross_entropy'
}

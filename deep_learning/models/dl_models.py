"""
This module defines a deep learning model that integrates BERT with CNN and RNN layers.
It allows for the processing of text data with additional numerical features, suitable for tasks like fake news classification.
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class DeepLearningModels(nn.Module):
    """
    A deep learning model that integrates BERT with CNN and RNN layers.
    This model is designed for tasks such as fake news classification, where both text and numerical features are utilized.
    Args:
        bert_model_name (str): Name of the pre-trained BERT model to use.
        cnn_out_channels_per_kernel (int): Number of output channels for each CNN kernel.
        cnn_kernel_sizes (List[int]): List of kernel sizes for the CNN layers.
        cnn_dropout_prob (float): Dropout probability for the CNN layers.
        rnn_hidden_dim (int): Hidden dimension for the RNN layer.
        n_rnn_layers (int): Number of layers in the RNN.
        rnn_dropout_prob (float): Dropout probability for the RNN layer.
        rnn_type (str): Type of RNN to use ('rnn', 'lstm', or 'gru').
        fc_dropout_prob (float): Dropout probability for the fully connected layer.
        output_dim (int): Output dimension of the final layer (e.g., number of classes).
        num_numerical_features (int): Number of numerical features to integrate with the text data.
        feature_integration_method (str): Method to integrate numerical features ('concat' or 'add').
        freeze_bert_completely (bool): Whether to freeze all BERT parameters.
        freeze_bert_embeddings_only (bool): Whether to freeze only the BERT embedding parameters.
        unfreeze_last_n_bert_layers (int): Number of last BERT layers to unfreeze for training.
    """
    def __init__(self, bert_model_name: str = 'bert-base-uncased',
                 cnn_out_channels_per_kernel: int = 128,
                 cnn_kernel_sizes: List[int] = [3, 5, 7],
                 cnn_dropout_prob: float = 0.1,
                 rnn_hidden_dim: int = 128, n_rnn_layers: int = 1,
                 rnn_dropout_prob: float = 0.1, rnn_type: str = 'lstm',
                 fc_dropout_prob: float = 0.1,
                 output_dim: int = 2,
                 num_numerical_features: int = 0, 
                 feature_integration_method: str = 'concat',
                 freeze_bert_completely: bool = True,
                 freeze_bert_embeddings_only: bool = False,
                 unfreeze_last_n_bert_layers: int = 0
                ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.num_numerical_features = num_numerical_features
        self.feature_integration_method = feature_integration_method.lower()

        # BERT Freezing
        if freeze_bert_completely:
            print("Freezing all BERT parameters.")
            for param in self.bert.parameters():
                param.requires_grad = False
        elif freeze_bert_embeddings_only:
            print("Freezing only BERT embedding parameters.")
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        elif unfreeze_last_n_bert_layers > 0:
            print(f"Freezing BERT except for the last {unfreeze_last_n_bert_layers} encoder layers and pooler.")
            for param in self.bert.parameters():
                param.requires_grad = False
            if self.bert.encoder.layer is not None:
                for i in range(unfreeze_last_n_bert_layers):
                    if len(self.bert.encoder.layer) > i:
                        for param in self.bert.encoder.layer[-(i + 1)].parameters():
                            param.requires_grad = True
            if self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
        else:
            print("Fine-tuning all BERT parameters.")

        bert_output_dim = self.bert.config.hidden_size # Output dimension of BERT, typically 768 for BERT-base

        # CNN Layers (Trainable)
        self.convs = nn.ModuleList()
        for k_size in cnn_kernel_sizes:
            padding_val = (k_size - 1) // 2
            self.convs.append(
                nn.Conv1d(in_channels=bert_output_dim,
                          out_channels=cnn_out_channels_per_kernel,
                          kernel_size=k_size,
                          padding=padding_val)
            )
        self.cnn_dropout = nn.Dropout(cnn_dropout_prob)
        rnn_input_dim = cnn_out_channels_per_kernel * len(cnn_kernel_sizes)

        # RNN Layer (Trainable)
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(rnn_input_dim,
                              rnn_hidden_dim,
                              num_layers=n_rnn_layers,
                              bidirectional=True,
                              dropout=rnn_dropout_prob if n_rnn_layers > 1 else 0,
                              batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim,
                               rnn_hidden_dim,
                               num_layers=n_rnn_layers,
                               bidirectional=True,
                               dropout=rnn_dropout_prob if n_rnn_layers > 1 else 0,
                               batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(rnn_input_dim,
                              rnn_hidden_dim,
                              num_layers=n_rnn_layers,
                              bidirectional=True,
                              dropout=rnn_dropout_prob if n_rnn_layers > 1 else 0,
                              batch_first=True)
        else:
            raise ValueError("rnn_type must be 'rnn', 'lstm' or 'gru'")

        # Fully Connected Layer (Trainable)
        self.fc_dropout = nn.Dropout(fc_dropout_prob)
        fc_input_dim = rnn_hidden_dim * 2  # Bidirectional RNN
        if self.feature_integration_method == 'concat' and self.num_numerical_features > 0:
            fc_input_dim += self.num_numerical_features
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                numerical_features: Optional[torch.Tensor] = None):
        # 1. BERT Layer
        if self.bert.training:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output shape: (batch_size, seq_len, bert_hidden_size)
        bert_sequence_output = bert_outputs.last_hidden_state

        # 2. CNN Layers (Trainable)
        # Permute to satisfy Conv1D: (batch_size, bert_hidden_size, seq_len)
        cnn_input = bert_sequence_output.permute(0, 2, 1)
        # conved_outputs[i] shape: (batch_size, cnn_out_channels_per_kernel, seq_len)
        conved_outputs = [F.relu(conv(cnn_input)) for conv in self.convs]
        # Concatenate feature maps from all CNN layers
        # cnn_processed_sequence shape: (batch_size, cnn_out_channels_per_kernel * len(kernels), seq_len)
        cnn_processed_sequence = torch.cat(conved_outputs, dim=1)
        # Dropout sau CNN
        cnn_processed_sequence_dropout = self.cnn_dropout(cnn_processed_sequence)
        # Permute to prepare for RNN: (batch_size, seq_len, cnn_out_channels_per_kernel * len(kernels))
        rnn_input = cnn_processed_sequence_dropout.permute(0, 2, 1)

        # 3. RNN Layer (Trainable)
        if self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(rnn_input)
        else: # rnn, gru
            _, hidden = self.rnn(rnn_input)
        # Check if using bidirectional RNN, and concatenate the last hidden states
        if self.rnn.bidirectional:
            # hidden shape: (n_layers * n_directions, batch_size, rnn_hidden_dim)
            rnn_representation = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # For unidirectional RNN, we only take the last layer's hidden state
            # hidden[-1, :, :] shape: (batch_size, rnn_hidden_dim)
            rnn_representation = hidden[-1, :, :]

        rnn_representation_dropout = self.fc_dropout(rnn_representation)

        # 4. Integrate numerical features & FC Layer (Trainable)
        if self.feature_integration_method == 'concat' and self.num_numerical_features > 0 and numerical_features is not None:
            final_input_to_fc = torch.cat((rnn_representation_dropout, numerical_features), dim=1)
        else:
            final_input_to_fc = rnn_representation_dropout

        logits = self.fc(final_input_to_fc)
        return logits
    
"""
Deep Learning Trainer for Fake News Detection
This module provides a comprehensive trainer class for deep learning models,
specifically designed for fake news detection tasks. It supports various model architectures, optimizers, learning rate schedulers,
and evaluation metrics. The trainer can handle both text and numerical features, making it suitable for multi-modal inputs.
"""

import os
import json
import time
from typing import *
from loguru import logger
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AutoTokenizer
)
from deep_learning.models.dl_models import DeepLearningModels
from deep_learning.utils.fake_news_dataset import FakeNewsDataset


class DeepLearningTrainer:
    """
    A comprehensive trainer class for deep learning models, specifically designed for fake news detection tasks.
    This class supports various model architectures, optimizers, learning rate schedulers, and evaluation metrics.
    It can handle both text and numerical features, making it suitable for multi-modal inputs.
    Args:
        model_params (Dict[str, Any]): Parameters to initialize DeepLearningModels.
                                        Example: {'bert_model_name': 'bert-base-uncased', ...}
        tokenizer_name (str): Name or path of the tokenizer (e.g., 'bert-base-uncased').
        optimizer_name (str): Name of the optimizer ('adamw', 'adam').
        lr (float): Learning rate.
        lr_scheduler_name (Optional[str]): Name of the learning rate scheduler ('linear_warmup', 'cosine_warmup').
        warmup_steps_ratio (float): Ratio of total training steps for warmup.
        criterion_name (str): Name of the loss function.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        max_len (int): Maximum sequence length for tokenizer.
        early_stopping_patience (int): Patience for early stopping.
        metric_for_best_model (str): Metric to monitor for saving the best model and early stopping.
                                     Options: 'val_loss', 'accuracy', 'precision', 'recall', 'f1'.
        device (Optional[str]): Device to use ('cuda', 'cpu'). Autodetects if None.
        output_dir (str): Directory to save models, logs, and plots.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self,
                 model_name: str,
                 model_params: Dict[str, Any],
                 tokenizer_name: str,
                 optimizer_name: str = 'adamw',
                 lr: float = 2e-5,
                 lr_scheduler_name: Optional[str] = 'linear_warmup',
                 warmup_steps_ratio: float = 0.1, # Tỷ lệ warmup steps so với total_steps
                 criterion_name: str = 'cross_entropy',
                 num_epochs: int = 10,
                 batch_size: int = 16,
                 max_len: int = 128,
                 early_stopping_patience: int = 3,
                 metric_for_best_model: str = 'f1',
                 device: Optional[str] = None,
                 output_dir: str = "dl_outputs",
                 seed: int = 42):

        self.model_name = model_name
        self.model_params = model_params
        self.tokenizer_name = tokenizer_name
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.lr_scheduler_name = lr_scheduler_name.lower() if lr_scheduler_name else None
        self.warmup_steps_ratio = warmup_steps_ratio
        self.criterion_name = criterion_name.lower()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.early_stopping_patience = early_stopping_patience
        self.metric_for_best_model = metric_for_best_model.lower()
        self.output_dir = output_dir
        self.seed = seed
        self.current_epoch = 0

        self._set_seed(self.seed)
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")

        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': []
        }
        self.best_metric_value = -float('inf') if 'loss' not in self.metric_for_best_model else float('inf')
        self.epochs_no_improve = 0
        self.best_epoch = 0

        os.makedirs(self.output_dir, exist_ok=True)
        logger.add(os.path.join(self.output_dir, f"{self.model_name}.log"), rotation="500 MB")

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_device(self, device_str: Optional[str]) -> torch.device:
        if device_str:
            return torch.device(device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_tokenizer(self):
        logger.info(f"Initializing tokenizer: {self.tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            return tokenizer
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise

    def _initialize_model(self) -> DeepLearningModels:
        logger.info(f"Initializing model with params: {self.model_params}")
        try:
            model = DeepLearningModels(**self.model_params)
            model.to(self.device)
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model initialized. Total parameters: {total_params}. Trainable parameters: {num_trainable_params}")
            return model
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def _initialize_optimizer(self, model_parameters) -> torch.optim.Optimizer:
        logger.info(f"Initializing optimizer: {self.optimizer_name} with LR: {self.lr}")
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(model_parameters, lr=self.lr, eps=1e-8)
        elif self.optimizer_name == 'adam':
            return torch.optim.Adam(model_parameters, lr=self.lr)
        else:
            logger.error(f"Unsupported optimizer: {self.optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _initialize_scheduler(self, optimizer, num_training_steps: int):
        if self.lr_scheduler_name == 'linear_warmup':
            num_warmup_steps = int(num_training_steps * self.warmup_steps_ratio)
            logger.info(f"Initializing Linear Warmup scheduler. Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
            return get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=num_warmup_steps,
                                                 num_training_steps=num_training_steps)
        elif self.lr_scheduler_name == 'cosine_warmup':
            num_warmup_steps = int(num_training_steps * self.warmup_steps_ratio)
            logger.info(f"Initializing Cosine Warmup scheduler. Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
            return get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)
        elif self.lr_scheduler_name is None:
            logger.info("No learning rate scheduler will be used.")
            return None
        else:
            logger.error(f"Unsupported LR scheduler: {self.lr_scheduler_name}. Available options: 'linear_warmup', 'cosine_warmup', None.")
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler_name}. Available options: 'linear_warmup', 'cosine_warmup', None.")

    def _initialize_criterion(self) -> nn.Module:
        logger.info(f"Initializing criterion: {self.criterion_name}")
        if self.criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.criterion_name == 'bce_with_logits':
            return nn.BCEWithLogitsLoss()
        else:
            logger.error(f"Unsupported criterion: {self.criterion_name}. Available options: 'cross_entropy', 'bce_with_logits'.")
            raise ValueError(f"Unsupported criterion: {self.criterion_name}. Available options: 'cross_entropy', 'bce_with_logits'.")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return metrics

    def _train_one_epoch(self, train_loader: DataLoader,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            numerical_features = batch.get('numerical_features')
            if numerical_features is not None:
                numerical_features = numerical_features.to(self.device)

            logits = self.model(input_ids, attention_mask, numerical_features)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if batch_idx % (len(train_loader) // 10 + 1) == 0 :
                 logger.info(f"Epoch {self.current_epoch+1} Batch {batch_idx+1}/{len(train_loader)} Train Loss: {loss.item():.4f}")


        avg_loss = total_loss / len(train_loader)
        metrics = self._compute_metrics(np.array(all_labels), np.array(all_preds))
        return avg_loss, metrics

    def _evaluate_one_epoch(self, data_loader: DataLoader, stage: str = "Validation"):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                numerical_features = batch.get('numerical_features')
                if numerical_features is not None:
                    numerical_features = numerical_features.to(self.device)

                logits = self.model(input_ids, attention_mask, numerical_features)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if batch_idx % (len(data_loader) // 10 + 1) == 0 : # Log progress
                    logger.info(f"Epoch {self.current_epoch+1} Batch {batch_idx+1}/{len(data_loader)} {stage} Loss: {loss.item():.4f}")


        avg_loss = total_loss / len(data_loader)
        metrics = self._compute_metrics(np.array(all_labels), np.array(all_preds))
        return avg_loss, metrics, np.array(all_labels), np.array(all_preds)
    
    def fit(self, train_texts: List[str], train_labels: List[int], 
            train_numerical_features: Optional[List[List[float]]] = None,
            val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None, 
            val_numerical_features: Optional[List[List[float]]] = None):
        """
        Trains the model on the provided training data and evaluates on validation data if available.
        Args:
            train_texts (List[str]): List of training text samples.
            train_labels (List[int]): List of training labels.
            train_numerical_features (Optional[List[List[float]]]): List of numerical feature vectors for training samples.
            val_texts (Optional[List[str]]): List of validation text samples.
            val_labels (Optional[List[int]]): List of validation labels.
            val_numerical_features (Optional[List[List[float]]]): List of numerical feature vectors for validation samples.
        Returns:
            Dict[str, List[float]]: Training history containing loss and metrics for both training and validation sets.
        """
        logger.info("Starting training process...")

        train_dataset = FakeNewsDataset(
            texts=train_texts,
            labels=train_labels,
            numerical_features=train_numerical_features,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        val_loader = None
        if val_texts and val_labels:
            val_dataset = FakeNewsDataset(
                texts=val_texts,
                labels=val_labels,
                numerical_features=val_numerical_features,
                tokenizer=self.tokenizer,
                max_len=self.max_len
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        else:
            logger.warning("No validation data provided. Early stopping and best model saving based on validation set will be disabled.")


        # Optimizer and Scheduler
        self.optimizer = self._initialize_optimizer(self.model.parameters())
        num_training_steps = len(train_loader) * self.num_epochs
        scheduler = self._initialize_scheduler(self.optimizer, num_training_steps)

        start_time_total = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            logger.info(f"--- Epoch {epoch + 1}/{self.num_epochs} ---")

            train_loss, train_metrics = self._train_one_epoch(train_loader, self.optimizer, scheduler)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['train_f1'].append(train_metrics['f1'])

            logger.info(f"Epoch {epoch + 1} Train: Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.4f}, "
                        f"Precision={train_metrics['precision']:.4f}, Recall={train_metrics['recall']:.4f}, F1={train_metrics['f1']:.4f}")

            current_metric_for_saving = train_metrics[self.metric_for_best_model] if 'loss' not in self.metric_for_best_model else train_loss
            stage_for_saving = "train"

            if val_loader:
                val_loss, val_metrics, _, _ = self._evaluate_one_epoch(val_loader, stage="Validation")
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_recall'].append(val_metrics['recall'])
                self.history['val_f1'].append(val_metrics['f1'])

                logger.info(f"Epoch {epoch + 1} Val: Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                            f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")

                current_metric_for_saving = val_metrics[self.metric_for_best_model] if 'loss' not in self.metric_for_best_model else val_loss
                stage_for_saving = "val"


            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds")

            # Early Stopping and Best Model Saving
            improved = False
            if 'loss' in self.metric_for_best_model: # Lower is better for loss
                if current_metric_for_saving < self.best_metric_value:
                    self.best_metric_value = current_metric_for_saving
                    improved = True
            else: # Higher is better for other metrics
                if current_metric_for_saving > self.best_metric_value:
                    self.best_metric_value = current_metric_for_saving
                    improved = True

            if improved:
                logger.info(f"Epoch {epoch + 1}: {self.metric_for_best_model} ({stage_for_saving}) improved to {self.best_metric_value:.4f}. Saving model...")
                self.save_model(filename=f"{self.model_name}.pth")
                self.epochs_no_improve = 0
                self.best_epoch = epoch + 1
            elif val_loader: # Only apply early stopping if there's a validation set
                self.epochs_no_improve += 1
                logger.info(f"Epoch {epoch + 1}: {self.metric_for_best_model} ({stage_for_saving}) did not improve from {self.best_metric_value:.4f}. "
                            f"Early stopping counter: {self.epochs_no_improve}/{self.early_stopping_patience}")
                if self.epochs_no_improve >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            else: # No validation, just save last model if no improvement logic is based on train
                if epoch == self.num_epochs -1: # Save last model if no validation
                     self.save_model(filename="last_dl_model.pth")


        total_training_time = time.time() - start_time_total
        logger.info(f"Training finished. Total time: {total_training_time:.2f} seconds.")
        logger.info(f"Best model achieved at epoch {self.best_epoch} with {self.metric_for_best_model}: {self.best_metric_value:.4f}")
        self.plot_training_history()
        return self.history

    def evaluate(self,
                 texts: List[str], labels: List[int], numerical_features: Optional[List[List[float]]] = None,
                 model_path: Optional[str] = None, stage: str = "Test") -> Dict[str, Any]:
        """
        Evaluates the model on the provided test data.
        Args:
            texts (List[str]): List of text samples for evaluation.
            labels (List[int]): List of true labels for the text samples.
            numerical_features (Optional[List[List[float]]]): List of numerical feature vectors for the samples.
            model_path (Optional[str]): Path to a specific model file to load for evaluation.
            stage (str): Stage of evaluation, e.g., "Test", "Validation". Used for logging and saving results.
        Returns:
            Dict[str, Any]: Evaluation results including loss, metrics, classification report, and confusion matrix.
        """
        if model_path:
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path} for {stage} evaluation.")
        elif os.path.exists(os.path.join(self.output_dir, f"{self.model_name}.pth")):
            self.load_model(os.path.join(self.output_dir, f"{self.model_name}.pth"))
            logger.info(f"Loaded best model for {stage} evaluation.")
        else:
            logger.warning(f"No specific model_path provided and f{self.model_name}.pth not found. Evaluating with the current model state.")

        dataset = FakeNewsDataset(
            texts=texts,
            labels=labels,
            numerical_features=numerical_features,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        logger.info(f"Starting {stage} evaluation...")
        avg_loss, metrics, y_true, y_pred = self._evaluate_one_epoch(data_loader, stage=stage)

        logger.info(f"{stage} Results: Loss={avg_loss:.4f}, Acc={metrics['accuracy']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

        report = classification_report(y_true, y_pred, zero_division=0)
        logger.info(f"\n{stage} Classification Report:\n{report}")

        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, class_names=[str(i) for i in range(self.model_params.get('output_dim', 2))],
                                   title=f"{stage} Confusion Matrix")

        results = {
            'loss': avg_loss,
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist() # Convert to list for easier saving (e.g. JSON)
        }
        # Save test results
        with open(os.path.join(self.output_dir, f"{stage.lower()}_results.json"), 'w') as f:
            json.dump(results, f, indent=4)

        return results

    def predict(self, texts: List[str], numerical_features: Optional[List[List[float]]] = None,
                model_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts labels for the provided texts using the trained model.
        Args:
            texts (List[str]): List of text samples for prediction.
            numerical_features (Optional[List[List[float]]]): List of numerical feature vectors for the samples.
            model_path (Optional[str]): Path to a specific model file to load for prediction.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predicted labels, probabilities, and logits.
            - Predicted labels: Array of predicted class indices.
            - Probabilities: Softmax probabilities for each class.
            - Logits: Raw model outputs before softmax.
        """
        if model_path:
            self.load_model(model_path)
        elif os.path.exists(os.path.join(self.output_dir, f"{self.model_name}.pth")):
            self.load_model(os.path.join(self.output_dir, f"{self.model_name}.pth"))
        else:
            logger.warning(f"No model_path provided and {self.model_name}.pth not found. Predicting with current model state.")

        self.model.eval()
        # Create a dummy dataset for prediction (labels are not used)
        dummy_labels = [0] * len(texts) # Dummy labels
        dataset = FakeNewsDataset(
            texts=texts,
            labels=dummy_labels, # Not used for prediction but required by Dataset
            numerical_features=numerical_features,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_logits_list = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                num_feats = batch.get('numerical_features')
                if num_feats is not None:
                    num_feats = num_feats.to(self.device)
                logits = self.model(input_ids, attention_mask, num_feats)
                all_logits_list.append(logits.cpu().numpy())

        all_logits = np.concatenate(all_logits_list, axis=0)
        probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
        predicted_labels = np.argmax(probabilities, axis=1)
        return predicted_labels, probabilities, all_logits


    def save_model(self, filename: str = "model.pth"):
        """
        Saves the model's state_dict, optimizer state, and other relevant information to a file.
        Args:
            filename (str): Name of the file to save the model. Default is "model.pth".
        """
        path = os.path.join(self.output_dir, filename)
        try:
            checkpoint = {
                'epoch': self.current_epoch + 1 if hasattr(self, 'current_epoch') else 0,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                'best_metric_value': self.best_metric_value,
                'model_params': self.model_params, # Save model config
                'tokenizer_name': self.tokenizer_name,
            }
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")
            raise

    def load_model(self, model_path: str):
        """
        Loads the model's state_dict, optimizer state, and other relevant information from a file.
        Args:
            model_path (str): Path to the model file to load.
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = self._initialize_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.best_metric_value = checkpoint.get('best_metric_value', self.best_metric_value)
            self.epoch = checkpoint.get('epoch', 0)
            self.optimizer = checkpoint.get('optimizer_state_dict', None)
            self.model_params = checkpoint.get('model_params', self.model_params)
            self.tokenizer_name = checkpoint.get('tokenizer_name', self.tokenizer_name)
            logger.info(f"Model loaded successfully from {model_path}. Previous best metric: {checkpoint.get('best_metric_value')}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def plot_training_history(self):
        if not self.history['train_loss']:
            logger.warning("No training history to plot.")
            return

        epochs_ran = range(1, len(self.history['train_loss']) + 1)
        plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style

        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('Training and Validation History', fontsize=16)

        # Loss
        axs[0, 0].plot(epochs_ran, self.history['train_loss'], 'o-', label='Training Loss', color='royalblue')
        if self.history['val_loss']:
            axs[0, 0].plot(epochs_ran, self.history['val_loss'], 'o-', label='Validation Loss', color='orangered')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        # Accuracy
        axs[0, 1].plot(epochs_ran, self.history['train_accuracy'], 'o-', label='Training Accuracy', color='royalblue')
        if self.history['val_accuracy']:
            axs[0, 1].plot(epochs_ran, self.history['val_accuracy'], 'o-', label='Validation Accuracy', color='orangered')
        axs[0, 1].set_title('Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()

        # F1-score (Macro)
        axs[1, 0].plot(epochs_ran, self.history['train_f1'], 'o-', label='Training F1 (Macro)', color='royalblue')
        if self.history['val_f1']:
            axs[1, 0].plot(epochs_ran, self.history['val_f1'], 'o-', label='Validation F1 (Macro)', color='orangered')
        axs[1, 0].set_title('F1 Score (Macro)')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('F1 Score')
        axs[1, 0].legend()

        # Precision (Macro)
        axs[1, 1].plot(epochs_ran, self.history['train_precision'], 'o-', label='Training Precision (Macro)', color='royalblue')
        if self.history['val_precision']:
            axs[1, 1].plot(epochs_ran, self.history['val_precision'], 'o-', label='Validation Precision (Macro)', color='orangered')
        axs[1, 1].set_title('Precision (Macro)')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].legend()

        # Recall (Macro)
        axs[2, 0].plot(epochs_ran, self.history['train_recall'], 'o-', label='Training Recall (Macro)', color='royalblue')
        if self.history['val_recall']:
            axs[2, 0].plot(epochs_ran, self.history['val_recall'], 'o-', label='Validation Recall (Macro)', color='orangered')
        axs[2, 0].set_title('Recall (Macro)')
        axs[2, 0].set_xlabel('Epoch')
        axs[2, 0].set_ylabel('Recall')
        axs[2, 0].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "training_history.png")
        plt.savefig(plot_path)
        logger.info(f"Training history plot saved to {plot_path}")
        plt.show()

    def plot_confusion_matrix(self, cm, class_names, title='Confusion Matrix'):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path)
        logger.info(f"Confusion matrix plot saved to {cm_path}")
        plt.show()

"""
This module provides a class for evaluating and predicting with LLMs using PyTorch and Hugging Face Transformers.
"""

from typing import *
import numpy as np
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainerCallback,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig
)
from deep_learning.utils.fake_news_dataset import FakeNewsDataset
from llms.utils.metrics import compute_metrics

id2Label = {
    0: "False", # Fake News
    1: "True"   # Real News
}
label2Id = {v: k for k, v in id2Label.items()}


class LLM_EvaluatorAndPredictor:
    """
    A class for evaluating and predicting with LLMs using PyTorch and Hugging Face Transformers.
    This class supports evaluation on datasets and prediction on individual examples.
    It can handle both classification and regression tasks, and it provides detailed reports
    including classification reports and metrics computation.

    Attributes:
        model (torch.nn.Module): The PyTorch model to evaluate and predict with.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        device (torch.device): The device to run the model on (CPU or GPU).

    Methods:
        evaluate(eval_dataset, data_collator=None, compute_metrics_fn=None, batch_size=16, dataset_name_for_report="Dataset"):
            Evaluates the model on the provided dataset and computes metrics if a function is provided.
        predict(inputs_for_prediction):
            Makes predictions on the provided inputs and returns the predicted labels.
    """
    def __init__(self, model: Union[torch.nn.Module, str],
                 tokenizer: Union[AutoTokenizer, str],
                 device: Optional[Union[str, torch.device]] = None):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"ModelEvaluatorAndPredictor initialized on device: {self.device}")

    def evaluate(
        self,
        eval_dataset,
        data_collator=None,
        compute_metrics_fn=None, 
        batch_size=16,
        dataset_name_for_report="Dataset"
    ):
        """
        Evaluates the model on the provided dataset and computes metrics if a function is provided.
        Args:
            eval_dataset (torch.utils.data.Dataset): The dataset to evaluate the model on.
            data_collator (callable, optional): A function to collate data into batches. Defaults to None.
            compute_metrics_fn (callable, optional): A function to compute metrics from predictions and labels. Defaults to None.
            batch_size (int, optional): The batch size for evaluation. Defaults to 16.
            dataset_name_for_report (str, optional): The name of the dataset for reporting purposes. Defaults to "Dataset".
        Returns:
            dict: A dictionary containing the computed metrics.
        """
        print(f"\n--- Starting Evaluation on {dataset_name_for_report} ---")
        
        if data_collator is None:
            print("No data_collator provided, using default DataCollatorWithPadding.")
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False
        )

        all_logits = []
        all_labels = []

        progress_bar_eval = tqdm(eval_dataloader, desc=f"Evaluating {dataset_name_for_report}")
        for batch in progress_bar_eval:
            if "labels" in batch:
                labels = batch.pop("labels").to(self.device)
            elif "label" in batch:
                labels = batch.pop("label").to(self.device)
            else:
                raise ValueError("Batch from DataLoader does not contain 'labels' or 'label' key.")

            inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        final_logits = np.concatenate(all_logits, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        
        metrics = {}
        if compute_metrics_fn:
            class MockEvalPrediction:
                def __init__(self, predictions, label_ids):
                    self.predictions = predictions
                    self.label_ids = label_ids
            
            eval_pred = MockEvalPrediction(predictions=final_logits, label_ids=final_labels)
            print("Computing metrics...")
            metrics = compute_metrics_fn(eval_pred)
            print(f"Evaluation Metrics for {dataset_name_for_report}:")
            for k, v in metrics.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for class_k, class_v in v.items():
                        print(f"    {class_k}: {class_v:.4f}")
                elif isinstance(v, (float, np.float_)):
                     print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print("No compute_metrics_fn provided. Skipping custom metrics computation.")

        predictions_for_report = np.argmax(final_logits, axis=1)
        target_names_report = None
        num_unique_labels_in_data = len(np.unique(final_labels))
        target_names_report = [id2Label[i] for i in sorted(id2Label.keys()) if i < num_unique_labels_in_data]
        
        if target_names_report and len(target_names_report) == num_unique_labels_in_data :
            print(f"\nClassification Report for {dataset_name_for_report}:")
            print(classification_report(final_labels, predictions_for_report, target_names=target_names_report, digits=4, zero_division=0))
        else:
            print(f"\nCould not generate detailed classification report for {dataset_name_for_report} (label mapping or class count mismatch).")
            print(f"Unique labels in data: {np.unique(final_labels)}")


        return metrics

    def predict(self, inputs_for_prediction):
        """
        Makes predictions on the provided inputs and returns the predicted labels.
        Args:
            inputs_for_prediction (str or list of str): The input text(s) to predict labels for.
        Returns:
            list: A list of dictionaries containing the input text, predicted label ID, and predicted label name.
        """
        print(f"\n--- Starting Prediction for examples ---")
        self.model.eval()

        results = []
        current_max_length = 512
        label_map = id2Label

        if not isinstance(inputs_for_prediction, list):
            inputs_for_prediction = [inputs_for_prediction]

        for i, example_input in enumerate(inputs_for_prediction):
            tokenized_input = None
            original_input_display = ""

            if example_input:
                sentence = example_input
                original_input_display = f"Sentence: '{sentence}'"
                tokenized_input = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=current_max_length
                )
            else:
                results.append({"input": str(example_input), "error": "Invalid VI input format"})
                continue
                
            inputs_to_model = {k: v.to(self.device) for k, v in tokenized_input.items()}
            with torch.no_grad():
                outputs = self.model(**inputs_to_model)
                logits = outputs.logits
                predicted_class_id = torch.argmax(logits, dim=-1).item()
            
            predicted_label_name = label_map.get(predicted_class_id, "Unknown")
            
            print(f"\nExample {i+1}: {original_input_display}")
            print(f"  Predicted ID: {predicted_class_id}, Label: {predicted_label_name}\n")
            results.append({
                "input": original_input_display,
                "predicted_id": predicted_class_id,
                "predicted_label": predicted_label_name
            })
        
        return results



class LLM_Trainer:
    """
    A class for training and evaluating LLMs using PyTorch and Hugging Face Transformers.
    This class supports training with LoRA (Low-Rank Adaptation) and provides methods for training,
    evaluating, and predicting with the model. It handles datasets for training, validation, and testing,
    and it can be configured with various training parameters.

    Attributes:
        model_name (str): The name of the pre-trained model to use.
        train_dataset (FakeNewsDataset): The dataset for training.
        val_dataset (FakeNewsDataset): The dataset for validation.
        test_dataset (FakeNewsDataset): The dataset for testing.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForSequenceClassification): The model for sequence classification.
        data_collator (DataCollatorWithPadding): The data collator for padding sequences.

    Methods:
        train(output_dir, lora_kwargs=None, model_kwargs=None):
            Trains the model with the provided configurations and saves it to the output directory.
        evaluate(output_dir, batch_size=16, checkpoint_path=None):
            Evaluates the model on the test dataset and returns evaluation metrics.
    """
    def __init__(self,
                 model_name: str,
                 train_dict: Dict[str, List[Any]],
                 val_dict: Dict[str, List[Any]],
                 test_dict: Dict[str, List[Any]]):
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset = FakeNewsDataset(
            texts=train_dict["texts"],
            labels=train_dict["labels"],
            tokenizer=self.tokenizer
        )
        self.val_dataset = FakeNewsDataset(
            texts=val_dict["texts"],
            labels=val_dict["labels"],
            tokenizer=self.tokenizer
        )
        self.test_dataset = FakeNewsDataset(
            texts=test_dict["texts"],
            labels=test_dict["labels"],
            tokenizer=self.tokenizer
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(id2Label),
            device_map="auto"
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train(self, output_dir: str,
              lora_kwargs: Optional[Dict[str, Any]] = None,
              model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Trains the model with the provided LoRA and model configurations and saves it to the output directory.
        Args:
            output_dir (str): The directory to save the trained model.
            lora_kwargs (dict, optional): LoRA configuration parameters. Defaults to None.
            model_kwargs (dict, optional): Model training parameters. Defaults to None.
        """
        print(f"\n--- Starting Training for {self.model_name} ---")
        
        if lora_kwargs is None:
            lora_kwargs = {
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["query", "key", "value"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.SEQ_CLS
            }

        if model_kwargs is None:
            model_kwargs = {
                "learning_rate": 3e-5,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 16,
                "num_train_epochs": 100,
                "weight_decay": 0.01,
                "max_grad_norm": 0.3,
                "warmup_ratio": 0.03,
                "fp16": True,
                "bf16": False,
                "eval_strategy": "epoch",
                "eval_steps": 100,
                "save_strategy": "epoch",
                "logging_strategy": "epoch",
                "lr_scheduler_type": "cosine",
                "group_by_length": True,
                "load_best_model_at_end": True,
                "metric_for_best_model": "f1_macro",
                "optim": "paged_adamw_8bit",
                "report_to": "tensorboard"
            }

        print(f"Training with LoRA configuration: {lora_kwargs}")
        print(f"Training with model configuration: {model_kwargs}")

        prepared_model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(**lora_kwargs)
        peft_model = get_peft_model(prepared_model, lora_config)
        peft_model.config.use_cache = False
        peft_model.enable_input_require_grads()

        print("Trainable parameters:", peft_model.get_trainable_parameters())
        print("Untrainable parameters:", peft_model.get_non_trainable_parameters())
        print("Total parameters:", peft_model.get_all_parameters())

        class LogLossCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None and "train_loss" in logs:
                    print(f"Step {state.global_step} - Training Loss: {logs['train_loss']:.4f}")
                if logs is not None and "eval_loss" in logs:
                    print(f"Step {state.global_step} - Evaluation Loss: {logs['eval_loss']:.4f}")

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)

        training_args = TrainingArguments(
            output_dir=output_dir,
            **model_kwargs
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[LogLossCallback, early_stopping_callback]
        )

        trainer.train()
        print(f"\n--- Training completed. Saving model to {output_dir} ---")


    def evaluate(self, output_dir: str, batch_size: int = 16, checkpoint_path: str = None):
        """
        Evaluates the model on the test dataset and returns evaluation metrics.
        Args:
            output_dir (str): The directory to save the evaluation results.
            batch_size (int, optional): The batch size for evaluation. Defaults to 16.
            checkpoint_path (str, optional): The path to the model checkpoint for evaluation. Defaults to None.
        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        print(f"\n--- Starting Evaluation ---")

        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided for evaluation.")
        if not checkpoint_path:
            raise ValueError("Checkpoint path cannot be empty.")
        if not self.test_dataset:
            raise ValueError("Test dataset is empty or not provided.")
        
        config = PeftConfig.from_pretrained(checkpoint_path)

        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, 
            num_labels=len(id2Label),
            device_map="auto"
            trust_remote_code=True
        )

        eval_model = PeftModel.from_pretrained(model, checkpoint_path).eval()

        evaluator = LLM_EvaluatorAndPredictor(
            model=eval_model,
            tokenizer=self.tokenizer,
        )

        results = evaluator.evaluate(
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            compute_metrics_fn=compute_metrics,
            batch_size=batch_size,
            dataset_name_for_report="roberta"
        )

        print(f"\n--- Evaluation completed. Results saved to {output_dir} ---")
        return results
    
    def predict(self, inputs_for_prediction: List[str], checkpoint_path: str = None):
        """
        Makes predictions on the provided inputs using the trained model.
        Args:
            inputs_for_prediction (list of str): The input text(s) to predict labels for.
            checkpoint_path (str, optional): The path to the model checkpoint for prediction. Defaults to None.
        Returns:
            list: A list of dictionaries containing the input text, predicted label ID, and predicted label name.
        """
        print(f"\n--- Starting Prediction ---")
        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided for prediction.")
        if not inputs_for_prediction:
            raise ValueError("No inputs provided for prediction.")
        
        config = PeftConfig.from_pretrained(checkpoint_path)

        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, 
            num_labels=len(id2Label),
            device_map="auto"
            trust_remote_code=True
        )

        eval_model = PeftModel.from_pretrained(model, checkpoint_path).eval()

        evaluator = LLM_EvaluatorAndPredictor(
            model=eval_model,
            tokenizer=self.tokenizer,
        )

        predictions = evaluator.predict(inputs_for_prediction=inputs_for_prediction)
        print(f"\n{self.model_name.upper()} Inference Predictions:", predictions)
        
        return predictions

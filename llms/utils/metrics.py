"""
This module computes various evaluation metrics for model predictions.
"""

import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(p) -> dict[str, float]:
    """
    Compute metrics for model predictions.
    Args:
        p: A PredictionOutput object containing predictions and label_ids.
    Returns:
        A dictionary with computed metrics: accuracy, precision, recall, f1 (macro), and f1 per class.
    """

    preds_logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    preds_labels = np.argmax(preds_logits, axis=1)

    # --- Accuracy ---
    accuracy = accuracy_metric.compute(predictions=preds_labels, references=labels)["accuracy"]

    # --- Precision, Recall, F1 (Macro) ---
    precision = precision_metric.compute(predictions=preds_labels, references=labels, average="macro", zero_division=0)["precision"]
    recall = recall_metric.compute(predictions=preds_labels, references=labels, average="macro", zero_division=0)["recall"]
    f1 = f1_metric.compute(predictions=preds_labels, references=labels, average="macro")["f1"]

    # --- F1 for each class ---
    f1_per_class = f1_metric.compute(predictions=preds_labels, references=labels, average=None)["f1"]
    metrics_per_class = {}
    if f1_per_class is not None:
        for i, f1_val in enumerate(f1_per_class):
            metrics_per_class[f"f1_class_{i}"] = f1_val

    results = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_per_class": metrics_per_class
    }

    return results


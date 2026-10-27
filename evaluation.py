"""Classification and multi-label evaluation module for RadInsight AI.

Calculates genuine Scikit-Learn metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC,
Hamming Loss) on labeled test radiographs with full per-class breakdowns and confusion matrices.
Never fabricates metrics when data is unmeasured.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                                                        
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.settings import (
    DATA_DIR,
    DEFAULT_DECISION_THRESHOLDS,
    SUPPORTED_FINDINGS,
    THEME_COLORS,
)
from image_processing import preprocess_chest_xray

logger = logging.getLogger(__name__)

def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive multi-label classification metrics using Scikit-Learn.

    Args:
        y_true: Binary ground-truth matrix of shape (N_samples, N_classes).
        y_pred_probs: Continuous predicted probabilities in [0, 1] of shape (N_samples, N_classes).
        thresholds: Dictionary of per-class decision thresholds.
        class_names: List of class labels.

    Returns:
        Structured dictionary containing global metrics, per-class breakdown,
        and confusion matrix elements.
    """
    if class_names is None:
        class_names = list(SUPPORTED_FINDINGS)

    num_samples, num_classes = y_true.shape
    thresh_dict = thresholds or DEFAULT_DECISION_THRESHOLDS

                                         
    y_pred_binary = np.zeros_like(y_true, dtype=int)
    for idx, c_name in enumerate(class_names):
        t = thresh_dict.get(c_name, 0.50)
        y_pred_binary[:, idx] = (y_pred_probs[:, idx] >= t).astype(int)

                                   
    subset_acc = accuracy_score(y_true, y_pred_binary)
    hamm_loss = hamming_loss(y_true, y_pred_binary)

                                                    
    macro_f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_binary, average="micro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)

                                                                                     
    valid_aucs = []
    per_class_metrics: Dict[str, Dict[str, Any]] = {}

    for idx, c_name in enumerate(class_names):
        col_true = y_true[:, idx]
        col_prob = y_pred_probs[:, idx]
        col_pred = y_pred_binary[:, idx]

                                              
        tn, fp, fn, tp = confusion_matrix(col_true, col_pred, labels=[0, 1]).ravel()

        cls_prec = precision_score(col_true, col_pred, zero_division=0)
        cls_rec = recall_score(col_true, col_pred, zero_division=0)
        cls_f1 = f1_score(col_true, col_pred, zero_division=0)
        cls_acc = (tp + tn) / max(num_samples, 1)

                                                                             
        unique_vals = np.unique(col_true)
        if len(unique_vals) > 1:
            try:
                cls_auc = float(roc_auc_score(col_true, col_prob))
                valid_aucs.append(cls_auc)
            except Exception:
                cls_auc = None
        else:
            cls_auc = None                                                          

        per_class_metrics[c_name] = {
            "accuracy": round(float(cls_acc), 4),
            "precision": round(float(cls_prec), 4),
            "recall": round(float(cls_rec), 4),
            "f1_score": round(float(cls_f1), 4),
            "roc_auc": round(float(cls_auc), 4) if cls_auc is not None else None,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "support": int(np.sum(col_true)),
            "threshold_used": round(float(thresh_dict.get(c_name, 0.50)), 4),
        }

    macro_auc = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None

    return {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "subset_accuracy": round(float(subset_acc), 4),
        "hamming_loss": round(float(hamm_loss), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_roc_auc": macro_auc,
        "per_class": per_class_metrics,
    }

def evaluate_test_dataset(
    classifier_instance: Any,
    manifest_path: Optional[Path] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run genuine evaluation across a labeled test dataset.

    Args:
        classifier_instance: Initialized ChestXRayClassifier instance.
        manifest_path: Path to ground_truth.json containing list of {image_file, labels}.
        thresholds: Dictionary of decision thresholds.

    Returns:
        Dictionary containing evaluation status and calculated metrics, or
        clean empty-state notification if no evaluation set is present.
    """
    if manifest_path is None:
        manifest_path = DATA_DIR / "ground_truth.json"

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return {
            "evaluated": False,
            "message": (
                f"No ground-truth manifest found at {manifest_path}. "
                "To run an evaluation, place labeled test chest radiographs into 'data/sample_test/' "
                "and configure 'data/ground_truth.json'."
            ),
        }

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        return {
            "evaluated": False,
            "message": f"Failed to parse test manifest JSON: {str(e)}",
        }

    if not records or len(records) == 0:
        return {
            "evaluated": False,
            "message": "Ground-truth manifest is empty. Add labeled samples to compute metrics.",
        }

    class_names = list(SUPPORTED_FINDINGS)
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    y_true_list = []
    y_prob_list = []
    processed_files = []

    for item in records:
        img_rel_path = item.get("image_file")
        if not img_rel_path:
            continue

                                                        
        img_path = DATA_DIR / img_rel_path
        if not img_path.exists():
            img_path = manifest_path.parent / img_rel_path
        if not img_path.exists():
            logger.warning(f"Test image not found on disk: {img_path}")
            continue

        try:
            tensor_batch, _, _ = preprocess_chest_xray(str(img_path))
            pred_result = classifier_instance.predict(tensor_batch)
            raw_scores = pred_result["raw_scores"]

                                 
            gt_labels = set(item.get("labels", []))
            row_true = np.zeros(num_classes, dtype=int)
            row_prob = np.zeros(num_classes, dtype=float)

            for idx, c_name in enumerate(class_names):
                if c_name in gt_labels:
                    row_true[idx] = 1
                row_prob[idx] = raw_scores.get(c_name, 0.0)

            y_true_list.append(row_true)
            y_prob_list.append(row_prob)
            processed_files.append(str(img_path.name))

        except Exception as e:
            logger.error(f"Error evaluating sample {img_path}: {e}")
            continue

    if len(y_true_list) == 0:
        return {
            "evaluated": False,
            "message": "No test images could be loaded from the manifest. Verify image paths on disk.",
        }

    y_true_mat = np.array(y_true_list, dtype=int)
    y_prob_mat = np.array(y_prob_list, dtype=float)

    metrics = compute_multilabel_metrics(
        y_true_mat, y_prob_mat, thresholds=thresholds, class_names=class_names
    )
    metrics["evaluated"] = True
    metrics["processed_files"] = processed_files
    metrics["manifest_source"] = str(manifest_path)
    metrics["model_status"] = classifier_instance.status
    metrics["y_true"] = y_true_mat
    metrics["y_prob"] = y_prob_mat

    return metrics

def create_confusion_matrix_figure(per_class_metrics: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """Generate a clean 2x4 grid of 2x2 confusion matrices for all supported findings."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for idx, (c_name, m) in enumerate(per_class_metrics.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        tp, fp, tn, fn = m["true_positives"], m["false_positives"], m["true_negatives"], m["false_negatives"]
        matrix = np.array([[tn, fp], [fn, tp]])

        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{c_name}\n(F1: {m['f1_score']:.2f})", fontsize=11, fontweight="bold", color="#1e3a8a")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], fontsize=9)
        ax.set_yticklabels(["Neg", "Pos"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

                              
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    color="white" if val > matrix.max() / 2 else "#0f172a",
                    fontweight="bold", fontsize=10,
                )

    plt.tight_layout()
    return fig

def create_roc_curves_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Optional[plt.Figure]:
    """Plot multi-class ROC curves for findings with valid ground-truth variation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_any = False

    colors = ["#1e3a8a", "#0d9488", "#0284c7", "#d97706", "#7c3aed", "#e11d48", "#059669", "#475569"]

    for idx, c_name in enumerate(class_names):
        col_true = y_true[:, idx]
        col_prob = y_prob[:, idx]
        if len(np.unique(col_true)) > 1:
            fpr, tpr, _ = roc_curve(col_true, col_prob)
            auc_val = roc_auc_score(col_true, col_prob)
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, label=f"{c_name} (AUC = {auc_val:.2f})", color=color, lw=1.8)
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random Guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold", color="#1e3a8a")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

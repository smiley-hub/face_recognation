"""
TODO: Implement compute_metrics() below.

Run this file directly to verify your implementation:
    python src/evaluate.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# -------------------------------------
# Do not modify
# -------------------------------------
@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run inference over a full DataLoader.

    Returns:
        probs  : (n, 2) softmax probabilities — column 0 = NOT YOU, column 1 = YOU
        preds  : (n,)   predicted class indices (0 or 1)
        labels : (n,)   ground-truth class indices (0 or 1)
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_probs),
        torch.cat(all_preds),
        torch.cat(all_labels),
    )


# -------------------------------------
# YOUR TASK
# -------------------------------------
def compute_metrics(   
) -> dict:
    """
    Compute binary classification metrics treating label 1 (YOU) as positive.

    Args:
        preds  : (n,) integer tensor of predicted class indices (0 or 1).
        labels : (n,) integer tensor of ground-truth class indices (0 or 1).

    Returns a dict with four float values:
        {
            "accuracy" : correct / total,
            "precision": TP / (TP + FP),   # 0.0 if TP + FP == 0
            "recall"   : TP / (TP + FN),   # 0.0 if TP + FN == 0
            "f1"       : 2 * P * R / (P + R),  # 0.0 if P + R == 0
        }

    Definitions (positive class = YOU = label 1):
        TP: predicted YOU,     truly YOU
        FP: predicted YOU,     truly NOT YOU
        FN: predicted NOT YOU, truly YOU
        TN: predicted NOT YOU, truly NOT YOU
    """
    
    raise NotImplementedError("Implement compute_metrics() in src/evaluate.py")


# -------------------------------------
# Do not modify
# -------------------------------------
def print_report(
    metrics: dict,
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Print a formatted per-sample evaluation report. PROVIDED."""
    print("=" * 58)
    print("Evaluation Report  (10 test images)")
    print("=" * 58)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print()
    header = f"  {'#':<4} {'True':<10} {'P(YOU)':<9} {'P(NOT YOU)':<12} {'Pred'}"
    print(header)
    print("  " + "-" * 50)
    for i in range(len(labels)):
        true_lbl = "YOU"     if labels[i].item() == 1 else "NOT YOU"
        pred_lbl = "YOU"     if probs[i, 1].item() >= 0.5 else "NOT YOU"
        mark     = "✓" if true_lbl == pred_lbl else "✗"
        print(
            f"  {i+1:<4} {true_lbl:<10} "
            f"{probs[i,1].item():.4f}    "
            f"{probs[i,0].item():.4f}       "
            f"{pred_lbl} {mark}"
        )


# -------------------------------------
# Do not modify
# Sanity check — run with: python src/evaluate.py
# -------------------------------------
if __name__ == "__main__":
    print("Running compute_metrics sanity checks ...\n")

    # Perfect predictions
    preds  = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    m = compute_metrics(preds, labels)
    assert m["accuracy"]  == 1.0, f"Expected accuracy 1.0, got {m['accuracy']}"
    assert m["precision"] == 1.0, f"Expected precision 1.0, got {m['precision']}"
    assert m["recall"]    == 1.0, f"Expected recall 1.0, got {m['recall']}"
    assert m["f1"]        == 1.0, f"Expected f1 1.0, got {m['f1']}"
    print("  Perfect-prediction check ✓")

    # All predicted as NOT YOU → recall = 0, f1 = 0
    preds  = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    m = compute_metrics(preds, labels)
    assert m["accuracy"]  == 0.5, f"Expected 0.5, got {m['accuracy']}"
    assert m["precision"] == 0.0, f"Expected 0.0, got {m['precision']}"
    assert m["recall"]    == 0.0, f"Expected 0.0, got {m['recall']}"
    assert m["f1"]        == 0.0, f"Expected 0.0, got {m['f1']}"
    print("  All-negative edge case ✓")

    # Mixed case
    preds  = torch.tensor([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    m = compute_metrics(preds, labels)
    # TP=3, FP=0, FN=2, TN=5  →  acc=0.8, prec=1.0, rec=0.6, f1=0.75
    assert abs(m["accuracy"]  - 0.8)  < 1e-6, f"Expected 0.8, got {m['accuracy']}"
    assert abs(m["precision"] - 1.0)  < 1e-6, f"Expected 1.0, got {m['precision']}"
    assert abs(m["recall"]    - 0.6)  < 1e-6, f"Expected 0.6, got {m['recall']}"
    assert abs(m["f1"]        - 0.75) < 1e-6, f"Expected 0.75, got {m['f1']}"
    print("  Mixed-prediction check ✓")

    print("\nAll compute_metrics sanity checks passed.")

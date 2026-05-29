"""
TODO: Implement train_one_epoch() below.

Usage (after implementing):
    python src/train.py --config ../config.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_dataloaders
from model import build_model


# -------------------------------------
# Do not modify
# -------------------------------------
def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility. PROVIDED — do not modify."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------
# YOUR TASK
# -------------------------------------
def train_one_epoch(
) -> tuple[float, float]:
    """
    Run one full training pass over the DataLoader.

    Steps to implement:
      1. Set the model to training mode (model.train()).
      2. Loop over batches. For each (images, labels) batch:
           a. Move images and labels to device.
           b. Zero the optimizer gradients.
           c. Forward pass: compute logits = model(images).
           d. Compute loss = criterion(logits, labels).
           e. Backward pass: loss.backward().
           f. Update parameters: optimizer.step().
      3. Accumulate total loss (weighted by batch size) and correct predictions.
      4. Return (avg_loss, accuracy) for the full epoch.

    Args:
        model:     the CNN model (already on device).
        loader:    DataLoader yielding (images, labels) batches.
        criterion: loss function (nn.CrossEntropyLoss).
        optimizer: parameter optimizer.
        device:    torch.device.

    Returns:
        avg_loss (float): mean cross-entropy loss across all samples.
        accuracy (float): fraction of correctly classified samples.
        """
    
    raise NotImplementedError("Implement train_one_epoch() in src/train.py")


# -------------------------------------
# Do not modify
# -------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader (no parameter updates). PROVIDED."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits         = model(images)
        loss           = criterion(logits, labels)
        total_loss    += loss.item() * len(labels)
        correct       += (logits.argmax(dim=1) == labels).sum().item()
        total         += len(labels)

    return total_loss / total, correct / total


# -------------------------------------
# Do not modify
# -------------------------------------
def run(config_path: str) -> dict:
    """Full training pipeline. PROVIDED — do not modify."""
    with open(config_path) as f:
        cfg = json.load(f)

    tcfg = cfg["training"]
    dcfg = cfg["data"]

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        positive_dir=dcfg["positive_dir"],
        negative_dir=dcfg["negative_dir"],
        image_size=dcfg["image_size"],
        batch_size=tcfg["batch_size"],
    )
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model     = build_model(cfg["model"]["backbone"], cfg["model"]["num_classes"])
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tcfg["learning_rate"],
    )

    ckpt_dir = Path(dcfg["checkpoint_dir"])
    log_dir  = Path(dcfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_ctr = 0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, tcfg["epochs"] + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= tcfg.get("patience", 999):
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    np.save(str(log_dir / "history.npy"), history)
    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print(f"History saved     → {log_dir / 'history.npy'}")
    return history


# -------------------------------------
# Entry point: do not modify
# -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNG403 A2 — Face Authentication Training")
    parser.add_argument("--config", default="../config.json", help="Path to config.json")
    args = parser.parse_args()
    run(args.config)

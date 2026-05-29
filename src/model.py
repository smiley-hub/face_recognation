"""
TODO: Implement build_model() below.

Run this file directly to check your implementation:
    python src/model.py
"""

import torch
import torch.nn as nn
from torchvision import models


# -------------------------------------
# Do not modify
# -------------------------------------
def load_pretrained_backbone(backbone_name: str = "resnet18") -> nn.Module:
    """Load a ImageNet-pretrained ResNet18. PROVIDED — do not modify."""
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model   = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: '{backbone_name}'. Only 'resnet18' is supported.")
    return model


# -------------------------------------
# YOUR TASK
# -------------------------------------
def build_model(backbone_name: str = "resnet18", num_classes: int = 2) -> nn.Module:
    """
    Steps to implement:
      1. Load the pretrained backbone with load_pretrained_backbone().
      2. Freeze ALL backbone parameters so they are not updated during training
         (set requires_grad = False on every parameter).
      3. Replace the final fully-connected layer (model.fc) with a NEW
         nn.Linear that maps from the backbone's feature dimension to num_classes.
         This new layer must remain trainable (requires_grad = True by default).
      4. Return the modified model.

    Note: Do NOT unfreeze the backbone. Only the new head should be trainable.
    """
    model = load_pretrained_backbone(backbone_name)

    for param in model.parameters():
        param.requires_grad = False 

    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# -------------------------------------
# Sanity check — run with: python src/model.py
# -------------------------------------
if __name__ == "__main__":
    print("Running model sanity checks ...\n")

    model = build_model("resnet18", num_classes=2)

    total     = sum(p.numel() for p in model.parameters())
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = total - frozen

    print(f"  Total parameters     : {total:,}")
    print(f"  Frozen parameters    : {frozen:,}")
    print(f"  Trainable parameters : {trainable:,}")

    # The head alone should be trainable (512*2 weights + 2 biases = 1026)
    assert trainable == 1026, (
        f"Expected 1026 trainable params (head only), got {trainable}. "
        "Make sure you freeze the backbone before replacing model.fc."
    )
    print("  Trainable param count ✓")

    # Output shape check
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, 2), f"Expected output shape (4, 2), got {out.shape}"
    print(f"  Output shape {out.shape} ✓")

    # The new fc should indeed be trainable
    assert model.fc.weight.requires_grad, "model.fc.weight should require grad"
    print("  model.fc is trainable ✓")

    print("\nAll model sanity checks passed.")

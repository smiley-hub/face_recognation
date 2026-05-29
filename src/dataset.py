"""
This file is COMPLETE. Do not modify it.

It provides:
  - download_negative_class(): downloads the fixed set of 5 LFW identities
  - FaceDataset: a PyTorch Dataset for face images
  - get_dataloaders(): builds train/val/test DataLoaders
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# -------------------------------------
# Fixed negative class — DO NOT CHANGE
# These 5 identities and their split assignments ensure that every student
# uses the exact same negative samples.
# -------------------------------------
NEGATIVE_IDENTITIES = [
    "Serena Williams",
    "Angelina Jolie",
    "Alejandro Toledo",
    "Recep Tayyip Erdogan",
    "George W Bush"
]

# Per-person image indices used for each split.
# Indices are stable: sklearn sorts images by filename within each identity.
NEGATIVE_SPLIT = {
    "train": [0, 1, 2, 3],  # 4 × 5 people = 20 negative train images
    "val":   [4],            # 1 × 5 people =  5 negative val images
    "test":  [5],            # 1 × 5 people =  5 negative test images
}

# Class labels
LABEL_NOT_YOU = 0
LABEL_YOU     = 1


# -------------------------------------
# Negative-class downloader
# -------------------------------------
def download_negative_class(save_dir: str = "data/negative") -> None:
    """
    Download LFW, extract the five fixed identities, and save as JPEG files.

    Directory layout after this call:
        data/negative/
            train/  Serena_Williams_0000.jpg  Angelina_Jolie_0000.jpg  ...
            val/    Serena_Williams_0004.jpg  ...
            test/   Serena_Williams_0005.jpg  ...

    This function is idempotent — safe to call multiple times.
    """
    from sklearn.datasets import fetch_lfw_people  # lazy import

    save_dir = Path(save_dir)
    # Skip download if already present
    n_expected = sum(len(v) for v in NEGATIVE_SPLIT.values()) * len(NEGATIVE_IDENTITIES)
    n_found    = sum(1 for p in save_dir.rglob("*.jpg"))
    if n_found >= n_expected:
        print(f"Negative class already downloaded ({n_found} images in {save_dir}). Skipping.")
        return

    print("Downloading LFW dataset (first run only — ~200 MB) ...")
    lfw     = fetch_lfw_people(min_faces_per_person=6, color=True, resize=1.0)
    names   = lfw.target_names   # sorted array of identity names
    images  = lfw.images         # float32 (n_samples, h, w, 3) in [0, 1]
    targets = lfw.target         # integer class index per sample

    for split in NEGATIVE_SPLIT:
        (save_dir / split).mkdir(parents=True, exist_ok=True)

    for name in NEGATIVE_IDENTITIES:
        match = np.where(names == name)[0]

        class_idx     = match[0]
        person_images = images[targets == class_idx]  # sorted by filename

        im_name = name.replace(" ", "_")
        for split, indices in NEGATIVE_SPLIT.items():
            for idx in indices:
                img   = Image.fromarray((person_images[idx] * 255).astype(np.uint8))
                fname = f"{im_name}_{idx:04d}.jpg"
                img.save(save_dir / split / fname)

    n_saved = sum(len(v) for v in NEGATIVE_SPLIT.values()) * len(NEGATIVE_IDENTITIES)
    print(f"Negative class saved: {n_saved} images --> {save_dir}/")
    for split, indices in NEGATIVE_SPLIT.items():
        n = len(indices) * len(NEGATIVE_IDENTITIES)
        print(f"  {split:<5}: {n} images")


# -------------------------------------
# PyTorch Dataset
# -------------------------------------
class FaceDataset(Dataset):
    """
    Binary face dataset.
      label 1  -->  YOU      (positive class)
      label 0  -->  NOT YOU  (negative class)
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
    ) -> None:
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# -------------------------------------
# Transforms
# -------------------------------------
def get_transforms(image_size: int = 224, augment: bool = False):
    """Return an inference or training transform pipeline."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


# -------------------------------------
# DataLoader builder
# -------------------------------------
def _collect_images(folder: str, label: int) -> Tuple[List[str], List[int]]:
    """Return sorted (paths, labels) for all images in folder."""
    folder = Path(folder)
    exts   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths  = sorted(p for p in folder.iterdir() if p.suffix in exts)
    if not paths:
        raise FileNotFoundError(
            f"No images found in '{folder}'. "
            "Make sure the folder exists and contains image files."
        )
    return [str(p) for p in paths], [label] * len(paths)


def get_dataloaders(
    positive_dir: str,
    negative_dir: str,
    image_size: int = 224,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders.

    Expected layout:
        positive_dir/{train,val,test}/   — your personal images
        negative_dir/{train,val,test}/   — fixed LFW negatives

    Returns:
        train_loader, val_loader, test_loader
    """
    train_tfm = get_transforms(image_size, augment=True)
    eval_tfm  = get_transforms(image_size, augment=False)

    loaders = []
    for split, tfm in [("train", train_tfm), ("val", eval_tfm), ("test", eval_tfm)]:
        pos_paths, pos_labels = _collect_images(f"{positive_dir}/{split}", LABEL_YOU)
        neg_paths, neg_labels = _collect_images(f"{negative_dir}/{split}", LABEL_NOT_YOU)

        ds = FaceDataset(
            pos_paths + neg_paths,
            pos_labels + neg_labels,
            transform=tfm,
        )
        loaders.append(DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ))

    return tuple(loaders)


# -------------------------------------
# Sanity-check - used by notebook
# -------------------------------------
def check_no_test_leakage(positive_dir: str) -> None:
    """
    Assert that no test image appears in train or val splits.
    Compares MD5 hashes so renaming a file does not evade the check.
    Raises AssertionError with a clear message if leakage is detected.
    """
    def hashes(folder):
        return {hashlib.md5(Path(p).read_bytes()).hexdigest()
                for p in Path(folder).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    train_h = hashes(f"{positive_dir}/train")
    val_h   = hashes(f"{positive_dir}/val")
    test_h  = hashes(f"{positive_dir}/test")

    overlap_train = train_h & test_h
    overlap_val   = val_h   & test_h

    assert not overlap_train, (
        f"DATA LEAKAGE DETECTED: {len(overlap_train)} test image(s) found in train/ --> ZERO GRADE"
    )
    assert not overlap_val, (
        f"DATA LEAKAGE DETECTED: {len(overlap_val)} test image(s) found in val/ --> ZERO GRADE"
    )
    print(" No test-image leakage detected.")

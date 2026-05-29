# CNG403 Assignment 2: Personal Face Authentication

## Overview

This assignment guides you through fine-tuning a **pretrained CNN** to solve a real binary classification problem: given a face image, determine whether it belongs to **YOU** or **NOT YOU**.

**Key Learning Objectives:**
- Understand transfer learning and feature reuse from pretrained models
- Fine-tune ResNet18 on a small personal dataset
- Evaluate a model with precision, recall, and F1 score
- Deploy a trained model as an interactive Gradio web application

---

## Project Structure

```
assignment2/
├── src/                          # Source code directory
│   ├── dataset.py                # Complete: fixed negative class loader + FaceDataset
│   ├── model.py                  # [TODO] Implement: build_model()
│   ├── train.py                  # [TODO] Implement: train_one_epoch()
│   ├── evaluate.py               # [TODO] Implement: compute_metrics()
│   └── app.py                    # [TODO] Implement: predict() and build_app()
│
├── data/
│   ├── positive/
│   │   ├── train/                # Your 20 personal training images
│   │   ├── val/                  # Your 5 personal validation images
│   │   └── test/                 # Your 5 personal test images (DO NOT USE IN TRAINING)
│   └── negative/
│       ├── train/                # Auto-populated by dataset.py (20 images)
│       ├── val/                  # Auto-populated by dataset.py (5 images)
│       └── test/                 # Auto-populated by dataset.py (5 images)
│
├── checkpoints/
│   └── best_model.pt             # Saved during training (excluded from submission)
│
├── logs/
│   └── history.npy               # Training history (excluded from submission)
│
├── config.json                   # Hyperparameters and paths
├── notebook.ipynb                # Main student interface — run sections in order
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Your Tasks: What to Implement

| File | Function | Points | Sanity Check |
|------|----------|--------|--------------|
| [src/model.py] | `build_model()` | 20 pts | `python src/model.py` |
| [src/train.py] | `train_one_epoch()` | 25 pts | Training must converge |
| [src/evaluate.py] | `compute_metrics()` | 15 pts | `python src/evaluate.py` |
| [src/app.py] | `predict()` + `build_app()` | 20 pts | 5 screenshots required |
| Questions (Q1–Q4) | Written answers in notebook | 20 pts | — |
| **Total** | | **100 pts** | |

---

## Getting Started

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare your positive images

Collect **30 photos of your own face** and organize them:

```
data/positive/
    train/   <-- 20 images
    val/     <-- 5 images
    test/    <-- 5 images  ⚠️  DO NOT use these in training or validation
```

### Step 3: Follow the notebook in order

```
1. Setup  -->  2. Dataset  -->  3. Model  -->  4. Training  -->  5. Evaluation  -->  6. Gradio App
```

---

## Data Details

| Split | YOU (positive) | NOT YOU (negative) | Total |
|-------|---------------|-------------------|-------|
| Train | 20 | 20 | 40 |
| Val | 5 | 5 | 10 |
| Test | 5 | 5 | 10 |

Negative class is fixed for all students: Serena Williams, Angelina Jolie, Alejandro Toledo, Recep Tayyip Erdogan, George W Bush.

---

## Submission

Submit a `.zip` excluding `data/` and `checkpoints/`, plus a `screenshots/` folder with 5 Gradio prediction screenshots.

**⚠️ Including test images in train/ or val/ = zero grade.**

Good luck!

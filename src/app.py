"""
TODO: Implement predict() and build_app() below.

After implementing, launch the app from the notebook
or directly with: python src/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model import build_model


# -------------------------------------
# Do not modify
# -------------------------------------
def load_model(config_path: str = "../config.json") -> tuple[nn.Module, torch.device]:
    """Load the best saved checkpoint. PROVIDED — do not modify."""
    with open(config_path) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg["model"]["backbone"], cfg["model"]["num_classes"])
    ckpt   = Path(cfg["data"]["checkpoint_dir"]) / "best_model.pt"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt}'. "
            "Run training (Section 4 of the notebook) first."
        )

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {ckpt} (device: {device})")
    return model, device


# -------------------------------------
# Do not modify
# -------------------------------------
def get_inference_transform(image_size: int = 224):
    """Return the evaluation-time transform. PROVIDED — do not modify."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


# -------------------------------------
# YOUR TASK 1
# -------------------------------------
def predict(image, model:nn.Module, device:torch.device, image_size=224
) -> dict:
    """
    Run inference on a single PIL image.

    Steps to implement:
      1. Apply get_inference_transform(image_size) to the image.
      2. Add a batch dimension with .unsqueeze(0) and move to device.
      3. Run the model inside a torch.no_grad() context.
      4. Apply torch.softmax(..., dim=1) on the logits to get probabilities.
      5. Return a dict:  {"NOT YOU": float, "YOU": float}
         where each value is a probability in [0, 1].

    Args:
        image     : PIL Image (RGB).
        model     : loaded nn.Module, already on device, in eval mode.
        device    : torch.device.
        image_size: resize target (must match training).

    Returns:
        dict with keys "NOT YOU" and "YOU" mapping to float probabilities.
        """
    
    transform = get_inference_transform(image_size)  #this will convert the image to a tensor and normalize it  example [3, 224, 224]
    image = transform(image).unsqueeze(0)# I add a batch dimension to the image tensor so it becomes [1, 3, 224, 224] and move it to the same device as the model
    image = image.to(device)

    with torch.no_grad():
        logits = model(image) #model logits outputs 2 class scores of the input image 
        probs = torch.softmax(logits, dim=1) #softmax is applied to convert the logits into probabilities for each class.
        return {
            "NOT YOU": probs[0, 0].item(),
            "YOU": probs[0, 1].item(),
        }
    raise NotImplementedError("Implement predict() in src/app.py")


# -------------------------------------
# YOUR TASK 2
# -------------------------------------
def build_app(config_path: str = "../config.json"):
    """
    Build and return a Gradio Interface for face authentication.

    Requirements:
      - Import gradio as gr.
      - Load the model with load_model(config_path).
      - Define an inference function that:
          • accepts a numpy image array (Gradio's default image format),
          • converts it to PIL with Image.fromarray(...).convert("RGB"),
          • calls predict() and returns the result dict.
      - Create and return gr.Interface with:
          • fn       = your inference function
          • inputs   = gr.Image(label="Upload your face")
          • outputs  = gr.Label(num_top_classes=2, label="Prediction")
          • title    = "Personal Face Authentication"
          • description = a short description of what the app does
    """

    import gradio as gr 
    
    with open (config_path)as f:# load the config file to get the image size 
        cfg = json.load(f)
    image_size = cfg["data"]["image_size"]

    
    model, device = load_model(config_path) #load the trained model and the  device it is on (cpu ot cuda)

    def inference_fn(image_array):
        image = Image.fromarray(image_array).convert("RGB") #convert the input image array to a PIL image in RGB format
        return predict(image, model, device, image_size) #call the predict function to get the probabilities for "NOT YOU" and "YOU"
    
    return gr.Interface(
        fn = inference_fn,
        inputs = gr.Image(label="Upload your face"),
        outputs = gr.Label(num_top_classes=2, label="Prediction"),
        title = "Personal Face Authentication",
        description = "Upload a face image and the model will predict whether it is YOU or NOT YOU."
            )

    raise NotImplementedError("Implement build_app() in src/app.py")


# -------------------------------------
# Entry point: do not modify
# -------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch()

"""
Evaluation Script for Complex-AFNet Models
===========================================

This script evaluates trained models on the test dataset and reports:
- Accuracy, Precision, Recall, F1 (overall and per-class)
- Confusion Matrix
- ROC-AUC and PR-AUC

USAGE:
------
1. Place your model weights in a folder (e.g., ./weights/)
2. Ensure your dataset is organized as:

   data_root/
   ├── GASF/
   │   └── test/
            └── record_number/
                ├── segment1_label1/
                └── segment2_label0/

   └── GADF/
       └── test/
           └── record_number/
                ├── segment1_label1/
                └── segment2_label0/
    set mode to subfolder names accordingly in function if different otherwise leave it as None
    example:
    data_root/
        ├── GASF/
                └── record_number/
                    ├── segment1_label1/
                    └── segment2_label0/

        └── GADF/
                └── record_number/
                    ├── segment1_label1/
                    └── segment2_label0/

3. Run the script:

   # Evaluate a single model
   python evaluate.py --model resnet18 --weights ./weights/resnet18.pth --data_root ./data

   # Evaluate all models (place weights in ./weights/ with names matching model names)
   python evaluate.py --all --data_root ./data

AVAILABLE MODELS:
-----------------
- resnet18          : Standard ResNet-18 (real-valued)
- c_resnet18        : Complex-valued ResNet-18
- c_afnet           : Complex AFNet (proposed model)
- inverted_c_afnet  : Inverted Complex AFNet (hybrid)
- grouped_c_afnet   : Grouped Complex AFNet (dual-stream)

"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# Model imports
from models.ResNet import resnet18
from models.C_ResNet import c_resnet18
from models.C_AFNet import c_afnet
from models.R_AFNet import r_afnet
from models.Inverted_C_AFNet import inverted_c_afnet
from models.Grouped_C_AFNet import grouped_c_afnet

# Dataset imports
from data.dataset import CombinedDataset, GASFDataset, GADFDataset, ComplexDataset
from data.transforms import get_transforms


# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================

# Model registry - maps model names to their factory functions
MODEL_REGISTRY = {
    "resnet18": resnet18,
    "c_resnet18": c_resnet18,
    "c_afnet": c_afnet,
    "r_afnet": r_afnet,
    "inverted_c_afnet": inverted_c_afnet,
    "grouped_c_afnet": grouped_c_afnet,
}

# Dataset registry
DATASET_REGISTRY = {
    "combined": CombinedDataset,
    "gasf": GASFDataset,
    "gadf": GADFDataset,
    "complex": ComplexDataset,
}

# Class names for display
CLASS_NAMES = ["Normal", "AF"]


# Model to dataset type mapping
# Complex models use 'complex' dataset (3 channels), real models use 'gadf' dataset (3 channels)
MODEL_DATASET_MAP = {
    "resnet18": "gadf",
    "r_afnet": "complex",
    "c_resnet18": "complex",
    "c_afnet": "complex",
    "inverted_c_afnet": "complex",
    "grouped_c_afnet": "complex",
}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def load_model(model_name, weights_path, input_channels=3, num_classes=1, device=None):
    """Load a model with pretrained weights."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    if device is None:
        device = torch.device("cpu")

    # Create model
    model = MODEL_REGISTRY[model_name](input_channels=input_channels, num_classes=num_classes)

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Determine the state_dict based on how the model was saved
    if isinstance(checkpoint, dict):
        # Check for common checkpoint keys
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume the dict itself is the state_dict
            state_dict = checkpoint
    else:
        # checkpoint might be a full model object
        try:
            state_dict = checkpoint.state_dict()
        except AttributeError:
            raise ValueError(f"Unable to extract state_dict from {weights_path}")
    
    # Handle DataParallel saved models (remove 'module.' prefix)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)
    model.eval()
    
    return model


def create_dataloader(data_root, mode=None, dataset_type="combined", batch_size=32, num_workers=4):
    """Create a test dataloader."""
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_type}. Available: {list(DATASET_REGISTRY.keys())}")

    transforms = get_transforms()
    
    # Create a minimal config dict for the dataset
    class MinimalConfig:
        def __init__(self, data_root):
            self.data_root = data_root
    
    # Temporarily set the data path for dataset initialization
    Dataset = DATASET_REGISTRY[dataset_type]
    
    # Create dataset - pass data_root via environment or modify dataset to accept it
    import yaml
    
    # Create a temporary config
    temp_config = {
        "default_config": {
            "data": {
                "data_dir": data_root,
                "input_shape": [3, 224, 224],
            }
        }
    }
    
    # Write temporary config
    temp_config_path = "/tmp/eval_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(temp_config, f)
    
    test_dataset = Dataset(mode=mode, transforms=transforms, config_path=temp_config_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Run evaluation and return predictions and labels."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Evaluating", ncols=80):
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Handle complex outputs
        if torch.is_complex(outputs):
            outputs = outputs.abs()

        # Get probabilities and predictions
        probs = torch.sigmoid(outputs).view(-1)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def compute_metrics(predictions, probabilities, labels):
    """Compute all evaluation metrics."""
    metrics = {}
    
    # Overall metrics
    metrics["accuracy"] = accuracy_score(labels, predictions)
    metrics["precision_macro"] = precision_score(labels, predictions, average="macro")
    metrics["recall_macro"] = recall_score(labels, predictions, average="macro")
    metrics["f1_macro"] = f1_score(labels, predictions, average="macro")
    
    # Per-class metrics
    metrics["precision_per_class"] = precision_score(labels, predictions, average=None)
    metrics["recall_per_class"] = recall_score(labels, predictions, average=None)
    metrics["f1_per_class"] = f1_score(labels, predictions, average=None)
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(labels, predictions)
    
    # ROC-AUC and PR-AUC
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
        metrics["pr_auc"] = average_precision_score(labels, probabilities)
    except ValueError:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    
    # Specificity and NPV
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return metrics


def print_metrics(metrics, model_name):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 60)
    print(f"RESULTS: {model_name}")
    print("=" * 60)
    
    print(f"\n{'OVERALL METRICS':-^60}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall:       {metrics['recall_macro']:.4f} (macro)")
    print(f"  F1 Score:     {metrics['f1_macro']:.4f} (macro)")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.4f}")
    
    print(f"\n{'PER-CLASS METRICS':-^60}")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*40}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10} {metrics['precision_per_class'][i]:>10.4f} "
              f"{metrics['recall_per_class'][i]:>10.4f} {metrics['f1_per_class'][i]:>10.4f}")
    
    print(f"\n{'CLINICAL METRICS':-^60}")
    print(f"  Sensitivity (Recall):  {metrics['sensitivity']:.4f}")
    print(f"  Specificity:           {metrics['specificity']:.4f}")
    print(f"  PPV (Precision):       {metrics['ppv']:.4f}")
    print(f"  NPV:                   {metrics['npv']:.4f}")
    
    print(f"\n{'CONFUSION MATRIX':-^60}")
    cm = metrics["confusion_matrix"]
    print(f"                  Predicted")
    print(f"                  {CLASS_NAMES[0]:>8} {CLASS_NAMES[1]:>8}")
    print(f"  Actual {CLASS_NAMES[0]:>8} {cm[0, 0]:>8} {cm[0, 1]:>8}")
    print(f"         {CLASS_NAMES[1]:>8} {cm[1, 0]:>8} {cm[1, 1]:>8}")
    print("=" * 60 + "\n")


def evaluate_single_model(args):
    """Evaluate a single model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Auto-select dataset type based on model
    dataset_type = MODEL_DATASET_MAP.get(args.model, "gadf")
    print(f"[INFO] Using dataset type: {dataset_type} for model: {args.model}")
    
    # All models use 3 input channels (complex or gadf)
    input_channels = 3
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_model(
        args.model,
        args.weights,
        input_channels=input_channels,
        num_classes=1,
        device=device,
    )
    
    # Create dataloader
    print(f"Loading dataset from: {args.data_root}")
    if args.mode:
        print(f"Using subfolder: {args.mode}")
    dataloader = create_dataloader(
        args.data_root,
        mode=args.mode,
        dataset_type=dataset_type,
        batch_size=args.batch_size,
    )
    print(f"Test samples: {len(dataloader.dataset)}")
    
    # Evaluate
    predictions, probabilities, labels = evaluate_model(model, dataloader, device)
    
    # Compute and print metrics
    metrics = compute_metrics(predictions, probabilities, labels)
    print_metrics(metrics, args.model)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Complex-AFNet models on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ResNet18 (uses GADF dataset automatically)
  python evaluate.py --model resnet18 --weights ./weights/ResNet18.pth --data_root ./data

  # Evaluate C-AFNet (uses Complex dataset automatically)
  python evaluate.py --model c_afnet --weights ./weights/C_AFNet.pth --data_root ./data

  # Evaluate with subfolder (e.g., test/train/val)
  python evaluate.py --model c_afnet --weights ./weights/C_AFNet.pth --data_root ./data --mode test

Model-Dataset Mapping:
  - resnet18, r_afnet         -> GADF dataset (3 channels)
  - c_resnet18, c_afnet       -> Complex dataset (3 channels)
  - inverted_c_afnet          -> Complex dataset (3 channels)
  - grouped_c_afnet           -> Complex dataset (3 channels)
        """,
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights file (.pth)",
    )
    
    # Data settings
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing GASF and GADF folders",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Subfolder name (e.g., 'test', 'train', 'val'). Use None if data is directly in GASF/GADF folders (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isfile(args.weights):
        parser.error(f"Weights file not found: {args.weights}")
    
    evaluate_single_model(args)


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import gc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import csv
from datetime import datetime

from data.dataset import ComplexDataset, GASFDataset, GADFDataset, CombinedDataset
from data.transforms import get_transforms
from models.ResNet import resnet18
from models.C_ResNet import c_resnet18
from models.Grouped_C_AFNet import grouped_c_afnet
from models.Inverted_C_AFNet import inverted_c_afnet


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Get the path to the default config
        config_path = "config/default.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config["default_config"]


def test_model(model, test_loader, device):
    """Test model and return predictions, probabilities, and true labels"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_inputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Save some inputs for visualization
            if len(all_inputs) < 10:  # Store 10 inputs max
                all_inputs.append(inputs.detach().cpu())

            # Forward pass
            outputs = model(inputs)

            # Get predictions and probabilities
            if torch.is_complex(outputs):
                # For complex outputs, use magnitude
                outputs = outputs.abs()

            # Get probabilities using sigmoid
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).float()

            # Append to lists
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels), all_inputs


def calculate_metrics(true_labels, predictions, probabilities):
    """Calculate and return evaluation metrics"""
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # ROC and AUC
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(
        true_labels, probabilities
    )
    pr_auc = average_precision_score(true_labels, probabilities)

    # Specific metrics for medical diagnostics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "npv": npv,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "pr_curve": (precision_curve, recall_curve),
    }

    return metrics


def plot_confusion_matrix(metrics):
    """Plot confusion matrix from metrics"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "AF"],
        yticklabels=["Normal", "AF"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


def plot_roc_curve(metrics):
    """Plot ROC curve from metrics"""
    plt.figure(figsize=(8, 6))
    fpr, tpr = metrics["roc_curve"]
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f'ROC curve (area = {metrics["roc_auc"]:.2f})',
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("roc_curve.png")
    plt.show()


def plot_pr_curve(metrics):
    """Plot precision-recall curve from metrics"""
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve = metrics["pr_curve"]
    plt.plot(
        recall_curve,
        precision_curve,
        color="green",
        lw=2,
        label=f'PR curve (area = {metrics["pr_auc"]:.2f})',
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("pr_curve.png")
    plt.show()


def save_metrics_to_csv(
    metrics, model_type, dataset_type, model_path, output_path=None
):
    """Save metrics to a CSV file"""
    if output_path is None:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"metrics.csv"

    # Extract metrics for saving
    metrics_to_save = {
        "model_type": model_type,
        "dataset_type": dataset_type,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "specificity": metrics["specificity"],
        "npv": metrics["npv"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(output_path)

    # Write to CSV
    with open(output_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics_to_save.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics_to_save)

    print(f"Metrics saved to {output_path}")
    return output_path


def clear_memory():
    """Clear GPU memory and collect garbage"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    # Clear memory
    clear_memory()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Load dataset
    config = load_config()
    try:
        # Load transformations
        transforms = get_transforms()
        # Create datasets
        dataset_type = "combined"  # Options: "complex", "gasf", "gadf", "combined"
        dataset_classes = {
            "complex": ComplexDataset,
            "gasf": GASFDataset,
            "gadf": GADFDataset,
            "combined": CombinedDataset,
        }
        # Get the appropriate dataset class
        Dataset = dataset_classes[dataset_type]
        test_dataset = Dataset(mode="test", transforms=transforms)
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        print(f"Test dataset loaded with {len(test_dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    try:
        # Get config for model settings
        input_channels = config.get("data", {}).get("input_channels", 3)
        num_classes = config.get("data", {}).get("num_classes", 1)

        # Model registry for easy selection
        model_registry = {
            "resnet18": lambda: resnet18(input_channels=input_channels, num_classes=num_classes),
            "c_resnet18": lambda: c_resnet18(input_channels=input_channels, num_classes=num_classes),
            "inverted_c_afnet": lambda: inverted_c_afnet(input_channels=input_channels, num_classes=num_classes),
            "grouped_c_afnet": lambda: grouped_c_afnet(input_channels=input_channels, num_classes=num_classes),
        }

        # Select model and path (update these as needed)
        model_name = "resnet18"
        model_path = "saved_models_new/resnet18_gasf_gadf/resnet18_gasf_gadf_model_epoch_18.pth"

        if model_name not in model_registry:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(model_registry.keys())}")

        model = model_registry[model_name]()
        state = torch.load(
            model_path,
            map_location=device,
        )
        # Handle DataParallel saved models
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        print(f"Model loaded: {model_name} from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test model
    try:
        predictions, probabilities, true_labels, sample_inputs = test_model(
            model, test_loader, device
        )

        # Calculate metrics
        metrics = calculate_metrics(true_labels, predictions, probabilities)

        # Print results
        print("\nTest Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Negative Predictive Value: {metrics['npv']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
    except Exception as e:
        print(f"Error testing model: {e}")
        return

    # Visualizations
    # try:
    #     # Plot confusion matrix
    #     plot_confusion_matrix(metrics)

    #     # # Plot ROC curve
    #     # plot_roc_curve(metrics)

    #     # # Plot precision-recall curve
    #     # plot_pr_curve(metrics)

    #     # Print summary
    #     print_summary(metrics, model_name, dataset_type)
    # except Exception as e:
    #     print(f"Error in visualization: {e}")

    # Save metrics to CSV
    try:
        output_path = save_metrics_to_csv(metrics, model_name, dataset_type, model_path)
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")

    print("\nTesting complete.")


if __name__ == "__main__":
    main()

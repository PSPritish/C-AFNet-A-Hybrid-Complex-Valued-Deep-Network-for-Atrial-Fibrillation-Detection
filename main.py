import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data.dataloader import get_dataloaders
from models.Grouped_C_AFNet import grouped_c_afnet
from models.ResNet import resnet18
from models.Inverted_C_AFNet import inverted_c_afnet
from models.C_ResNet import c_resnet18
from train import Trainer
import yaml
import time
import pandas as pd


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Get the path to the default config
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        config_path = os.path.join(config_dir, "default.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config["default_config"]


def main():
    # Load config from YAML
    config = load_config()

    # Extract variables for use in script
    learning_rate = config.get("training", {}).get("learning_rate", 0.001)
    epochs = config.get("training", {}).get("epochs", 5)
    batch_size = config.get("training", {}).get("batch_size", 32)
    save_dir = config.get("logging", {}).get("model_save_path", "./saved_models")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Get model config
    model_config = config.get("model", {})
    model_arch = model_config.get("architecture", "resnet18")
    input_channels = config.get("data", {}).get("input_channels", 3)
    num_classes = config.get("data", {}).get("num_classes", 1)

    # Get data loaders
    dataset_type = config.get("data", {}).get("dataset_type", "combined")
    dataloaders = get_dataloaders(dataset_type=dataset_type)

    # Create model based on architecture config
    model_registry = {
        "resnet18": lambda: resnet18(input_channels=input_channels, num_classes=num_classes),
        "c_resnet18": lambda: c_resnet18(input_channels=input_channels, num_classes=num_classes),
        "inverted_c_afnet": lambda: inverted_c_afnet(input_channels=input_channels, num_classes=num_classes),
        "grouped_c_afnet": lambda: grouped_c_afnet(input_channels=input_channels, num_classes=num_classes),
    }

    if model_arch not in model_registry:
        raise ValueError(f"Unknown model architecture: {model_arch}. Available: {list(model_registry.keys())}")

    model = model_registry[model_arch]()
    print(f"Created model: {model_arch}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Set up loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()

    # Convert learning_rate to float if it's a string
    learning_rate = float(config.get("training", {}).get("learning_rate", 0.001))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Add ReduceLROnPlateau scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # Reduce LR when val_loss stops decreasing
        factor=0.5,  # Multiply LR by this factor when reducing
        patience=3,  # Number of epochs with no improvement to wait
        min_lr=1e-6,  # Lower bound on the learning rate
    )

    # Set up trainer
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    # Calculate training time
    start_time = time.time()
    print("\nStarting training...")
    history = trainer.train()
    print(f"\nTraining complete. Model saved to {save_dir}")
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # Save training history to CSV
    history_df = pd.DataFrame(history)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    history_file = os.path.join(save_dir, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to {history_file}")

    # Print best metrics
    best_epoch_idx = int(history_df["val_loss"].idxmin())
    best_metrics = history_df.iloc[best_epoch_idx]
    print(f"\nBest Results (Epoch {int(best_metrics['Epoch'])}):")
    print(f"  Val Loss: {best_metrics['val_loss']:.4f}")
    print(f"  Val Acc: {best_metrics['val_acc']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()

import os
import sched
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data.dataloader import get_dataloaders
from models.grouped_c_af_net import dual_stream_phase_mag_resnet_18
from models.components.losses import ComplexMagnitudeAndPhaseLoss
from models.resnet_18 import resnet18
from models.hybrid_resnet import hybrid_resnet18
from models.proposed_model import hybrid_resnet_RO_18
from train import Trainer
from models.complex_resnet import complex_resnet18
from torchvision import models
import yaml
import time
import pandas as pd
import wandb


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

    # Create run name
    run_name = f"{config.get('run_name', 'nill')}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Initialize wandb with the FULL config from YAML
    wandb.login(key=config.get("wandb_api_key", ""))
    wandb.init(
        entity="CVNN_Based_Networks",
        project=config.get("model", {}).get("architecture", "AF_Detection"),
        name=run_name,
        config=config,
        settings={"console": "wrap"},  # Modern way to capture terminal output
    )

    # dataset_classes = {
    #     "combined": CombinedDataset,
    #     "gasf": GASFDataset,
    #     "gadf": GADFDataset,
    #     "complex": ComplexDataset,
    # }

    # Get data loaders (complex data for complex model)
    dataloaders = get_dataloaders(dataset_type="combined")

    # Create model directly
    # model = complex_resnet18(config)
    model = resnet18()
    # model = DualResNet(config["data"]["num_classes"])
    # model = DualResNetCF(config["data"]["num_classes"])
    # model = hybrid_resnet18()
    # model = hybrid_resnet_RO_18()
    # model = dual_stream_phase_mag_resnet_18()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # model = torch.compile(model)  # Compile the model for better performance doesn't work with complex
    # Log model architecture to wandb
    wandb.watch(model, log="all")

    # Set up simple loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # criterion = ComplexMagnitudeAndPhaseLoss()

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
    config["use_wandb"] = True
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

    # Log final metrics to wandb
    best_epoch_idx = history_df["val_loss"].idxmin()
    best_metrics = history_df.iloc[best_epoch_idx]
    wandb.log(
        {
            "best_val_loss": best_metrics["val_loss"],
            "best_val_acc": best_metrics["val_acc"],
            "best_f1": best_metrics["f1"],
            "best_epoch": best_metrics["Epoch"],
        }
    )

    # Finish wandb run
    wandb.finish()


torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

if __name__ == "__main__":
    main()

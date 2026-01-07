import os
import yaml
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from data.dataset import ComplexDataset, CombinedDataset, GASFDataset, GADFDataset
from data.transforms import get_transforms


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Get the path to the default config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        config_path = os.path.join(config_dir, "default.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config["default_config"]


def get_dataloaders(dataset_type, config_path=None):
    """
    Create train, validation and test dataloaders

    Args:
        dataset_type: Type of dataset to use ('combined', 'gasf', 'gadf', 'complex')
        config_path: Path to config file. If None, loads default config.

    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    # Load configuration
    config = load_config(config_path)

    # Get parameters from config
    batch_size = config.get("training", {}).get("batch_size", 32)
    data_dir = config.get("data", {}).get("data_dir", "/path/to/default/data")

    # Get dataloader parameters from config
    dataloader_config = config.get("data", {}).get("dataloader", {})
    num_workers = dataloader_config.get("num_workers", 4)
    pin_memory = dataloader_config.get("pin_memory", torch.cuda.is_available())
    drop_last_train = dataloader_config.get("drop_last_train", True)
    shuffle_train = dataloader_config.get("shuffle_train", True)
    shuffle_val = dataloader_config.get("shuffle_val", False)
    shuffle_test = dataloader_config.get("shuffle_test", False)
    prefetch_factor = dataloader_config.get("prefetch_factor", 2)
    persistent_workers = dataloader_config.get("persistent_workers", True)
    use_cache = dataloader_config.get("use_cache", False)
    cache_size = dataloader_config.get("cache_size", 2000)
    # Get transforms
    transforms = get_transforms()

    # Set up datasets based on type
    dataset_classes = {
        "combined": CombinedDataset,
        "gasf": GASFDataset,
        "gadf": GADFDataset,
        "complex": ComplexDataset,
    }

    if dataset_type not in dataset_classes:
        raise ValueError(
            f"Invalid dataset type: {dataset_type}. Must be one of {list(dataset_classes.keys())}"
        )
    # Get the appropriate dataset class
    Dataset = dataset_classes[dataset_type]

    # Create datasets
    train_dataset = Dataset(
        mode="train", transforms=transforms, config_path=config_path
    )

    val_dataset = Dataset(mode="val", transforms=transforms, config_path=config_path)
    test_dataset = Dataset(mode="test", transforms=transforms, config_path=config_path)

    # Create dataloaders with settings from config
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


if __name__ == "__main__":
    # Test the dataloaders
    dataloaders = get_dataloaders()

    print(f"Train dataloader batches: {len(dataloaders['train'])}")
    print(f"Validation dataloader batches: {len(dataloaders['val'])}")
    print(f"Test dataloader batches: {len(dataloaders['test'])}")

    # Check a batch
    for images, labels in dataloaders["train"]:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import modules relative to the project root
from data.dataset import ComplexDataset
from data.transforms import get_transforms

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.io import read_image
from torchvision import transforms as T
import yaml


class Compute_stats:
    def __init__(self, mode, transforms=None, config_path=None):
        """
        Args:
        mode: 'train', 'test', or 'val'
        transforms: Optional transforms to apply
        config_path: Path to config file. If None, loads default config.
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Get data directory from config
        data_dir = self.config.get("data", {}).get("data_dir", "/path/to/default/data")

        self.mode = mode.lower()
        self.gasf_dir = os.path.join(data_dir, "GASFC_train_test_val", self.mode)
        self.gadf_dir = os.path.join(data_dir, "GADFC_train_test_val", self.mode)
        self.classes = ["normal", "paroxysmal"]
        self.transforms = transforms
        self.class_to_idx = {"label0": 0, "label1": 1}
        cnt = [0, 0]
        self.data = []

        # Use config for input shape if specified
        self.input_shape = self.config.get("data", {}).get("input_shape", [3, 128, 128])

        # Rest of your implementation remains the same
        subfolders = [
            f
            for f in os.listdir(self.gadf_dir)
            if os.path.isdir(os.path.join(self.gadf_dir, f))
        ]
        for folder in subfolders:
            gadf_path = os.path.join(self.gadf_dir, folder)
            gasf_path = os.path.join(self.gasf_dir, folder)

            image_files = [f for f in os.listdir(gadf_path) if f.endswith(".jpeg")]

            for image_file in image_files:
                gadf_image_path = os.path.join(gadf_path, image_file)
                gasf_image_path = os.path.join(gasf_path, image_file)
                if not os.path.exists(gadf_image_path):
                    raise FileNotFoundError(f"Missing GADF image: {gadf_image_path}")
                if not os.path.exists(gasf_image_path):
                    raise FileNotFoundError(f"Missing GASF image: {gasf_image_path}")
                label_str = image_file.split("_")[-1].split(".")[0]
                if label_str not in self.class_to_idx:
                    raise ValueError(f"Unknown label: {label_str} in file {image_file}")
                label = self.class_to_idx[label_str]
                if label == 0:
                    cnt[0] += 1
                else:
                    cnt[1] += 1
                self.data.append((gadf_image_path, gasf_image_path, label))

        # Compute stats safely
        self.gasf_mean, self.gasf_std = self._compute_stats(
            [s[1] for s in self.data], transforms
        )
        self.gadf_mean, self.gadf_std = self._compute_stats(
            [s[0] for s in self.data], transforms
        )

    def _load_config(self, config_path=None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Get the path to the default config
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config"
            )
            config_path = os.path.join(config_dir, "default.yaml")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        return config["default_config"]

    def _compute_stats(self, image_paths, transform):
        means, stds = [], []
        for img_path in image_paths:
            try:
                # Replace Image.open with read_image
                img = read_image(img_path)

                # Convert to float (0-1 range)
                img = img.float() / 255.0

                if transform:
                    img_tensor = transform(img)
                else:
                    img_tensor = img  # Already a tensor, no need for ToTensor

                means.append(img_tensor.mean(dim=(1, 2)).cpu().numpy())
                stds.append(img_tensor.std(dim=(1, 2)).cpu().numpy())
            except Exception as e:
                print(f"Warning: Failed to compute stats for {img_path}: {str(e)}")
                continue

        if not means:
            print(
                "Warning: No valid images for stats computation, using default mean/std"
            )
            return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

        return np.mean(means, axis=0), np.mean(stds, axis=0)


if __name__ == "__main__":
    # Example usage
    transform = get_transforms()
    # dataset = ComplexDataset("train", transforms=transform)
    stats_calculator = Compute_stats("train", transforms=transform)
    print(
        f"GASF Mean: {stats_calculator.gasf_mean}, GASF Std: {stats_calculator.gasf_std}"
    )
    print(
        f"GADF Mean: {stats_calculator.gadf_mean}, GADF Std: {stats_calculator.gadf_std}"
    )
    # print("mean and std in complex dataset class to check")
    # print(f"GASF Mean: {dataset.gasf_mean}, GASF Std: {dataset.gasf_std}")
    # print(f"GADF Mean: {dataset.gadf_mean}, GADF Std: {dataset.gadf_std}")

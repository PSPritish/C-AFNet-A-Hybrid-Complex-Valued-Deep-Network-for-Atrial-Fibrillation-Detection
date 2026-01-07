import yaml
import os
import torchvision.transforms as T
import torch


def get_transforms():
    """
    Get transforms with resize dimensions from config

    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    # Define path to config file
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "default.yaml",
    )

    # Load config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Extract image size from config
    default_config = config["default_config"]
    input_shape = default_config.get("data", {}).get("input_shape", [3, 224, 224])

    # The input_shape is [channels, height, width]
    # For resize, we need (height, width)
    image_size = (input_shape[1], input_shape[2])

    # Create and return the transforms
    return T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Resize(image_size),
        ]
    )


# For direct usage in other files
transforms = get_transforms()

if __name__ == "__main__":
    # Test the transforms
    print(f"Transforms: {transforms}")

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from tabulate import tabulate
import sys
import os


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(
    model,
    input_size=(3, 224, 224),
    batch_size=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Generate a comprehensive summary of a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to summarize
        input_size (tuple): Input dimensions excluding batch size (C, H, W)
        batch_size (int): Batch size for memory calculation
        device (str): Device to use for forward pass

    Returns:
        None: Prints summary to console
    """
    # Move model to specified device
    model = model.to(device)
    model.eval()

    # Check if model is complex-valued
    is_complex = any(torch.is_complex(p) for name, p in model.named_parameters())
    has_complex_layers = any("Complex" in type(m).__name__ for m in model.modules())
    model_type = "Complex-valued" if is_complex or has_complex_layers else "Real-valued"

    # Create a hook function to capture layer information
    summary = OrderedDict()
    hooks = []

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # For each module, store key info
            summary[module_idx] = OrderedDict()
            summary[module_idx]["name"] = class_name
            summary[module_idx]["id"] = id(module)

            # Handle different output types
            if isinstance(output, (list, tuple)):
                summary[module_idx]["output_shape"] = list(output[0].size())
            else:
                if isinstance(output, torch.Tensor):
                    summary[module_idx]["output_shape"] = list(output.size())
                else:
                    summary[module_idx]["output_shape"] = "?"

            # Count parameters
            params = 0
            trainable_params = 0
            for p_name, p in module.named_parameters(recurse=False):
                param_count = p.numel()
                params += param_count
                if p.requires_grad:
                    trainable_params += param_count

            summary[module_idx]["nb_params"] = params
            summary[module_idx]["trainable"] = trainable_params
            summary[module_idx]["non_trainable"] = params - trainable_params

        # Skip certain modules that are handled by their parent
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # Register hooks for all modules
    model.apply(register_hook)

    # Create a dummy input tensor
    x = torch.zeros(batch_size, *input_size, device=device)

    # Handle case for complex input
    if has_complex_layers:
        # Create a proper 3-channel complex tensor
        real_part = x[:, :3]  # First 3 channels as real
        imag_part = torch.zeros_like(real_part)  # Zeros as imaginary

        # Create complex tensor per channel
        channels = []
        for i in range(3):
            channels.append(torch.complex(real_part[:, i], imag_part[:, i]))

        # Stack to form [B, 3, H, W] complex tensor
        x = torch.stack(channels, dim=1)

    # Make a forward pass to trigger hooks
    try:
        model(x)
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Calculate memory usage
    input_size_bytes = np.prod(input_size) * batch_size * 4  # 4 bytes for float32
    if is_complex or has_complex_layers:
        input_size_bytes *= 2  # Complex values use twice the memory

    # Prepare table data
    table_data = []
    for layer in summary.values():
        if isinstance(layer["output_shape"], list):
            layer["output_shape"] = [batch_size] + layer["output_shape"][1:]
            output_shape_str = str(layer["output_shape"])
        else:
            output_shape_str = str(layer["output_shape"])

        row = [
            f"{layer['name']}-{layer['id']}"[-70:],
            output_shape_str,
            "{:,}".format(layer["nb_params"]),
        ]
        table_data.append(row)

    # Create a more readable version with just layer type and count
    readable_table = []
    layer_counts = {}
    for layer in summary.values():
        layer_name = layer["name"]
        if layer_name not in layer_counts:
            layer_counts[layer_name] = 1
        else:
            layer_counts[layer_name] += 1

    layer_idx = 1
    for layer in summary.values():
        if isinstance(layer["output_shape"], list):
            output_shape_str = str(layer["output_shape"])
        else:
            output_shape_str = str(layer["output_shape"])

        row = [
            f"{layer['name']}-{layer_idx}",
            output_shape_str,
            "{:,}".format(layer["nb_params"]),
        ]
        readable_table.append(row)
        layer_idx += 1

    # Print summary
    headers = ["Layer (type)", "Output Shape", "Param #"]
    print(tabulate(readable_table, headers=headers, tablefmt="grid"))

    # Print model summary
    print("\n" + "=" * 50)
    print(f"Model Type: {model_type}")
    print(f"Device: {device}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print(f"Estimated memory usage: {input_size_bytes / (1024 * 1024):.2f} MB")
    print("=" * 50)

    # Return summary dict for further processing if needed
    return summary


def test_model_summary(model_name="complex_resnet18"):
    """Test the model summary utility with different models"""
    from models.resnet import AFResNet
    from models.complex_resnet import complex_resnet18, complex_resnet34
    from data.dataloader import load_config

    config = load_config()

    if model_name == "resnet":
        model = AFResNet(config)
        input_size = (3, 128, 128)
    elif model_name == "complex_resnet18":
        # Set up complex model config
        if "data" not in config:
            config["data"] = {}
        config["data"]["input_shape"] = [3, 128, 128]  # 3 complex channels
        model = complex_resnet18(config)
        input_size = (3, 128, 128)
    elif model_name == "complex_resnet34":
        if "data" not in config:
            config["data"] = {}
        config["data"]["input_shape"] = [3, 128, 128]  # 3 complex channels
        model = complex_resnet34(config)
        input_size = (3, 128, 128)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"=== Model Summary for {model_name} ===")
    model_summary(model, input_size, device="cpu")

    # Also print total parameter count
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_model_summary(sys.argv[1])
    else:
        test_model_summary()

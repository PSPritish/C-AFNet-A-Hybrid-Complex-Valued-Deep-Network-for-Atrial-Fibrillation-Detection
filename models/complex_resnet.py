import torch
import torch.nn as nn
from models.architectures.complex_layers import (
    ComplexConv2d,
    ComplexNaiveBatchNorm2d,
    ComplexMaxPool2d,
    ComplexAdaptiveAvgPool2d,
    ComplexDropout,
)
from models.architectures.complex_activations import ModReLU, zReLU, ComplexCardioid


class ComplexBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ComplexBasicBlock, self).__init__()
        self.conv1 = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = ComplexNaiveBatchNorm2d(out_channels)
        self.relu1 = ModReLU(out_channels)  # For first two activations
        self.conv2 = ComplexConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = ComplexNaiveBatchNorm2d(out_channels)
        self.relu2 = ModReLU(out_channels)  # For the output after addition

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)  # Use the correct channel count

        return out


class ComplexBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ComplexBottleneck, self).__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = ComplexNaiveBatchNorm2d(out_channels)
        self.conv2 = ComplexConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = ComplexNaiveBatchNorm2d(out_channels)
        self.conv3 = ComplexConv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = ComplexNaiveBatchNorm2d(out_channels * self.expansion)
        self.relu1 = ModReLU(out_channels)  # For first two activations
        self.relu2 = ModReLU(out_channels * self.expansion)  # For final activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu1(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)  # Use the correct channel count

        return out


class ComplexResNet(nn.Module):
    def __init__(
        self, block, layers, num_classes=1, input_channels=2, zero_init_residual=False
    ):
        super(ComplexResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = ComplexConv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = ComplexNaiveBatchNorm2d(64)
        self.relu = ModReLU(64)
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = ComplexAdaptiveAvgPool2d((1, 1))
        # self.dropout = ComplexDropout(p=0.5)

        # For binary classification, use a single output with sigmoid
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                # Access the weights of the underlying real/imaginary convolutions
                nn.init.kaiming_normal_(
                    m.real_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.kaiming_normal_(
                    m.imag_conv.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, ComplexNaiveBatchNorm2d):
                # Make sure these attribute names match your implementation
                if hasattr(m, "weight_real"):
                    nn.init.constant_(m.weight_real, 1)
                    nn.init.constant_(m.weight_imag, 0)
                    nn.init.constant_(m.bias_real, 0)
                    nn.init.constant_(m.bias_imag, 0)
                # If using a different implementation with real_bn/imag_bn structure
                elif hasattr(m, "real_bn"):
                    nn.init.constant_(m.real_bn.weight, 1)
                    nn.init.constant_(m.real_bn.bias, 0)
                    nn.init.constant_(m.imag_bn.weight, 0)
                    nn.init.constant_(m.imag_bn.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ComplexBottleneck):
                    if hasattr(m.bn3, "weight_real"):
                        nn.init.constant_(m.bn3.weight_real, 0)
                        nn.init.constant_(m.bn3.weight_imag, 0)
                    elif hasattr(m.bn3, "real_bn"):
                        nn.init.constant_(m.bn3.real_bn.weight, 0)
                        nn.init.constant_(m.bn3.imag_bn.weight, 0)
                elif isinstance(m, ComplexBasicBlock):
                    if hasattr(m.bn2, "weight_real"):
                        nn.init.constant_(m.bn2.weight_real, 0)
                        nn.init.constant_(m.bn2.weight_imag, 0)
                    elif hasattr(m.bn2, "real_bn"):
                        nn.init.constant_(m.bn2.real_bn.weight, 0)
                        nn.init.constant_(m.bn2.imag_bn.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                ComplexConv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                ComplexNaiveBatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4 and x.shape[1] == 6:
            real = x[:, :3]
            imag = x[:, 3:]
            x = torch.stack((real, imag), dim=-1)  # (B, 3, H, W, 2)
        if not torch.is_complex(x) and x.dim() == 5 and x.size(-1) == 2:
            real_part = x[..., 0]
            imag_part = x[..., 1]
            x = torch.complex(real_part, imag_part)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.dropout(x)

        # Convert to magnitude for classification
        x = torch.abs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def complex_resnet18(config):
    """
    Create a complex-valued ResNet-18 model

    Args:
        config: Configuration dictionary

    Returns:
        ComplexResNet: Complex-valued ResNet-18 model
    """
    # Read from config consistently
    num_classes = config.get("model", {}).get("num_classes", 1)
    input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

    print(
        f"Creating complex ResNet-18 with {input_channels} complex channels and {num_classes} classes"
    )

    return ComplexResNet(
        ComplexBasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        input_channels=input_channels,
    )


def complex_resnet34(config):
    """
    Create a complex-valued ResNet-34 model

    Args:
        config: Configuration dictionary

    Returns:
        ComplexResNet: Complex-valued ResNet-34 model
    """
    # Read from config consistently
    num_classes = config.get("model", {}).get("num_classes", 1)
    input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

    print(
        f"Creating complex ResNet-34 with {input_channels} complex channels and {num_classes} classes"
    )

    return ComplexResNet(
        ComplexBasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        input_channels=input_channels,
    )


def complex_resnet50(config):
    """
    Create a complex-valued ResNet-50 model

    Args:
        config: Configuration dictionary

    Returns:
        ComplexResNet: Complex-valued ResNet-50 model
    """
    # Read from config consistently
    num_classes = config.get("model", {}).get("num_classes", 1)
    input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

    print(
        f"Creating complex ResNet-50 with {input_channels} complex channels and {num_classes} classes"
    )

    return ComplexResNet(
        ComplexBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        input_channels=input_channels,
    )

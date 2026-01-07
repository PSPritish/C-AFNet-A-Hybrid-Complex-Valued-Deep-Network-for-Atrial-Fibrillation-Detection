from matplotlib.pyplot import cla
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # Add this line

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None
    ):  # Renamed for consistency
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample  # Renamed for consistency

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # Not in-place
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights with Kaiming/He initialization
        self._initialize_weights()

    def _make_layer(self, block, num_of_residual_blocks, out_channels, stride=1):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers.append(
            block(self.in_channels, out_channels, stride, identity_downsample)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_of_residual_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        """Initialize weights according to the ResNet paper methodology"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize with ones and zeros as in the paper
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # He initialization for fully connected layer
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        # Optional: Zero-initialize the last BN in each residual branch
        # This improves training according to "Bag of Tricks" paper
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


def resnet18(input_channels=3, num_classes=1, config=None):
    """
    Create a standard ResNet-18 model

    Args:
        config: Configuration dictionary

    Returns:
        ResNet: Standard ResNet-18 model
    """
    # Read from config consistently if provided
    if config:
        num_classes = config.get("data", {}).get("num_classes", 1)
        input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

        print(
            f"Creating ResNet-18 with {input_channels} channels and {num_classes} classes"
        )
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channels, num_classes)


def resnet34(config=None):
    """
    Create a standard ResNet-34 model

    Args:
        config: Configuration dictionary

    Returns:
        ResNet: Standard ResNet-34 model
    """
    # Read from config consistently if provided
    if config:
        num_classes = config.get("data", {}).get("num_classes", 1)
        input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

        print(
            f"Creating ResNet-34 with {input_channels} channels and {num_classes} classes"
        )
    else:
        # Default parameters for backward compatibility
        num_classes = 1
        input_channels = 3

    return ResNet(BasicBlock, [3, 4, 6, 3], input_channels, num_classes)


def resnet50(config=None):
    """
    Create a standard ResNet-50 model

    Args:
        config: Configuration dictionary

    Returns:
        ResNet: Standard ResNet-50 model
    """
    # Read from config consistently if provided
    if config:
        num_classes = config.get("data", {}).get("num_classes", 1)
        input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

        print(
            f"Creating ResNet-50 with {input_channels} channels and {num_classes} classes"
        )
    else:
        # Default parameters for backward compatibility
        num_classes = 1
        input_channels = 3

    return ResNet(Bottleneck, [3, 4, 6, 3], input_channels, num_classes)


def resnet101(config=None):
    """
    Create a standard ResNet-101 model

    Args:
        config: Configuration dictionary

    Returns:
        ResNet: Standard ResNet-101 model
    """
    # Similar config handling as above
    if config:
        num_classes = config.get("data", {}).get("num_classes", 1)
        input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

        print(
            f"Creating ResNet-101 with {input_channels} channels and {num_classes} classes"
        )
    else:
        num_classes = 1
        input_channels = 3

    return ResNet(Bottleneck, [3, 4, 23, 3], input_channels, num_classes)


def resnet152(config=None):
    """
    Create a standard ResNet-152 model

    Args:
        config: Configuration dictionary

    Returns:
        ResNet: Standard ResNet-152 model
    """
    # Similar config handling as above
    if config:
        num_classes = config.get("data", {}).get("num_classes", 1)
        input_channels = config.get("data", {}).get("input_shape", [3, 224, 224])[0]

        print(
            f"Creating ResNet-152 with {input_channels} channels and {num_classes} classes"
        )
    else:
        num_classes = 1
        input_channels = 3

    return ResNet(Bottleneck, [3, 8, 36, 3], input_channels, num_classes)

from numpy import save
import torch
import torch.nn as nn

from models.architectures.complex_activations import ModReLU
from models.architectures.complex_layers import (
    ComplexConv2d,
    ComplexMaxPool2d,
    ComplexNaiveBatchNorm2d,
)
from models.C_ResNet import ComplexBasicBlock, ComplexBottleneck
from models.ResNet import BasicBlock, Bottleneck


class Inverted_C_AFNet(nn.Module):
    def __init__(
        self,
        normal_block,
        complex_block,
        layers,
        input_channels,
        num_classes,
        zero_init_residual=False,
    ):
        super(Inverted_C_AFNet, self).__init__()
        self.in_channels = 64

        self.conv1 = ComplexConv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = ComplexNaiveBatchNorm2d(64)
        self.relu = ModReLU(64)
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(complex_block, layers[0], 64)

        saved_channels = self.in_channels
        self.layer2_real = self._make_layer(normal_block, layers[1], 128, stride=2)
        self.in_channels = saved_channels
        self.layer2_imag = self._make_layer(normal_block, layers[1], 128, stride=2)

        self.layer3 = self._make_layer(complex_block, layers[2], 256, stride=2)

        saved_channels = self.in_channels
        self.layer4_real = self._make_layer(normal_block, layers[3], 512, stride=2)
        self.in_channels = saved_channels
        self.layer4_imag = self._make_layer(normal_block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * complex_block.expansion, num_classes)

        # Initialize weights with Kaiming/He initialization
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
            elif isinstance(m, nn.Conv2d):
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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, num_of_residual_blocks, out_channels, stride=1):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Use different downsample types based on block type
            if issubclass(block, ComplexBasicBlock) or issubclass(
                block, ComplexBottleneck
            ):
                # For complex blocks, use complex layers
                identity_downsample = nn.Sequential(
                    ComplexConv2d(
                        self.in_channels,
                        out_channels * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    ComplexNaiveBatchNorm2d(out_channels * block.expansion),
                )
            else:
                # For regular blocks, use regular layers
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
        # Fix Captum's possible stripped batch
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Handle Captum 6-channel format
        if x.dim() == 4 and x.shape[1] == 6:
            real = x[:, :3]
            imag = x[:, 3:]
            x = torch.stack((real, imag), dim=-1)  # (B, 3, H, W, 2)

        if x.dim() == 5 and x.size(-1) == 2:
            # Complex tensor with last dimension for real/imaginary parts
            real_part = x[..., 0]
            imag_part = x[..., 1]
            x = torch.complex(real_part, imag_part)

        if x.dtype != torch.complex64:
            raise ValueError(
                f"Input should be a complex tensor, got {x.dtype}. "
                "Ensure the input is in complex format."
            )
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)

        out_real = self.layer2_real(out.real)
        out_imag = self.layer2_imag(out.imag)
        out = torch.complex(out_real, out_imag)

        out = self.layer3(out)

        out_real = self.layer4_real(out.real)
        out_imag = self.layer4_imag(out.imag)
        out = torch.complex(out_real, out_imag)

        out = self.avgpool(out)

        out = torch.abs(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


def inverted_c_afnet(input_channels=3, num_classes=1, zero_init_residual=False):
    return Inverted_C_AFNet(
        normal_block=BasicBlock,
        complex_block=ComplexBasicBlock,
        layers=[2, 2, 2, 2],
        input_channels=input_channels,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
    )
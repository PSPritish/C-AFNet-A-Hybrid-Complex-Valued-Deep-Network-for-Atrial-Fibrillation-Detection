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


class Grouped_C_AFNet(nn.Module):
    def __init__(
        self,
        normal_block,
        complex_block,
        layers,
        input_channels,
        num_classes,
        zero_init_residual=False,
    ):
        super(Grouped_C_AFNet, self).__init__()
        self.in_channels = 64

        # Initialize the first convolutional layer for complex input
        self.initial_layer = nn.Sequential(
            ComplexConv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            ComplexNaiveBatchNorm2d(64),
            ModReLU(64, inplace=False),
        )
        self.layer1 = self._make_layer(complex_block, layers[0], 64)
        self.layer2 = self._make_layer(complex_block, layers[1], 128, stride=2)
        saved_channels = self.in_channels
        self.layer3_mag = self._make_layer(normal_block, layers[2], 256, stride=2)
        self.in_channels = saved_channels
        self.layer3_phase = self._make_layer(normal_block, layers[2], 256, stride=2)
        saved_channels = self.in_channels
        self.layer4_mag = self._make_layer(normal_block, layers[3], 512, stride=2)
        self.in_channels = saved_channels
        self.layer4_phase = self._make_layer(normal_block, layers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_gate = nn.Sequential(
            nn.Conv2d(
                2 * self.in_channels, self.in_channels, kernel_size=1
            ),  # 1024 → 512
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, 2, kernel_size=1),  # 512 → 2 (w_mag, w_phase)
            nn.Softmax(dim=1),  # normalize: w_mag + w_phase = 1
        )
        self.fc = nn.Linear(self.in_channels, num_classes)
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

        if x.dim() == 5:
            real_part = x[..., 0]
            imag_part = x[..., 1]
            x = torch.complex(real_part, imag_part)

        if x.dtype != torch.complex64:
            raise ValueError(
                f"Input tensor must be of type torch.complex64, but got {x.dtype}"
            )
        out = self.initial_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)

        mag = torch.abs(out)
        phase = torch.angle(out)
        phase = (phase + torch.pi) / (2 * torch.pi)

        out_mag = self.layer3_mag(mag)
        out_phase = self.layer3_phase(phase)
        out_mag = self.layer4_mag(out_mag)
        out_phase = self.layer4_phase(out_phase)
        out_mag = self.avgpool(out_mag)
        out_phase = self.avgpool(out_phase)

        out = torch.cat((out_mag, out_phase), dim=1)
        attn_weights = self.attn_gate(out)
        w_mag = attn_weights[:, 0:1]
        w_phase = attn_weights[:, 1:2]
        fused = w_mag * out_mag + w_phase * out_phase
        fused = torch.flatten(fused, start_dim=1)

        out = self.fc(fused)

        return out


def grouped_c_afnet(
    input_channels=3, num_classes=1, zero_init_residual=False
):
    return Grouped_C_AFNet(
        normal_block=BasicBlock,
        complex_block=ComplexBasicBlock,
        layers=[2, 2, 2, 2],
        input_channels=input_channels,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
    )

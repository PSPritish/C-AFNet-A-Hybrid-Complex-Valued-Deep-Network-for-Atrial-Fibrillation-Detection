import torch
import torch.nn as nn
from torch import Tensor


class ModReLU(nn.Module):
    def __init__(self, num_features, inplace=False):
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        # Reshape bias for proper broadcasting across spatial dimensions
        # If x has shape [batch, channels, height, width]
        # This gives bias shape [1, channels, 1, 1]
        bias_reshaped = self.bias.view(1, -1, 1, 1)

        out_real = self.relu(x.real + bias_reshaped)
        out_imag = self.relu(x.imag + bias_reshaped)
        return torch.complex(out_real, out_imag)


class ComplexConv2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(ComplexConv2d, self).__init__()
        self.real_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.imag_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    def forward(self, x):
        # Handle [batch, channels, height, width, 2] format
        if not torch.is_complex(x) and x.dim() == 5 and x.size(-1) == 2:
            real_part = x[..., 0]
            imag_part = x[..., 1]
            x = torch.complex(real_part, imag_part)

        # Ensure dtype is complex64
        if x.dtype != torch.complex64:
            x = x.to(dtype=torch.complex64)

        real = self.real_conv(x.real)
        imag = self.imag_conv(x.imag)
        output = torch.complex(real, imag)
        return output

"""
    R-AFNet (Ablated Real Version)
    
    This model serves as the "ablated real" version of the backbone. While it maintains 
    architectural symmetry with C-AFNet (using the same staging and class names like 
    ComplexConv2d and ModReLU), all complex-valued interactions have been explicitly removed:

    1. ComplexConv2d: The cross-terms required for complex multiplication (ac - bd, ad + bc) 
       are disabled. It strictly performs parallel real-valued convolutions: 
       Real_out = Conv(Real_in) and Imag_out = Conv(Imag_in).

    2. ModReLU: The magnitude-based coupling is removed. It functions as two independent 
       standard ReLUs applied separately to the real and imaginary components.

    Result: The network effectively operates as two independent, parallel real-valued 
    streams (Real and Imaginary) that do not exchange information until the final fusion.
"""
class ComplexMaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

        self.max_pool = torch.nn.MaxPool2d(
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode,
        )

    def forward(self, x):
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor, Got {x.dtype}")

        return (self.max_pool(x.real)).type(torch.complex64) + 1j * (
            self.max_pool(x.imag)
        ).type(torch.complex64)


class ComplexAvgPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super(ComplexAvgPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

        self.avg_pool = torch.nn.AvgPool2d(
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    def forward(self, x):
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")

        return (self.avg_pool(x.real)).type(torch.complex64) + 1j * (
            self.avg_pool(x.imag)
        ).type(torch.complex64)


class ComplexAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, input):
        return (self.adaptive_pool(input.real)).type(torch.complex64) + 1j * (
            self.adaptive_pool(input.imag)
        ).type(torch.complex64)


class ComplexDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.real_drop = torch.nn.Dropout(self.p)
        self.imag_drop = torch.nn.Dropout(self.p)

    def forward(self, input):
        return (self.real_drop(input.real)).type(torch.complex64) + 1j * (
            self.imag_drop(input.imag)
        ).type(torch.complex64)


class ComplexNaiveBatchNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device

        self.real_bn = torch.nn.BatchNorm2d(
            self.num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats,
        )
        self.imag_bn = torch.nn.BatchNorm2d(
            self.num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats,
        )

    def forward(self, input):
        if not input.is_complex():
            raise ValueError(f"Input should be complex, Got {input.dtype}")

        return (self.real_bn(input.real)).type(torch.complex64) + 1j * (
            self.imag_bn(input.imag)
        ).type(torch.complex64)


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
        self.relu1 = ModReLU(out_channels)
        self.conv2 = ComplexConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = ComplexNaiveBatchNorm2d(out_channels)
        self.relu2 = ModReLU(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)

        return out


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


class R_AFNet(nn.Module):
    def __init__(
        self,
        normal_block,
        complex_block,
        layers,
        input_channels,
        num_classes,
        zero_init_residual=False,
    ):
        super(R_AFNet, self).__init__()
        self.in_channels = 64

        # Initialize the first convolutional layer for complex input components (Real)
        self.conv1_real = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )

        # Initialize the first convolutional layer for complex input components (Imag)
        self.conv1_imag = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )

        saved_channels = self.in_channels
        self.layer1_real = self._make_layer(normal_block, layers[0], 64)
        self.in_channels = saved_channels
        self.layer1_imag = self._make_layer(normal_block, layers[0], 64)

        self.layer2 = self._make_layer(complex_block, layers[1], 128, stride=2)

        saved_channels = self.in_channels
        self.layer3_real = self._make_layer(normal_block, layers[2], 256, stride=2)
        self.in_channels = saved_channels
        self.layer3_imag = self._make_layer(normal_block, layers[2], 256, stride=2)

        self.layer4 = self._make_layer(complex_block, layers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * complex_block.expansion, num_classes)

        # Initialize weights with Kaiming/He initialization
        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                nn.init.kaiming_normal_(
                    m.real_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.kaiming_normal_(
                    m.imag_conv.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, ComplexNaiveBatchNorm2d):
                if hasattr(m, "real_bn"):
                    nn.init.constant_(m.real_bn.weight, 1)
                    nn.init.constant_(m.real_bn.bias, 0)
                    nn.init.constant_(m.imag_bn.weight, 0)
                    nn.init.constant_(m.imag_bn.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                # Note: ComplexBottleneck and Bottleneck were not defined in the source,
                # checks removed to ensure class portability.
                if isinstance(m, ComplexBasicBlock):
                    if hasattr(m.bn2, "real_bn"):
                        nn.init.constant_(m.bn2.real_bn.weight, 0)
                        nn.init.constant_(m.bn2.imag_bn.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, num_of_residual_blocks, out_channels, stride=1):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Use different downsample types based on block type
            if issubclass(block, ComplexBasicBlock):
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
        else:
            real_part = x.real
            imag_part = x.imag

        out_real = self.conv1_real(real_part)
        out_imag = self.conv1_imag(imag_part)

        out_real = self.layer1_real(out_real)
        out_imag = self.layer1_imag(out_imag)

        out = torch.complex(out_real, out_imag)

        out = self.layer2(out)

        out_real = self.layer3_real(out.real)
        out_imag = self.layer3_imag(out.imag)
        out = torch.complex(out_real, out_imag)

        out = self.layer4(out)
        out = self.avgpool(out)

        out = torch.abs(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


def r_afnet(input_channels=3, num_classes=1):
    """
    Returns an instance of the R_AFNet model.
    Default configuration uses ResNet18-like layers [2, 2, 2, 2].
    """
    return R_AFNet(
        normal_block=BasicBlock,
        complex_block=ComplexBasicBlock,
        layers=[2, 2, 2, 2],
        input_channels=input_channels,
        num_classes=num_classes,
        zero_init_residual=False,
    )

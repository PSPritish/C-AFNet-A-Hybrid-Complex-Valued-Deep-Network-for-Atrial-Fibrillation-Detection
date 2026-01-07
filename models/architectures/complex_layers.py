import torch
import torch.nn as nn


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) + 1j * (
        fr(input.imag) + fi(input.real)
    ).type(dtype)


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
        # Continue with normal processing
        if x.dtype != torch.complex64:
            x = x.to(dtype=torch.complex64)
        real = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag = self.real_conv(x.imag) + self.imag_conv(x.real)
        output = torch.complex(real, imag)
        return output


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

        # check if the input is complex
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
        # check if the input is a complex tensor
        if not input.is_complex():
            raise ValueError(f"Input should be complex, Got {input.dtype}")

        return (self.real_bn(input.real)).type(torch.complex64) + 1j * (
            self.imag_bn(input.imag)
        ).type(torch.complex64)

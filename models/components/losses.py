import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class ComplexMagnitudeAndPhaseLoss(nn.Module):
    """Loss that considers both magnitude and phase of complex outputs"""

    def __init__(self, phase_weight=0.5, reduction="mean"):
        super(ComplexMagnitudeAndPhaseLoss, self).__init__()
        self.reduction = reduction
        self.phase_weight = phase_weight  # Weight for the phase component

    def forward(self, complex_outputs, targets):
        # Handle both standard complex tensors and tensors with extra dimension
        if not torch.is_complex(complex_outputs) and complex_outputs.dim() > targets.dim():
            # For DataParallel case with shape [batch, features, 2]
            # Extract magnitude manually from real and imaginary parts
            real_part = complex_outputs[..., 0] 
            imag_part = complex_outputs[..., 1]
            magnitude = torch.sqrt(real_part**2 + imag_part**2)
            
            # Calculate phase manually
            phase = torch.atan2(imag_part, real_part)
        else:
            # Standard complex tensor
            magnitude = torch.abs(complex_outputs)
            phase = torch.angle(complex_outputs)
    
        # Apply loss on correctly calculated magnitude
        magnitude_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)(
            magnitude, targets
        )

        # Phase component (directional information)
        # For positive samples, phase should be clustered (low variance)
        # For negative samples, phase is less important

        # Target-dependent phase regularization
        positive_samples = (targets == 1).float()
        phase_consistency = (1 - torch.cos(phase)) * positive_samples

        if self.reduction == "mean":
            phase_loss = phase_consistency.mean()
        elif self.reduction == "sum":
            phase_loss = phase_consistency.sum()

        # Combined loss
        return magnitude_loss + self.phase_weight * phase_loss


class ComplexDiceLoss(nn.Module):
    """Dice loss for complex-valued outputs (after magnitude)"""

    def __init__(self, smooth=1.0):
        super(ComplexDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class ComplexFocalLoss(nn.Module):
    """Focal loss specifically designed for complex models"""

    def __init__(self, alpha=1.0, gamma=2.0, magnitude_only=True, reduction="mean"):
        super(ComplexFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.magnitude_only = magnitude_only

    def forward(self, inputs, targets):
        # If we're only using magnitude (default)
        if self.magnitude_only or not torch.is_complex(inputs):
            # This is for when magnitude was already taken in the model
            BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

            if self.reduction == "mean":
                return F_loss.mean()
            elif self.reduction == "sum":
                return F_loss.sum()
            else:
                return F_loss
        else:
            # This is for when we have complex outputs directly
            # Take magnitude first
            magnitude = torch.abs(inputs)
            BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(magnitude, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

            if self.reduction == "mean":
                return F_loss.mean()
            elif self.reduction == "sum":
                return F_loss.sum()
            else:
                return F_loss

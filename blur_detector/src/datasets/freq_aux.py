"""
Frequency-domain auxiliary channel (Laplacian magnitude) for blur detection.

Given a 3-channel RGB tensor (already normalized via ImageNet stats), computes
the per-pixel Laplacian magnitude on the grayscale of the ORIGINAL image and
appends it as a 4th channel. The auxiliary channel is standardized per image
so the CNN can learn a threshold rather than an absolute scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard 3x3 Laplacian kernel (8-connected)
_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


# ImageNet normalization constants (to invert / estimate grayscale from normalized tensor)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_RGB_TO_GRAY = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)


class FreqAuxPreprocessor(nn.Module):
    """Prepend Laplacian-magnitude as a 4th channel to a normalized RGB batch.

    Input:  (N, 3, H, W) — ImageNet-normalized RGB.
    Output: (N, 4, H, W) — first 3 channels unchanged; 4th channel is the
            per-image standardized Laplacian magnitude of the grayscale image.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", _LAPLACIAN_KERNEL)
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)
        self.register_buffer("rgb2gray", _RGB_TO_GRAY)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recover original [0,1] RGB from normalized tensor
        rgb = x * self.std + self.mean
        # Grayscale (luminance)
        gray = (rgb * self.rgb2gray).sum(dim=1, keepdim=True)
        # Laplacian magnitude
        lap = F.conv2d(gray, self.kernel, padding=1)
        mag = lap.abs()
        # Per-image standardization: zero mean, unit std
        N = mag.shape[0]
        flat = mag.view(N, -1)
        mu = flat.mean(dim=1, keepdim=True).view(N, 1, 1, 1)
        sigma = flat.std(dim=1, keepdim=True).view(N, 1, 1, 1).clamp(min=1e-6)
        mag_std = (mag - mu) / sigma
        return torch.cat([x, mag_std], dim=1)


class FreqAuxModel(nn.Module):
    """Wrap a 4-channel-input backbone with an in-graph FreqAux preprocessor."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.freq_aux = FreqAuxPreprocessor()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.freq_aux(x))

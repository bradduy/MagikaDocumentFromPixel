import torch
import torch.nn as nn
import torchvision.models as tvm


NUM_CLASSES = 2


def build_model(
    backbone: str = "mobilenet_v3_small",
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        m = _mobilenet_v3_small(pretrained)
    elif backbone == "mobilenet_v3_large":
        m = _mobilenet_v3_large(pretrained)
    elif backbone == "efficientnet_b0":
        m = _efficientnet_b0(pretrained)
    elif backbone == "tiny_cnn":
        m = TinyCNN()
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    if in_channels != 3:
        _expand_first_conv(m, in_channels)
    return m


def _expand_first_conv(model: nn.Module, in_channels: int) -> None:
    """Replace the first conv to accept ``in_channels`` input channels while
    preserving pretrained RGB weights on the first 3 channels. Extra channels
    are initialized to the mean of the RGB kernels, scaled by 0.1 — a common
    warm-start for auxiliary-channel extension.
    """
    first = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            first = (name, mod)
            break
    if first is None:
        raise RuntimeError("No Conv2d found")
    name, conv = first
    new_conv = nn.Conv2d(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :3].copy_(conv.weight)
        if in_channels > 3:
            extra = conv.weight.mean(dim=1, keepdim=True) * 0.1
            new_conv.weight[:, 3:].copy_(extra.expand(-1, in_channels - 3, -1, -1))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    # Attach new conv in place
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_conv
    else:
        setattr(parent, last, new_conv)


def _mobilenet_v3_small(pretrained: bool) -> nn.Module:
    weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = tvm.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def _mobilenet_v3_large(pretrained: bool) -> nn.Module:
    weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = tvm.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def _efficientnet_b0(pretrained: bool) -> nn.Module:
    weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


class TinyCNN(nn.Module):
    """Magika-style 4-layer CNN, ~0.5M params, no pretrained weights needed."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(3, 16, stride=2),
            _conv_bn_relu(16, 32, stride=2),
            _conv_bn_relu(32, 64, stride=2),
            _conv_bn_relu(64, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

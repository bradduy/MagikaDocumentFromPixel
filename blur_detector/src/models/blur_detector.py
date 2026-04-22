import torch
import torch.nn as nn
import torchvision.models as tvm


NUM_CLASSES = 2


def build_model(backbone: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        return _mobilenet_v3_small(pretrained)
    if backbone == "mobilenet_v3_large":
        return _mobilenet_v3_large(pretrained)
    if backbone == "efficientnet_b0":
        return _efficientnet_b0(pretrained)
    if backbone == "tiny_cnn":
        return TinyCNN()
    raise ValueError(f"Unknown backbone: {backbone}")


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

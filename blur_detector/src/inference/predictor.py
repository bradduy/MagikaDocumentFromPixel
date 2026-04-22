from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


LABEL_MAP = {0: "sharp", 1: "blurred"}


@dataclass
class BlurPrediction:
    label: str          # "sharp" | "blurred" | "uncertain"
    confidence: float
    prob_sharp: float
    prob_blurred: float


def _build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class BlurPredictor:
    """
    Confidence-aware inference wrapper. Returns "uncertain" when max softmax
    is below ``confidence_threshold``.

    For maximum accuracy, pass a list of scales to enable multi-scale TTA —
    the production champion recipe (F1=0.9745 on GoPro test) uses scales
    [320, 384, 448] with the MobileNetV3-Large 384px checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: Union[int, Iterable[int]] = 384,
        confidence_threshold: float = 0.60,
        blur_prob_threshold: float = 0.50,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.blur_prob_threshold = blur_prob_threshold

        # Accept single int (no TTA) or list of ints (multi-scale TTA)
        if isinstance(image_size, int):
            self.scales: List[int] = [image_size]
        else:
            self.scales = list(image_size)
        self.transforms = [_build_transform(s) for s in self.scales]

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> BlurPrediction:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Average softmax across all configured scales
        accumulated = torch.zeros(2, device=self.device)
        for transform in self.transforms:
            tensor = transform(image).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(tensor), dim=1).squeeze(0)
            accumulated += probs
        probs = accumulated / len(self.transforms)

        p_sharp = probs[0].item()
        p_blurred = probs[1].item()
        confidence = max(p_sharp, p_blurred)

        if confidence < self.confidence_threshold:
            label = "uncertain"
        elif p_blurred >= self.blur_prob_threshold:
            label = "blurred"
        else:
            label = "sharp"

        return BlurPrediction(
            label=label,
            confidence=round(confidence, 4),
            prob_sharp=round(p_sharp, 4),
            prob_blurred=round(p_blurred, 4),
        )

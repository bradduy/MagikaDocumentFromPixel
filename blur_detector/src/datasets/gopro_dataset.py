from pathlib import Path
from typing import Optional, Tuple
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


LABEL_SHARP = 0
LABEL_BLURRED = 1


def _build_transforms(image_size: int, augment: bool, color_mode: str, aug_level: str = "medium") -> T.Compose:
    to_tensor = [T.ToTensor()]
    if color_mode == "RGB":
        to_tensor.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if not augment:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            *to_tensor,
        ])

    if aug_level == "light":
        return T.Compose([
            T.Resize((image_size + 16, image_size + 16)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            *to_tensor,
        ])
    if aug_level == "medium":
        return T.Compose([
            T.Resize((image_size + 16, image_size + 16)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            *to_tensor,
        ])
    if aug_level == "strong":
        # RandomResizedCrop: tests scale robustness; ColorJitter + grayscale; erasing
        # Avoid blur-augments — blur is the label signal.
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.1),
            *to_tensor,
            T.RandomErasing(p=0.25, scale=(0.02, 0.08)),
        ])
    raise ValueError(f"Unknown aug_level: {aug_level}")


class GoProDataset(Dataset):
    """
    Strategy A: each blur image → label 1, each sharp image → label 0.
    Supports both the flat layout and the official GoPro Large per-scene layout.

    Flat layout (legacy):
        gopro_root/
          train/blur/*.png   train/sharp/*.png
          test/blur/*.png    test/sharp/*.png

    Per-scene layout (official GoPro Large zip):
        gopro_root/
          train/<scene>/{blur, blur_gamma, sharp}/*.png
          test/<scene>/{blur, blur_gamma, sharp}/*.png
    """

    def __init__(
        self,
        gopro_root: str,
        split: str = "train",
        image_size: int = 128,
        color_mode: str = "RGB",
        augment: bool = True,
        val_fraction: float = 0.1,
        val_split: bool = False,
        seed: int = 42,
        aug_level: str = "medium",
        include_blur_gamma: bool = False,
    ):
        assert split in ("train", "val", "test")
        self.color_mode = color_mode
        self.transform = _build_transforms(
            image_size,
            augment=(augment and split == "train"),
            color_mode=color_mode,
            aug_level=aug_level,
        )

        root = Path(gopro_root)
        raw_split = "test" if split == "test" else "train"
        split_root = root / raw_split

        def _collect(subdir_name: str) -> list[Path]:
            flat = split_root / subdir_name
            if flat.is_dir():
                return sorted(
                    p for ext in ("*.png", "*.jpg", "*.jpeg")
                    for p in flat.glob(f"**/{ext}")
                )
            return sorted(
                p for ext in ("*.png", "*.jpg", "*.jpeg")
                for p in split_root.glob(f"*/{subdir_name}/{ext}")
            )

        blur_images = _collect("blur")
        sharp_images = _collect("sharp")

        samples: list[Tuple[Path, int]] = (
            [(p, LABEL_BLURRED) for p in blur_images] +
            [(p, LABEL_SHARP) for p in sharp_images]
        )

        if include_blur_gamma and split != "test":
            gamma_images = _collect("blur_gamma")
            samples.extend((p, LABEL_BLURRED) for p in gamma_images)

        if split in ("train", "val"):
            rng = random.Random(seed)
            rng.shuffle(samples)
            n_val = int(len(samples) * val_fraction)
            if split == "val":
                samples = samples[:n_val]
            else:
                samples = samples[n_val:]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path)
        if self.color_mode == "RGB":
            img = img.convert("RGB")
        else:
            img = img.convert("L").convert("RGB")  # grayscale as 3-ch for backbone compat
        return self.transform(img), label

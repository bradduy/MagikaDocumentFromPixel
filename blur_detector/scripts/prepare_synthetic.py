"""
Generate a synthetic sharp/blur dataset from BSDS500 images.
Downloads BSDS500 (~125MB), applies Gaussian + motion blur kernels,
writes to the same gopro-compatible layout:
    data/gopro/train/sharp/  data/gopro/train/blur/
    data/gopro/test/sharp/   data/gopro/test/blur/

Usage:
    python blur_detector/scripts/prepare_synthetic.py --data_root blur_detector/data/gopro
"""

import argparse
import math
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BSDS_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
TRAIN_RATIO = 0.85
SEED = 42

# --- blur kernel helpers ---

def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(img: Image.Image, size: int, angle: float) -> Image.Image:
    """Apply linear motion blur via numpy convolution."""
    k = np.zeros((size, size), dtype=np.float32)
    cx = size // 2
    radian = math.radians(angle)
    for i in range(size):
        offset = i - cx
        x = cx + round(offset * math.cos(radian))
        y = cx + round(offset * math.sin(radian))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    s = k.sum()
    if s > 0:
        k /= s

    arr = np.array(img, dtype=np.float32)
    from scipy.ndimage import convolve
    if arr.ndim == 3:
        blurred = np.stack([convolve(arr[:, :, c], k) for c in range(arr.shape[2])], axis=2)
    else:
        blurred = convolve(arr, k)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def apply_random_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    blur_type = rng.choice(["gaussian", "motion"])
    if blur_type == "gaussian":
        radius = rng.uniform(2.0, 6.0)
        return gaussian_blur(img, radius)
    else:
        size = rng.choice([11, 15, 21, 27])
        angle = rng.uniform(0, 180)
        try:
            return motion_blur(img, size, angle)
        except ImportError:
            return gaussian_blur(img, rng.uniform(2.0, 5.0))


# --- download ---

def download_bsds(dest: Path):
    tgz = dest / "BSR_bsds500.tgz"
    if not tgz.exists():
        print(f"Downloading BSDS500 from {BSDS_URL} ...")
        urllib.request.urlretrieve(BSDS_URL, tgz)
        print("Download complete.")
    else:
        print("Archive already exists, skipping download.")

    extracted = dest / "BSR"
    if not extracted.exists():
        print("Extracting ...")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(dest)
        print("Extraction complete.")
    return extracted


# --- main ---

def generate_split(images: list[Path], split_dir: Path, rng: random.Random):
    sharp_dir = split_dir / "sharp"
    blur_dir = split_dir / "blur"
    sharp_dir.mkdir(parents=True, exist_ok=True)
    blur_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(images):
        img = Image.open(src).convert("RGB")
        stem = f"{i:05d}"
        img.save(sharp_dir / f"{stem}.png")
        apply_random_blur(img, rng).save(blur_dir / f"{stem}.png")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="blur_detector/data/gopro")
    parser.add_argument("--bsds_cache", default="blur_detector/data/bsds_cache")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    bsds_cache = Path(args.bsds_cache)
    bsds_cache.mkdir(parents=True, exist_ok=True)

    extracted = download_bsds(bsds_cache)
    images_root = extracted / "BSDS500" / "data" / "images"

    all_images: list[Path] = []
    for subset in ("train", "val", "test"):
        all_images.extend((images_root / subset).glob("*.jpg"))

    rng = random.Random(SEED)
    rng.shuffle(all_images)
    n_train = int(len(all_images) * TRAIN_RATIO)
    train_imgs = all_images[:n_train]
    test_imgs = all_images[n_train:]

    print(f"\nGenerating train split ({len(train_imgs)} images)...")
    generate_split(train_imgs, data_root / "train", rng)

    print(f"\nGenerating test split ({len(test_imgs)} images)...")
    generate_split(test_imgs, data_root / "test", rng)

    print(f"\nDone. Dataset written to {data_root}")
    print(f"  train/sharp: {len(train_imgs)}  train/blur: {len(train_imgs)}")
    print(f"  test/sharp:  {len(test_imgs)}   test/blur:  {len(test_imgs)}")


if __name__ == "__main__":
    main()

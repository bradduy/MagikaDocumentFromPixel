"""
Validate that the GoPro dataset is in the expected layout.
Usage:
    python scripts/prepare_gopro.py --gopro_root data/gopro
"""

import argparse
from pathlib import Path


def check_split(root: Path, split: str):
    blur_dir = root / split / "blur"
    sharp_dir = root / split / "sharp"
    ok = True

    for d in (blur_dir, sharp_dir):
        if not d.is_dir():
            print(f"  MISSING: {d}")
            ok = False
        else:
            count = sum(len(list(d.glob(f"**/{ext}"))) for ext in ("*.png", "*.jpg", "*.jpeg"))
            print(f"  {d}  →  {count} images")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gopro_root", default="data/gopro", help="Path to GoPro dataset root")
    args = parser.parse_args()

    root = Path(args.gopro_root)
    if not root.exists():
        print(f"Dataset root not found: {root}")
        print("Download GoPro from https://seungjunnah.github.io/Datasets/gopro and extract here.")
        raise SystemExit(1)

    all_ok = True
    for split in ("train", "test"):
        print(f"\n[{split}]")
        all_ok = check_split(root, split) and all_ok

    if all_ok:
        print("\nDataset looks good — ready to train.")
    else:
        print("\nFix missing directories before training.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
Validate that the GoPro dataset is in the expected layout.
Usage:
    python scripts/prepare_gopro.py --gopro_root data/gopro
"""

import argparse
from pathlib import Path


def check_split(root: Path, split: str):
    """Supports both flat layout (split/blur, split/sharp) and the official
    GoPro Large per-scene layout (split/<scene>/{blur,blur_gamma,sharp})."""
    split_root = root / split
    if not split_root.is_dir():
        print(f"  MISSING: {split_root}")
        return False

    ok = True
    for subdir in ("blur", "sharp"):
        flat = split_root / subdir
        if flat.is_dir():
            count = sum(len(list(flat.glob(f"**/{ext}"))) for ext in ("*.png", "*.jpg", "*.jpeg"))
            print(f"  {flat}  →  {count} images  (flat layout)")
        else:
            files = [
                p for ext in ("*.png", "*.jpg", "*.jpeg")
                for p in split_root.glob(f"*/{subdir}/{ext}")
            ]
            scenes = sorted({p.parent.parent.name for p in files})
            if files:
                print(f"  {split_root}/*/{subdir}  →  {len(files)} images across {len(scenes)} scenes  (per-scene)")
            else:
                print(f"  MISSING {subdir}: not found under {split_root}")
                ok = False
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

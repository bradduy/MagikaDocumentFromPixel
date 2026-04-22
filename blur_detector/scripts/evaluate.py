"""
Evaluate a trained blur detector checkpoint on the GoPro test split.
Usage:
    python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.gopro_dataset import GoProDataset
from src.models.blur_detector import build_model
from src.utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="blur_detector/configs/config.yaml")
    parser.add_argument("--checkpoint", default="blur_detector/outputs/checkpoints/best.pt")
    parser.add_argument("--output_json", default="blur_detector/outputs/eval_results.json")
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = GoProDataset(
        gopro_root=str(ROOT / cfg["data"]["gopro_root"]),
        split="test",
        image_size=cfg["data"]["image_size"],
        color_mode=cfg["data"]["color_mode"],
        augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)
    print(f"Test set: {len(test_ds)} samples")

    model = build_model(backbone=cfg["model"]["backbone"], pretrained=False)
    model.load_state_dict(torch.load(ROOT / args.checkpoint, map_location=device))
    model = model.to(device).eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print(f"\n{metrics}")

    results = {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "confusion_matrix": metrics.confusion.tolist(),
    }
    out_path = ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

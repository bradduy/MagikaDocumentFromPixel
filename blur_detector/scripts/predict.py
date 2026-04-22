"""
Run the production blur detector on one or more images.
Defaults match the F1=0.9745 champion recipe.

Usage:
    python blur_detector/scripts/predict.py --checkpoint blur_detector/outputs/checkpoints/exp29_large_384_gamma/best.pt image1.jpg image2.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.blur_detector import build_model
from src.inference.predictor import BlurPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="Paths to images to classify")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="mobilenet_v3_large")
    p.add_argument("--scales", default="320,384,448",
                   help="Comma-separated TTA scales (default = champion recipe)")
    p.add_argument("--confidence_threshold", type=float, default=0.60)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(args.device)
    model = build_model(backbone=args.backbone, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    scales = [int(s) for s in args.scales.split(",")]
    predictor = BlurPredictor(
        model=model,
        image_size=scales,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )

    for path in args.images:
        pred = predictor.predict(path)
        print(json.dumps({
            "file": path,
            "label": pred.label,
            "confidence": pred.confidence,
            "prob_sharp": pred.prob_sharp,
            "prob_blurred": pred.prob_blurred,
        }))


if __name__ == "__main__":
    main()

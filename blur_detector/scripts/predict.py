"""
Run the production blur detector on one or more images.
Defaults match the F1=0.9803 champion recipe (Freq-Aux + 5-scale TTA).

Usage:
    python blur_detector/scripts/predict.py --checkpoint blur_detector/outputs/checkpoints/freq_aux/best.pt --freq_aux image1.jpg image2.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.freq_aux import FreqAuxModel
from src.models.blur_detector import build_model
from src.inference.predictor import BlurPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="Paths to images to classify")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="mobilenet_v3_large")
    p.add_argument("--scales", default="256,320,384,448,512",
                   help="Comma-separated TTA scales (default = champion recipe)")
    p.add_argument("--freq_aux", action="store_true",
                   help="Wrap backbone with FreqAuxModel (4-channel input: RGB + Laplacian). "
                        "Required when loading the freq_aux champion checkpoint.")
    p.add_argument("--confidence_threshold", type=float, default=0.60)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(args.device)
    in_channels = 4 if args.freq_aux else 3
    backbone_net = build_model(backbone=args.backbone, pretrained=False, in_channels=in_channels)
    model = FreqAuxModel(backbone_net) if args.freq_aux else backbone_net
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

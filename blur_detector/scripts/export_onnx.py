"""
Export a trained blur detector checkpoint to ONNX.
Usage:
    python scripts/export_onnx.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.freq_aux import FreqAuxModel
from src.models.blur_detector import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="blur_detector/configs/config.yaml")
    parser.add_argument("--checkpoint", default="blur_detector/outputs/checkpoints/best.pt")
    parser.add_argument("--backbone", default=None, help="Override backbone from config")
    parser.add_argument("--image_size", type=int, default=None, help="Override image_size")
    parser.add_argument("--onnx_path", default=None, help="Override output ONNX path")
    parser.add_argument("--freq_aux", action="store_true",
                        help="Wrap backbone with FreqAuxModel before export. The exported "
                             "graph still takes a 3-channel RGB tensor (the Laplacian channel "
                             "is computed inside the graph).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backbone = args.backbone or cfg["model"]["backbone"]
    image_size = args.image_size or cfg["data"]["image_size"]
    onnx_path = args.onnx_path or cfg["export"]["onnx_path"]

    device = torch.device("cpu")
    in_channels = 4 if args.freq_aux else 3
    backbone_net = build_model(backbone=backbone, pretrained=False, in_channels=in_channels)
    model = FreqAuxModel(backbone_net) if args.freq_aux else backbone_net
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dummy = torch.zeros(1, 3, image_size, image_size)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=cfg["export"]["opset_version"],
    )
    print(f"Model exported to {onnx_path}")

    # quick sanity check
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": dummy.numpy()})
    print(f"ONNX output shape: {out[0].shape}  — export OK")


if __name__ == "__main__":
    main()

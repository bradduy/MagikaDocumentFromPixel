"""
Train the blur detector. Supports CLI overrides for autoresearch sweeps.
Also auto-runs evaluation and appends a row to autoresearch/results.tsv.

Usage:
    python blur_detector/scripts/train.py --run_name exp3 --batch_size 256 --image_size 160 --loss focal --exp_id 3 --notes "focal loss 160px"
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.gopro_dataset import GoProDataset
from src.datasets.res_rand_collate import make_res_rand_collate
from src.datasets.freq_aux import FreqAuxModel
from src.models.blur_detector import build_model
from src.training.trainer import Trainer
from src.utils.metrics import compute_metrics


RESULTS_TSV = ROOT / "autoresearch" / "results.tsv"
RESULTS_FIELDS = [
    "exp_id", "run_name", "backbone", "image_size", "batch_size", "lr", "loss",
    "epochs_run", "accuracy", "precision", "recall", "f1", "roc_auc",
    "latency_ms_cpu", "train_time_min", "notes",
]


def _evaluate(model, test_loader, device):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
    return compute_metrics(all_labels, all_preds, all_probs)


def _cpu_latency(model, image_size: int, n: int = 30) -> float:
    m = model.cpu().eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        for _ in range(5):
            m(dummy)
        t0 = time.perf_counter()
        for _ in range(n):
            m(dummy)
        return round((time.perf_counter() - t0) / n * 1000, 2)


def _append_result(row: dict):
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, delimiter="\t")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="blur_detector/configs/config.yaml")
    p.add_argument("--run_name", default="default", help="Unique name → checkpoint subdir")
    p.add_argument("--backbone", default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--loss", default="ce", choices=["ce", "ce_ls", "focal"])
    p.add_argument("--aug_level", default="medium", choices=["light", "medium", "strong"])
    p.add_argument("--use_blur_gamma", action="store_true",
                   help="Include GoPro blur_gamma images as additional blurred training examples")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--res_rand_scales", default="",
                   help="Comma-separated scale set for resolution-randomized training "
                        "(e.g. 256,320,384,448,512). Empty = disabled (fixed-scale).")
    p.add_argument("--freq_aux", action="store_true",
                   help="Concatenate Laplacian-magnitude as a 4th input channel.")
    p.add_argument("--exp_id", type=int, default=0)
    p.add_argument("--notes", default="")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    backbone = args.backbone or cfg["model"]["backbone"]
    image_size = args.image_size or cfg["data"]["image_size"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["lr"]
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg["training"]["weight_decay"]
    epochs = args.epochs or cfg["training"]["epochs"]

    data_cfg = cfg["data"]
    gopro_root = str(ROOT / data_cfg["gopro_root"])
    print(f"[{args.run_name}] backbone={backbone} size={image_size} bs={batch_size} lr={lr} loss={args.loss} epochs={epochs}")

    train_ds = GoProDataset(
        gopro_root=gopro_root, split="train", image_size=image_size,
        color_mode=data_cfg["color_mode"], augment=True,
        val_fraction=data_cfg["train_val_split"],
        aug_level=args.aug_level,
        include_blur_gamma=args.use_blur_gamma,
    )
    val_ds = GoProDataset(
        gopro_root=gopro_root, split="val", image_size=image_size,
        color_mode=data_cfg["color_mode"], augment=False,
        val_fraction=data_cfg["train_val_split"],
        include_blur_gamma=args.use_blur_gamma,
    )
    test_ds = GoProDataset(
        gopro_root=gopro_root, split="test", image_size=image_size,
        color_mode=data_cfg["color_mode"], augment=False,
    )
    print(f"[{args.run_name}] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    train_collate = None
    if args.res_rand_scales:
        scales = tuple(int(s) for s in args.res_rand_scales.split(",") if s.strip())
        print(f"[{args.run_name}] Res-Rand enabled: train batches will be resized "
              f"uniformly to one of {scales} per batch.")
        train_collate = make_res_rand_collate(scales, seed=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers > 0), collate_fn=train_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))

    in_channels = 4 if args.freq_aux else 3
    backbone_net = build_model(backbone=backbone, pretrained=cfg["model"]["pretrained"],
                               in_channels=in_channels)
    model = FreqAuxModel(backbone_net) if args.freq_aux else backbone_net
    if args.freq_aux:
        print(f"[{args.run_name}] Freq-Aux enabled: Laplacian magnitude as 4th input channel.")

    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"] / args.run_name
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        lr=lr, weight_decay=weight_decay,
        epochs=epochs, early_stopping_patience=cfg["training"]["early_stopping_patience"],
        checkpoint_dir=str(ckpt_dir), loss=args.loss,
    )

    t0 = time.time()
    trainer.train()
    train_time_min = round((time.time() - t0) / 60, 2)
    print(f"[{args.run_name}] train_time={train_time_min}min")

    if args.skip_eval:
        return

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = _evaluate(trainer.model, test_loader, device)
    latency = _cpu_latency(trainer.model, image_size)
    print(f"[{args.run_name}] TEST: {metrics}")
    print(f"[{args.run_name}] latency={latency}ms")

    _append_result({
        "exp_id": args.exp_id,
        "run_name": args.run_name,
        "backbone": backbone,
        "image_size": image_size,
        "batch_size": batch_size,
        "lr": lr,
        "loss": args.loss,
        "epochs_run": epochs,
        "accuracy": round(metrics.accuracy, 4),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1": round(metrics.f1, 4),
        "roc_auc": round(metrics.roc_auc, 4),
        "latency_ms_cpu": latency,
        "train_time_min": train_time_min,
        "notes": args.notes,
    })
    print(f"[{args.run_name}] Result logged to {RESULTS_TSV}")


if __name__ == "__main__":
    main()

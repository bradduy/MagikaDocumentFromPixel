from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_metrics, EvalMetrics


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


def build_loss(name: str) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if name == "focal":
        return FocalLoss(gamma=2.0, alpha=0.25)
    raise ValueError(f"Unknown loss: {name}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        early_stopping_patience: int = 5,
        checkpoint_dir: str = "outputs/checkpoints",
        device: Optional[str] = None,
        loss: str = "ce",
        use_amp: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        is_rocm = getattr(torch.version, "hip", None) is not None
        self.use_amp = use_amp and self.device.type == "cuda" and not is_rocm
        if use_amp and is_rocm:
            print("[Trainer] AMP disabled on ROCm (hangs under HSA_OVERRIDE_GFX_VERSION); using fp32.")

        self.criterion = build_loss(loss)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": []}

    def train(self) -> nn.Module:
        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._eval_epoch()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_metrics.f1)

            print(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"{val_metrics}"
            )

            if val_metrics.f1 > best_f1:
                best_f1 = val_metrics.f1
                no_improve = 0
                self._save_checkpoint("best.pt")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        self._load_checkpoint("best.pt")
        return self.model

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(self.train_loader, desc="  train", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        for images, labels in tqdm(self.val_loader, desc="  val  ", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * len(images)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # prob of blurred

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return avg_loss, metrics

    def _save_checkpoint(self, name: str):
        torch.save(self.model.state_dict(), self.ckpt_dir / name)

    def _load_checkpoint(self, name: str):
        self.model.load_state_dict(torch.load(self.ckpt_dir / name, map_location=self.device))

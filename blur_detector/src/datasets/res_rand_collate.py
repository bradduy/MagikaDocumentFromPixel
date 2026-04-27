"""
Resolution-randomized collate.

Samples one resolution per batch from a configured scale set and resizes the
entire batch to that size. Pairs with a GoProDataset that emits tensors at a
large base resolution; this collate downsamples to the sampled scale so the
training distribution over resolutions matches the 5-scale TTA evaluation.
"""

import random
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def make_res_rand_collate(scales: Sequence[int], seed: int | None = None):
    """Return a collate_fn that picks one scale per batch.

    Args:
        scales: e.g. (256, 320, 384, 448, 512). One is sampled uniformly
                per batch and the batch is resized to (s, s).
        seed: optional, for reproducibility of the scale sequence.
    """
    rng = random.Random(seed)
    scale_list: List[int] = list(scales)

    def collate(batch: List[Tuple[torch.Tensor, int]]):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        s = rng.choice(scale_list)
        if imgs.shape[-1] != s:
            imgs = F.interpolate(imgs, size=(s, s), mode="bilinear", align_corners=False)
        return imgs, labels

    return collate

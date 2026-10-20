"""The pretrained backbone, and the cache of frozen-backbone representations.

While the backbone is frozen it computes the same 1280-dimensional vector for
an image on every epoch of every run, so computing it once and reusing it turns
a minutes-long training run into a seconds-long one. That is what makes it
practical to run a controlled sweep rather than one run per question.

The cache is only valid when the backbone is frozen AND the preprocessing is
deterministic. Augmentation and fine-tuning therefore bypass it and go through
the full image pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .data import SPLITS, VOCPerson, eval_transform
from .paths import CACHE

FEATURE_DIM = 1280
WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


def load_backbone(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with its 1000-way ImageNet classifier removed.

    What remains is ``features -> avgpool -> flatten``, i.e. the visual
    representation the network learned on ImageNet, ending in a 1280-vector.
    """
    model = efficientnet_b0(weights=WEIGHTS if pretrained else None)
    model.classifier = nn.Identity()
    return model


def cache_path(split: str) -> Path:
    return CACHE / f"{split}_effb0.pt"


@torch.inference_mode()
def extract_features(
    split: str,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    dataset = VOCPerson(split, transform=eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = load_backbone().to(device).eval()

    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    use_amp = amp and device == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(images)
        features.append(out.float().cpu())
        labels.append(targets)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(features), torch.cat(labels), dataset.ids


def build_cache(
    splits: Tuple[str, ...] = SPLITS, force: bool = False, **kwargs
) -> None:
    for split in splits:
        path = cache_path(split)
        if path.exists() and not force:
            print(f"{split}: cache already present at {path.name}")
            continue
        features, labels, ids = extract_features(split, **kwargs)
        torch.save({"x": features, "y": labels, "ids": ids}, path)
        print(
            f"{split}: cached {tuple(features.shape)} "
            f"({path.stat().st_size / 1e6:.1f} MB), positive rate {labels.mean():.3f}"
        )


def load_cache(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = cache_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m scripts.build_cache` first."
        )
    blob = torch.load(path, weights_only=False)
    return blob["x"], blob["y"]


def cached_loader(
    split: str,
    batch_size: int,
    shuffle: bool,
    device: str = "cuda",
    subset: Optional[int] = None,
    subset_seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """A DataLoader over cached features.

    The whole cache is ~50 MB, so it is moved to the GPU once and the loader
    only slices it. No workers, no host-to-device copy per batch.
    """
    x, y = load_cache(split)
    if subset is not None and subset < len(y):
        gen = torch.Generator().manual_seed(subset_seed)
        keep = torch.randperm(len(y), generator=gen)[:subset]
        x, y = x[keep], y[keep]
    x, y = x.to(device), y.to(device)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )

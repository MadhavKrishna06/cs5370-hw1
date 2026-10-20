"""Settings for one training run.

Every experiment in this assignment is produced by taking ``BASELINE`` and
changing a single field, so that each comparison stays controlled. The
dataclass is frozen, which makes a config safe to share between runs and
guarantees that an experiment cannot quietly mutate the baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class Config:
    # ---- shape of the custom head -------------------------------------
    hidden: int = 512
    n_hidden_layers: int = 1
    activation: str = "relu"
    dropout: float = 0.0

    # ---- optimization --------------------------------------------------
    optimizer: str = "adam"              # sgd | momentum | rmsprop | adam
    lr: float = 1e-3
    momentum: float = 0.9                # used by momentum and rmsprop
    rmsprop_alpha: float = 0.99
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0            # L2 penalty
    batch_size: int = 64
    epochs: int = 40

    # ---- data pipeline --------------------------------------------------
    mode: str = "cached"                 # cached | images
    augment: bool = False                # only meaningful when mode == "images"
    train_subset: Optional[int] = None   # cap the training set, to induce overfitting

    # ---- fine-tuning ----------------------------------------------------
    unfreeze_from: Optional[int] = None  # first index of efficientnet.features to train
    backbone_lr: Optional[float] = None  # defaults to lr / 10

    # ---- controls and instrumentation ------------------------------------
    seed: int = 0
    early_stop_patience: int = 8
    early_stop_metric: str = "val_loss"  # val_loss | val_f1
    amp: bool = True
    probe_every: int = 1                 # epochs between the expensive diagnostics
    num_workers: int = 4

    def with_(self, **changes) -> "Config":
        """Return a copy with some fields replaced. Keeps the baseline intact."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def effective_backbone_lr(self) -> float:
        return self.lr / 10.0 if self.backbone_lr is None else self.backbone_lr

    @property
    def is_finetuning(self) -> bool:
        return self.mode == "images" and self.unfreeze_from is not None


BASELINE = Config()

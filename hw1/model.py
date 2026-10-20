"""The custom classification head, and the full model used for fine-tuning."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .features import FEATURE_DIM, load_backbone

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

# Which diagnostics make sense for which activation: a rectified unit can be
# inactive, a squashing unit can saturate, and the two are measured differently.
RECTIFIED = {"relu", "leaky_relu", "gelu", "silu"}
SATURATING = {"sigmoid", "tanh"}


class Head(nn.Module):
    """``Linear -> activation -> dropout -> ... -> Linear(., 1)``.

    One output logit, for use with ``BCEWithLogitsLoss``.

    The most recent pre-activations and post-activations are kept when asked
    for, so the training loop can measure saturation and inactive units without
    a second forward pass.
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden: int = 512,
        n_hidden_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}"
            )
        if n_hidden_layers < 1:
            raise ValueError("the assignment requires at least one hidden layer")

        self.activation_name = activation
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        width = in_dim
        for _ in range(n_hidden_layers):
            self.linears.append(nn.Linear(width, hidden))
            self.activations.append(ACTIVATIONS[activation]())
            self.dropouts.append(nn.Dropout(dropout))
            width = hidden
        self.out = nn.Linear(width, 1)

        self.last_preactivations: List[torch.Tensor] = []
        self.last_activations: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, keep_activations: bool = False) -> torch.Tensor:
        if keep_activations:
            self.last_preactivations = []
            self.last_activations = []
        for linear, activation, dropout in zip(
            self.linears, self.activations, self.dropouts
        ):
            z = linear(x)
            a = activation(z)
            if keep_activations:
                self.last_preactivations.append(z.detach().float())
                self.last_activations.append(a.detach().float())
            x = dropout(a)
        return self.out(x).squeeze(-1)


class BackboneWithHead(nn.Module):
    """Pretrained backbone followed by the custom head, for the image pipeline.

    ``unfreeze_from`` is an index into ``efficientnet.features``: blocks from
    that index onwards are trained, everything before it stays frozen. Passing
    ``None`` keeps the whole backbone frozen, which reproduces the cached-feature
    setting through the slow path.
    """

    def __init__(self, head: Head, unfreeze_from: Optional[int] = None):
        super().__init__()
        self.backbone = load_backbone()
        self.head = head
        self.unfreeze_from = unfreeze_from
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.unfreeze_from is None:
            return
        blocks = self.backbone.features
        if not 0 <= self.unfreeze_from < len(blocks):
            raise ValueError(
                f"unfreeze_from must be in [0, {len(blocks) - 1}], "
                f"got {self.unfreeze_from}"
            )
        for index in range(self.unfreeze_from, len(blocks)):
            for parameter in blocks[index].parameters():
                parameter.requires_grad_(True)

    def set_train_mode(self) -> None:
        """Put the model in training mode, but keep frozen BatchNorm in eval.

        A frozen block in ``train()`` mode would still update its BatchNorm
        running statistics, so its output would drift across epochs even though
        none of its weights change. That would silently break both the
        comparison against the cached features and the controls in the
        fine-tuning experiment.
        """
        self.train()
        if self.unfreeze_from is None:
            self.backbone.eval()
            return
        for index, block in enumerate(self.backbone.features):
            if index < self.unfreeze_from:
                block.eval()

    def forward(self, x: torch.Tensor, keep_activations: bool = False) -> torch.Tensor:
        return self.head(self.backbone(x), keep_activations=keep_activations)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_groups(self, head_lr: float, backbone_lr: float) -> list:
        """Discriminative learning rates: the pretrained weights move slower
        than the randomly initialized head."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": list(self.head.parameters()), "lr": head_lr, "name": "head"}]
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"}
            )
        return groups


def build_head(cfg) -> Head:
    return Head(
        in_dim=FEATURE_DIM,
        hidden=cfg.hidden,
        n_hidden_layers=cfg.n_hidden_layers,
        activation=cfg.activation,
        dropout=cfg.dropout,
    )


def build_model(cfg) -> nn.Module:
    """A bare head for the cached pipeline, backbone + head for the image one."""
    head = build_head(cfg)
    if cfg.mode == "cached":
        return head
    if cfg.mode == "images":
        return BackboneWithHead(head, unfreeze_from=cfg.unfreeze_from)
    raise ValueError(f"mode must be 'cached' or 'images', got {cfg.mode!r}")

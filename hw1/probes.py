"""Measurements taken inside the network during training.

Final accuracy cannot distinguish an activation that saturates from one that
stays in its linear range, or an optimizer that takes small well-aimed steps
from one that thrashes. These probes record the quantities that can.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .model import RECTIFIED, SATURATING

# |pre-activation| above this puts a sigmoid/tanh unit in its flat region:
# sigmoid'(4) is about 0.018, roughly 7% of its peak slope.
SATURATION_THRESHOLD = 4.0


def _norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(tensor.detach().float().pow(2).sum())
    return total ** 0.5


def gradient_norms(model: nn.Module) -> Dict[str, float]:
    """L2 norm of the gradient, overall and per named parameter.

    Reported per layer because the interesting failure of a squashing
    activation is that the gradient reaching the *first* layer is far smaller
    than the one reaching the last.
    """
    out: Dict[str, float] = {}
    grads: List[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grads.append(parameter.grad)
        out[f"grad_norm/{name}"] = _norm([parameter.grad])
    out["grad_norm/total"] = _norm(grads)
    return out


def weight_norms(model: nn.Module) -> Dict[str, float]:
    out: Dict[str, float] = {}
    weights = [p for _, p in model.named_parameters() if p.requires_grad]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dim() > 1:
            out[f"weight_norm/{name}"] = _norm([parameter])
    out["weight_norm/total"] = _norm(weights)
    return out


def snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Copy the trainable parameters, for measuring how far a step moved them."""
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def update_norms(
    model: nn.Module, previous: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """How far the parameters moved, in absolute and relative terms.

    ``update_ratio`` (step size divided by weight size) is the scale-free
    quantity that makes the learning-rate and optimizer comparisons readable:
    it is roughly constant for a healthy run, tiny when the learning rate is
    too small, and large and erratic when it is too large.
    """
    out: Dict[str, float] = {}
    deltas: List[torch.Tensor] = []
    current: List[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in previous:
            continue
        delta = parameter.detach() - previous[name]
        deltas.append(delta)
        current.append(parameter.detach())
        if parameter.dim() > 1:
            weight_size = _norm([parameter])
            step_size = _norm([delta])
            out[f"update_norm/{name}"] = step_size
            out[f"update_ratio/{name}"] = step_size / (weight_size + 1e-12)
    total_step = _norm(deltas)
    out["update_norm/total"] = total_step
    out["update_ratio/total"] = total_step / (_norm(current) + 1e-12)
    return out


def activation_stats(
    preactivations: List[torch.Tensor],
    activations: List[torch.Tensor],
    activation_name: str,
) -> Dict[str, float]:
    """Saturation and inactivity statistics for each hidden layer.

    ``zero_fraction`` counts individual zero outputs; ``dead_unit_fraction``
    counts units that produced zero for *every* example in the batch, which is
    the stronger claim and the one that matters for a dying ReLU.
    """
    out: Dict[str, float] = {}
    for layer, (z, a) in enumerate(zip(preactivations, activations)):
        prefix = f"act/layer{layer}"
        out[f"{prefix}/preact_mean"] = float(z.mean())
        out[f"{prefix}/preact_std"] = float(z.std())
        out[f"{prefix}/preact_absmax"] = float(z.abs().max())
        out[f"{prefix}/out_mean"] = float(a.mean())
        out[f"{prefix}/out_std"] = float(a.std())

        if activation_name in RECTIFIED:
            is_zero = a.abs() <= 1e-12
            out[f"{prefix}/zero_fraction"] = float(is_zero.float().mean())
            out[f"{prefix}/dead_unit_fraction"] = float(is_zero.all(dim=0).float().mean())
            out[f"{prefix}/negative_preact_fraction"] = float((z < 0).float().mean())
        if activation_name in SATURATING:
            saturated = z.abs() > SATURATION_THRESHOLD
            out[f"{prefix}/saturated_fraction"] = float(saturated.float().mean())
            if activation_name == "sigmoid":
                slope = a * (1.0 - a)
            else:
                slope = 1.0 - a.pow(2)
            out[f"{prefix}/mean_local_slope"] = float(slope.mean())
    return out


def flatten_probe(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in values.items()}

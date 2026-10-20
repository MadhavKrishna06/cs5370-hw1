"""One training loop, driven entirely by a Config.

Every experiment calls ``train`` with configs that differ in a single field, so
the loop itself is the control: same initialization scheme, same data order,
same stopping rule, same instrumentation for every run.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import probes
from .config import Config
from .data import image_loader
from .features import cached_loader
from .metrics import binary_metrics, roc_auc
from .model import BackboneWithHead, build_model
from .paths import RESULTS

# How many optimizer steps per probe epoch get the expensive per-layer
# measurements. The cheap total gradient norm is recorded on every step.
PROBE_STEPS = 10


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Make a run reproducible: same initialization, same shuffling, same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    """SGD, SGD+momentum, RMSProp or Adam over the trainable parameters only."""
    if isinstance(model, BackboneWithHead):
        groups = [
            {key: value for key, value in group.items() if key != "name"}
            for group in model.parameter_groups(cfg.lr, cfg.effective_backbone_lr)
        ]
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": trainable, "lr": cfg.lr}]

    common = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    name = cfg.optimizer
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.0, **common)
    if name == "momentum":
        return torch.optim.SGD(groups, momentum=cfg.momentum, **common)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, alpha=cfg.rmsprop_alpha, momentum=0.0, **common)
    if name == "adam":
        return torch.optim.Adam(groups, betas=tuple(cfg.adam_betas), **common)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def build_loaders(cfg: Config, dev: str) -> Tuple:
    """Cached features for the frozen-backbone studies, images for the rest."""
    generator = torch.Generator().manual_seed(cfg.seed)
    if cfg.mode == "cached":
        if cfg.augment:
            raise ValueError(
                "augmentation changes the image, so it cannot use the frozen-feature "
                "cache; set mode='images'"
            )
        train_loader = cached_loader(
            "train", cfg.batch_size, True, dev, cfg.train_subset, cfg.seed, generator
        )
        val_loader = cached_loader("val", 512, False, dev)
        return train_loader, val_loader

    train_loader = image_loader(
        "train",
        cfg.batch_size,
        shuffle=True,
        augment=cfg.augment,
        num_workers=cfg.num_workers,
        subset=cfg.train_subset,
        generator=generator,
    )
    val_loader = image_loader(
        "val", cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, criterion: nn.Module, dev: str, use_amp: bool
) -> Dict[str, float]:
    model.eval()
    logits_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    for inputs, targets in loader:
        inputs = inputs.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)
        logits_all.append(logits.float())
        targets_all.append(targets.float())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    out = {"loss": float(criterion(logits, targets))}
    out.update(binary_metrics(logits, targets))
    out["roc_auc"] = roc_auc(logits.cpu(), targets.cpu())
    return out


@dataclass
class Result:
    tag: str
    config: dict
    history: List[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val: Dict[str, float] = field(default_factory=dict)
    test: Optional[Dict[str, float]] = None
    train_seconds: float = 0.0
    trainable_parameters: int = 0
    stopped_early: bool = False
    diverged: bool = False
    amp_fell_back: bool = False

    def save(self, group: str) -> Path:
        directory = RESULTS / group
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.tag}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def train(cfg: Config, tag: str, verbose: bool = True) -> Tuple[Result, nn.Module]:
    """Train one configuration. Returns the result record and the best model.

    Model selection and stopping use the validation split only; the test split
    is never touched here.
    """
    dev = device()
    set_seed(cfg.seed)

    model = build_model(cfg).to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(cfg, model)

    # Mixed precision keeps the image pipeline inside 4 GB, but it rescales
    # gradients, so the probes unscale before measuring. The cached pipeline is
    # already fast and runs in full precision to keep its diagnostics exact.
    use_amp = bool(cfg.amp) and dev == "cuda" and cfg.mode == "images"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader, val_loader = build_loaders(cfg, dev)

    # Guard against a GPU-specific fp16 fault rather than training on NaNs.
    # On this machine (GTX 1650, Turing TU117) the half-precision convolution
    # kernel the backbone selects at batch size 64 returns NaN for a perfectly
    # ordinary input, while 32, 48 and 96 are all fine; it is a kernel
    # selection fault, not an overflow, and full precision is unaffected.
    # Rather than let that reach the loss, one real batch is checked first and
    # the run falls back to full precision if it fails.
    # The check runs in eval() mode on purpose: in train() mode the backbone's
    # BatchNorm layers would fold this batch into their running statistics, and
    # if the batch comes back NaN those statistics are permanently poisoned —
    # training would then look healthy while every evaluation returned NaN.
    amp_fell_back = False
    if use_amp:
        check_inputs, _ = next(iter(train_loader))
        was_training = model.training
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            probe_logits = model(check_inputs.to(dev))
        model.train(was_training)
        if not torch.isfinite(probe_logits).all():
            print(
                f"  [{tag}] mixed precision produced non-finite activations at "
                f"batch size {cfg.batch_size}; falling back to full precision"
            )
            use_amp = False
            amp_fell_back = True
            scaler = torch.amp.GradScaler("cuda", enabled=False)

    result = Result(
        tag=tag,
        config=cfg.to_dict(),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )
    best_state = copy.deepcopy(model.state_dict())
    higher_is_better = cfg.early_stop_metric != "val_loss"
    best_score = float("-inf") if higher_is_better else float("inf")
    epochs_without_improvement = 0

    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        if isinstance(model, BackboneWithHead):
            model.set_train_mode()
        else:
            model.train()

        epoch_snapshot = probes.snapshot(model)
        is_probe_epoch = epoch % cfg.probe_every == 0
        running_loss = 0.0
        seen = 0
        grad_norm_sum = 0.0
        steps = 0
        overflow_steps = 0
        step_ratios: List[float] = []
        layer_probe: Dict[str, float] = {}

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            detailed = is_probe_epoch and step < PROBE_STEPS

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(inputs, keep_activations=detailed)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if use_amp:
                # Undo the loss scaling, so the numbers below are real gradients.
                scaler.unscale_(optimizer)

            # With mixed precision the scaler deliberately probes for overflow,
            # so some steps carry non-finite gradients and are then skipped by
            # scaler.step. Those steps are counted but kept out of the mean,
            # which would otherwise be inf for the whole epoch.
            step_grad_norm = probes.gradient_norms(model)["grad_norm/total"]
            if np.isfinite(step_grad_norm):
                grad_norm_sum += step_grad_norm
                steps += 1
            else:
                overflow_steps += 1

            before_step = None
            if detailed:
                head = model.head if isinstance(model, BackboneWithHead) else model
                layer_probe.update(probes.gradient_norms(head))
                layer_probe.update(
                    probes.activation_stats(
                        head.last_preactivations,
                        head.last_activations,
                        head.activation_name,
                    )
                )
                before_step = probes.snapshot(model)

            scaler.step(optimizer)
            scaler.update()

            if before_step is not None:
                step_ratios.append(
                    probes.update_norms(model, before_step)["update_ratio/total"]
                )

            running_loss += float(loss) * len(targets)
            seen += len(targets)

        train_loss = running_loss / max(seen, 1)
        val = evaluate(model, val_loader, criterion, dev, use_amp)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_roc_auc": val["roc_auc"],
            "grad_norm_mean": grad_norm_sum / max(steps, 1),
            "amp_overflow_steps": overflow_steps,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        record.update(probes.update_norms(model, epoch_snapshot))
        record.update(probes.weight_norms(model))
        if step_ratios:
            record["update_ratio/step_mean"] = sum(step_ratios) / len(step_ratios)
        record.update(layer_probe)
        result.history.append(record)

        if not np.isfinite(train_loss):
            result.diverged = True
            if verbose:
                print(f"  [{tag}] epoch {epoch}: train loss is {train_loss}, stopping")
            break

        monitored = cfg.early_stop_metric
        score = val["loss"] if monitored == "val_loss" else val[monitored[4:]]
        improved = score > best_score if higher_is_better else score < best_score
        if improved:
            best_score = score
            result.best_epoch = epoch
            result.best_val_loss = val["loss"]
            result.best_val = val
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            marker = "  *" if improved else ""
            print(
                f"  [{tag}] epoch {epoch:>3}  train {train_loss:.4f}  "
                f"val {val['loss']:.4f}  acc {val['accuracy']:.4f}  "
                f"f1 {val['f1']:.4f}  grad {record['grad_norm_mean']:.3e}{marker}"
            )

        if epochs_without_improvement >= cfg.early_stop_patience:
            result.stopped_early = True
            if verbose:
                print(f"  [{tag}] no improvement for {cfg.early_stop_patience} epochs")
            break

    result.amp_fell_back = amp_fell_back
    result.train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    return result, model


def evaluate_on_test(cfg: Config, model: nn.Module) -> Dict[str, float]:
    """Final held-out evaluation. Called once, for the selected model only."""
    dev = device()
    criterion = nn.BCEWithLogitsLoss()
    if cfg.mode == "cached":
        loader = cached_loader("test", 512, False, dev)
    else:
        loader = image_loader(
            "test", cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
    use_amp = bool(cfg.amp) and dev == "cuda" and cfg.mode == "images"
    return evaluate(model, loader, criterion, dev, use_amp)


def run_group(
    group: str,
    configs: Dict[str, Config],
    verbose: bool = True,
    skip_existing: bool = True,
) -> Dict[str, Result]:
    """Train every config in a group and write one JSON per run.

    Runs that already have a saved record are skipped, so a sweep can be
    extended with new values without retraining the old ones. Delete the
    group's directory to force a clean rerun.
    """
    results: Dict[str, Result] = {}
    total = len(configs)
    for index, (tag, cfg) in enumerate(configs.items(), start=1):
        if skip_existing and (RESULTS / group / f"{tag}.json").exists():
            print(f"[{group}] {index}/{total} {tag}: already done, skipping")
            continue
        print(f"[{group}] {index}/{total} {tag}")
        result, _ = train(cfg, tag, verbose=verbose)
        result.save(group)
        results[tag] = result
        best = result.best_val
        print(
            f"    best epoch {result.best_epoch}  val loss {result.best_val_loss:.4f}  "
            f"acc {best.get('accuracy', float('nan')):.4f}  "
            f"f1 {best.get('f1', float('nan')):.4f}  "
            f"{result.train_seconds:.1f} s"
            f"{'  DIVERGED' if result.diverged else ''}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def load_group(group: str) -> Dict[str, dict]:
    directory = RESULTS / group
    if not directory.exists():
        return {}
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}

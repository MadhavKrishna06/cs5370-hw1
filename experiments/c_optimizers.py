"""Experiment C: SGD, momentum, RMSProp and Adam.

Question
    How do the four update rules differ in the trajectory they take, and how
    much of any difference in final performance is really a difference in how
    sensitive each one is to the learning rate?

Design
    Two arms, because a single arm cannot answer both questions:

    C1, shared learning rate.
        All four optimizers at lr = 1e-3. This is the comparison people usually
        report, and it is mostly a measurement of sensitivity: 1e-3 is a
        sensible Adam setting and a very small plain-SGD setting, so plain SGD
        is expected to look bad for a reason that has nothing to do with the
        quality of its final solution.

    C2, each optimizer at its own learning rate.
        A small sweep per optimizer on one seed, the best value chosen by
        validation loss, then three seeds at that value. This is the fair
        comparison of the update rules themselves.

    Reporting only C1 would conflate the two. Reporting only C2 would hide that
    the optimizers differ enormously in how much tuning they need.

Measured
    Loss curves and epochs to reach a fixed validation loss, for convergence
    speed; the update-to-weight ratio, which exposes the mechanism, since
    momentum accumulates a direction and so takes larger steps than its
    gradient alone would justify, while RMSProp and Adam divide by a
    squared-gradient estimate and so normalize step size per parameter; the
    spread across seeds, for stability.
"""

from __future__ import annotations

import sys
from typing import Dict

from hw1.config import Config
from hw1.train import run_group, train

from .common import CACHED_BASE, SEEDS, seeded

SHARED_LR = 1e-3

OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam")

# Grids per optimizer, centred on the value each one is normally used at.
TUNING_GRID: Dict[str, tuple] = {
    "sgd": (1e-2, 3e-2, 1e-1, 3e-1, 1.0),
    "momentum": (1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    "rmsprop": (1e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    "adam": (1e-5, 1e-4, 3e-4, 1e-3, 3e-3),
}


def shared_lr_configs() -> Dict[str, Config]:
    out: Dict[str, Config] = {}
    for optimizer in OPTIMIZERS:
        out.update(
            seeded(CACHED_BASE, f"shared_{optimizer}", optimizer=optimizer, lr=SHARED_LR)
        )
    return out


def tune_learning_rates(verbose: bool = False) -> Dict[str, float]:
    """One seed per candidate learning rate; keep the best validation loss.

    Selection uses the validation split only, and the chosen value is then
    re-run on fresh seeds, so the reported numbers are not the same runs that
    were used to pick the value.
    """
    best: Dict[str, float] = {}
    for optimizer in OPTIMIZERS:
        scores = []
        for lr in TUNING_GRID[optimizer]:
            cfg = CACHED_BASE.with_(optimizer=optimizer, lr=lr, seed=99)
            tag = f"tune_{optimizer}_lr{lr:.0e}"
            result, _ = train(cfg, tag, verbose=verbose)
            result.save("c_optimizers_tuning")
            scores.append((result.best_val_loss, lr))
            print(f"    {tag}: val loss {result.best_val_loss:.4f}")
        scores.sort()
        best[optimizer] = scores[0][1]
        print(f"  {optimizer}: chose lr = {best[optimizer]:.0e}")
    return best


def tuned_configs(best: Dict[str, float]) -> Dict[str, Config]:
    out: Dict[str, Config] = {}
    for optimizer, lr in best.items():
        out.update(
            seeded(CACHED_BASE, f"tuned_{optimizer}", optimizer=optimizer, lr=lr)
        )
    return out


def main() -> int:
    print("C1: all optimizers at a shared learning rate")
    run_group("c_optimizers", shared_lr_configs(), verbose=False)

    print("\nC2: tuning each optimizer on the validation split (seed 99)")
    best = tune_learning_rates()

    print("\nC2: three seeds at each optimizer's chosen learning rate")
    run_group("c_optimizers", tuned_configs(best), verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Experiment B: the effect of the learning rate.

Question
    What does a learning rate that is too small, roughly right, or too large do
    to the optimization, and which measurement shows it first?

Design
    Plain SGD, no momentum and no adaptive scaling, so the step taken is
    exactly the learning rate times the gradient and the relationship between
    the two is not mediated by anything else. Six values spanning five orders of
    magnitude, 1e-5 to 1.0, three seeds each, no early stopping, so a slow run
    is visibly slow rather than truncated and a diverging run is visibly
    diverging.

    Adam is deliberately not used here. Its per-parameter rescaling makes the
    usable range of learning rates much wider, which would hide the effect this
    experiment is about.

Measured
    Training and validation loss per epoch; the update-to-weight ratio, which
    is scale-free and therefore comparable across learning rates; mean gradient
    norm, which grows rather than shrinks once a run becomes unstable; and
    whether the loss became non-finite.
"""

from __future__ import annotations

import sys

from hw1.train import run_group

from .common import CACHED_BASE, sweep

# 10 and 100 are included because nothing in the 1e-5 to 1.0 range actually
# diverged; the claim "too large a learning rate breaks training" needs a
# run where it visibly breaks, not merely one that is noisy.
LEARNING_RATES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

SGD_BASE = CACHED_BASE.with_(optimizer="sgd", momentum=0.0)


def configs() -> dict:
    return sweep(
        SGD_BASE,
        "lr",
        LEARNING_RATES,
        label=lambda value: f"lr{value:.0e}",
    )


def main() -> int:
    run_group("b_learning_rate", configs(), verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

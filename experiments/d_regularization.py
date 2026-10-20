"""Experiment D: regularization and generalization.

Two of the three offered techniques are studied: dropout and L2. Both act on
the head alone, which means both can use the frozen-feature cache and can
therefore be swept densely enough to find where each one stops helping and
starts hurting. Data augmentation acts on the image and so belongs to the
image pipeline; it is covered in experiment E, where that pipeline is already
in use.

Question
    Does each technique reduce the gap between training and validation loss;
    does a smaller gap come with a better validation score; and does more of
    the technique eventually hurt?

Design
    The baseline head overfits on its own, which is what makes this measurable:
    unregularized training drives training loss down while validation loss
    turns back upwards after a few epochs. Each technique is then swept over
    seven strengths including zero, three seeds each, one variable at a time.

    Early stopping is active here (patience 12 out of 60 epochs), because the
    question is about the best generalization each setting can reach, not about
    where it ends up after a fixed number of epochs. The epoch at which each
    run peaked is recorded, since a technique that mainly buys *time* before
    overfitting is a different claim from one that improves the peak.

    A third, smaller arm trains on a 500-image subset. It is there to separate
    "this technique does little" from "there was little overfitting to remove":
    if the effect appears strongly once data is scarce, then the honest
    conclusion about the full-data result is that the gap was already small.

Measured
    Training and validation loss per epoch, the generalization gap at the
    best epoch, best validation accuracy and F1, the epoch of the peak, and
    the weight norm, which should fall visibly as L2 strength rises.
"""

from __future__ import annotations

import sys

from hw1.train import run_group

from .common import CACHED_BASE, sweep

# Extended to 0.95 and 0.99 after the first sweep: dropout was still helping
# at 0.9, so the grid did not yet contain the point where it starts to hurt.
DROPOUT_VALUES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
WEIGHT_DECAY_VALUES = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

# Early stopping on, so each setting is judged by the best it can do.
REG_BASE = CACHED_BASE.with_(epochs=60, early_stop_patience=12)

SUBSET_SIZE = 500
SUBSET_BASE = REG_BASE.with_(train_subset=SUBSET_SIZE)


def configs() -> dict:
    out = {}
    out.update(
        sweep(REG_BASE, "dropout", DROPOUT_VALUES, label=lambda v: f"dropout{v:g}")
    )
    out.update(
        sweep(REG_BASE, "weight_decay", WEIGHT_DECAY_VALUES, label=lambda v: f"l2_{v:g}")
    )
    # Scarce-data arm: the two extremes of each technique only.
    out.update(
        sweep(
            SUBSET_BASE,
            "dropout",
            (0.0, 0.5),
            label=lambda v: f"subset{SUBSET_SIZE}_dropout{v:g}",
        )
    )
    out.update(
        sweep(
            SUBSET_BASE,
            "weight_decay",
            (0.0, 1e-2),
            label=lambda v: f"subset{SUBSET_SIZE}_l2_{v:g}",
        )
    )
    return out


def main() -> int:
    run_group("d_regularization", configs(), verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

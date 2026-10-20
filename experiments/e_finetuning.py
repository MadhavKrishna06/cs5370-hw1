"""Experiment E: frozen representation versus partial fine-tuning.

Question
    When is it worth updating the pretrained weights rather than only the new
    head, and what does that buy relative to what it costs?

Design
    All runs use the image pipeline, so the only difference between arms is how
    much of the backbone is trainable. ``unfreeze_from`` names the first block
    of ``efficientnet.features`` that is allowed to learn:

        None  frozen backbone, head only          0.66 M trainable
        8     final 1x1 convolution and its BN    1.07 M trainable
        7     last MBConv stage onwards           1.79 M trainable
        6     two stages onwards                   3.81 M trainable
        5     three stages onwards                 4.36 M trainable

    The unfrozen blocks are the late ones. Early convolutions encode edges and
    colour, which transfer to almost any natural-image task; late blocks encode
    the class-specific combinations that ImageNet needed and this task may not.

    The pretrained weights use a learning rate ten times smaller than the head.
    The head starts from random values and has to move far; the backbone starts
    from a good solution and large steps would destroy it.

    Two seeds per arm rather than three: each run is about ten minutes, where a
    cached run is about ten seconds. The seed-to-seed spread measured in the
    cached experiments is used when judging whether differences here are real.

    The augmentation arms belong to experiment D's question but need the image
    pipeline, so they are run here: random resized crop, horizontal flip and
    colour jitter, applied with the backbone frozen and again with it partly
    unfrozen. Fine-tuning raises the trainable parameter count several-fold, so
    the point of the pairing is whether augmentation matters more once there is
    more capacity to overfit with.

Also recorded
    Trainable parameter count, seconds per epoch and total training time, so
    that any accuracy gain can be weighed against its cost; the
    training-to-validation gap, since fine-tuning adds capacity and may overfit
    sooner; and the head-only image run doubles as a check that the frozen
    feature cache used elsewhere agrees with the full pipeline.
"""

from __future__ import annotations

import sys
from typing import Dict

from hw1.config import Config
from hw1.train import run_group

from .common import CACHED_BASE

SEEDS = (0, 1)

UNFREEZE_LEVELS = (None, 8, 7, 6, 5)

# Fifteen epochs with patience 5: enough for the frozen arm to converge and for
# the fine-tuned arms to start overfitting, which is the behaviour of interest.
IMAGE_BASE = CACHED_BASE.with_(
    mode="images",
    batch_size=32,
    num_workers=4,
    epochs=15,
    early_stop_patience=5,
    optimizer="adam",
    lr=1e-3,
    backbone_lr=1e-4,
    amp=True,
)


def configs() -> Dict[str, Config]:
    out: Dict[str, Config] = {}
    for level in UNFREEZE_LEVELS:
        name = "frozen" if level is None else f"unfreeze{level}"
        for seed in SEEDS:
            out[f"{name}_seed{seed}"] = IMAGE_BASE.with_(
                unfreeze_from=level, seed=seed
            )
    for level in (None, 7):
        name = "frozen" if level is None else f"unfreeze{level}"
        for seed in SEEDS:
            out[f"{name}_augment_seed{seed}"] = IMAGE_BASE.with_(
                unfreeze_from=level, seed=seed, augment=True
            )
    return out


def main() -> int:
    run_group("e_finetuning", configs(), verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Selection runs for the final model.

Experiments A–E each answer one question in isolation, and the settings they
favour were measured in different places: dropout was swept on the frozen head
through the feature cache, augmentation was tested at two unfreeze levels, and
the best unfreeze level was found without either. Combining the three winners
and assuming the result is best would be extrapolation.

This group therefore runs the small cross of the two settings that are still
open — dropout strength and augmentation — at the unfreeze level experiment E
selected, on the image pipeline where the final model will actually live. One
seed per cell, chosen on validation loss; the selected configuration is then
retrained from scratch in ``scripts/final_model.py``.

The test split is not touched here.
"""

from __future__ import annotations

import sys
from typing import Dict

from hw1.config import Config
from hw1.train import run_group

from .e_finetuning import IMAGE_BASE

# Experiment E: validation loss 0.2523 at unfreeze_from=6, against 0.2663 at 7
# and 0.2568 at 5; the differences exceed the seed spread of about 0.005.
UNFREEZE = 6

DROPOUT_VALUES = (0.0, 0.5, 0.9)
AUGMENT_VALUES = (False, True)

SELECTION_BASE = IMAGE_BASE.with_(unfreeze_from=UNFREEZE, epochs=20, early_stop_patience=6)


def configs() -> Dict[str, Config]:
    out: Dict[str, Config] = {}
    for dropout in DROPOUT_VALUES:
        for augment in AUGMENT_VALUES:
            tag = f"uf{UNFREEZE}_dropout{dropout:g}_{'aug' if augment else 'noaug'}"
            out[tag] = SELECTION_BASE.with_(dropout=dropout, augment=augment, seed=0)
    return out


def main() -> int:
    run_group("f_selection", configs(), verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Filesystem layout. Every artifact of a run lands under the project root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
VOC_ROOT = DATA                 # torchvision expects <root>/VOCdevkit/VOC2007
CACHE = ROOT / "cache"          # frozen-backbone feature tensors
RESULTS = ROOT / "results"      # one JSON per training run
FIGURES = ROOT / "report" / "figures"
CHECKPOINTS = ROOT / "checkpoints"

for _directory in (CACHE, RESULTS, FIGURES, CHECKPOINTS):
    _directory.mkdir(parents=True, exist_ok=True)

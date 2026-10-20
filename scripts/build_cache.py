"""Compute and store the frozen-backbone representation of every image.

    python -m scripts.build_cache

While the backbone is frozen its output for a given image never changes, so
this runs once and every later head-only experiment reads 1280-dimensional
vectors instead of decoding and convolving images again.

The script finishes with a sanity check: a linear probe on the cached features.
If ImageNet features really do transfer to "is there a person here", a single
linear layer should already be well above the majority-class baseline. If it is
not, the cache is wrong and nothing built on top of it can be trusted.
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn

from hw1.data import SPLITS, read_person_labels
from hw1.features import FEATURE_DIM, build_cache, load_cache
from hw1.metrics import binary_metrics
from hw1.train import device, set_seed


def linear_probe(epochs: int = 100) -> dict:
    """Logistic regression on the cached features, trained full-batch."""
    dev = device()
    set_seed(0)
    x_train, y_train = load_cache("train")
    x_val, y_val = load_cache("val")
    x_train, y_train = x_train.to(dev), y_train.to(dev)
    x_val, y_val = x_val.to(dev), y_val.to(dev)

    model = nn.Linear(FEATURE_DIM, 1).to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=epochs, history_size=20)

    def closure():
        optimizer.zero_grad()
        loss = criterion(model(x_train).squeeze(-1), y_train)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        metrics = binary_metrics(model(x_val).squeeze(-1), y_val)
    return metrics


def main() -> int:
    if not torch.cuda.is_available():
        print("warning: no CUDA device found, the cache will be built on CPU and be slow")

    start = time.perf_counter()
    build_cache(
        SPLITS,
        device=device(),
        batch_size=32,
        num_workers=4,
        amp=True,
    )
    print(f"\ncache built in {time.perf_counter() - start:.1f} s")

    for split in SPLITS:
        x, y = load_cache(split)
        _, labels = read_person_labels(split)
        assert len(x) == len(labels), f"{split}: cache has {len(x)} rows, labels {len(labels)}"
        assert torch.equal(y, labels), f"{split}: cached labels do not match the annotation file"
        assert x.shape[1] == FEATURE_DIM, f"{split}: expected {FEATURE_DIM} features"
        assert torch.isfinite(x).all(), f"{split}: cache contains non-finite values"
    print("cache shapes, labels and finiteness verified")

    _, y_val = load_cache("val")
    majority = float(max(y_val.mean(), 1 - y_val.mean()))
    metrics = linear_probe()
    print(
        f"\nlinear probe on frozen features (validation):"
        f"\n  accuracy  {metrics['accuracy']:.4f}"
        f"\n  precision {metrics['precision']:.4f}"
        f"\n  recall    {metrics['recall']:.4f}"
        f"\n  f1        {metrics['f1']:.4f}"
        f"\n  majority-class baseline {majority:.4f}"
    )
    if metrics["accuracy"] < majority + 0.10:
        print(
            "\nFAILED: the linear probe is barely better than predicting the majority"
            " class, so the cache is probably misaligned with its labels"
        )
        return 1
    print("\nsanity check passed: the pretrained representation is already informative")
    return 0


if __name__ == "__main__":
    sys.exit(main())

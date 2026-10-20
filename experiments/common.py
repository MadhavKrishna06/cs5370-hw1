"""Shared scaffolding for the experiments.

Two rules apply to every experiment here:

1. One variable at a time. Each config is ``BASE.with_(field=value)``, so any
   difference in the results can only come from that field.
2. Repeat every configuration across several seeds. A single run differs from
   another by initialization and shuffling alone, and on a dataset this size
   that difference is large enough to invent effects that are not there.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from hw1.config import BASELINE, Config

SEEDS: Tuple[int, ...] = (0, 1, 2)

# The reference point for the frozen-backbone experiments. Full curves are kept
# (patience equals the epoch budget) so that slow, unstable and diverging runs
# can be seen rather than cut short; the best epoch is still recorded, so model
# selection is unaffected.
CACHED_BASE = BASELINE.with_(
    mode="cached",
    hidden=512,
    n_hidden_layers=1,
    activation="relu",
    dropout=0.0,
    optimizer="adam",
    lr=1e-3,
    weight_decay=0.0,
    batch_size=64,
    epochs=40,
    early_stop_patience=40,
    probe_every=1,
)


def seeded(
    base: Config, tag: str, seeds: Sequence[int] = SEEDS, **changes
) -> Dict[str, Config]:
    """Expand one configuration into one run per seed."""
    out: Dict[str, Config] = {}
    for seed in seeds:
        out[f"{tag}_seed{seed}"] = base.with_(seed=seed, **changes)
    return out


def sweep(
    base: Config,
    field: str,
    values: Iterable,
    seeds: Sequence[int] = SEEDS,
    label=None,
) -> Dict[str, Config]:
    """Vary a single field over a list of values, across seeds."""
    label = label or (lambda value: f"{field}={value}")
    configs: Dict[str, Config] = {}
    for value in values:
        configs.update(seeded(base, label(value), seeds, **{field: value}))
    return configs


def group_by_tag(tags: Iterable[str]) -> Dict[str, List[str]]:
    """Collapse ``name_seed0``, ``name_seed1``, ... back into ``name``."""
    grouped: Dict[str, List[str]] = {}
    for tag in tags:
        base = tag.rsplit("_seed", 1)[0]
        grouped.setdefault(base, []).append(tag)
    return grouped

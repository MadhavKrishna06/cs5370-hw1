"""Turn saved run records into tables and figures.

Nothing here trains anything. The experiments write one JSON per run; this
module reads those files, groups the seeds of a configuration together, and
reports means with the seed-to-seed spread, so that a difference between
configurations can be compared against the noise within one.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES
from .train import load_group

FIGURE_DPI = 150


def _split_tag(tag: str) -> Tuple[str, Optional[int]]:
    if "_seed" in tag:
        base, seed = tag.rsplit("_seed", 1)
        return base, int(seed)
    return tag, None


def runs_table(group: str, config_fields: Sequence[str] = ()) -> pd.DataFrame:
    """One row per run: the configuration, and the metrics at its best epoch.

    ``gap`` is validation loss minus training loss at the selected epoch, which
    is the quantity the regularization experiment is about. It is read at the
    best epoch rather than the last, because that is the model that would
    actually be kept.
    """
    rows: List[dict] = []
    for tag, record in load_group(group).items():
        base, seed = _split_tag(tag)
        history = record["history"]
        best_epoch = record["best_epoch"]
        at_best = next(
            (h for h in history if h["epoch"] == best_epoch), history[-1] if history else {}
        )
        row = {
            "tag": tag,
            "base": base,
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "val_loss": record["best_val_loss"],
            "val_accuracy": record["best_val"].get("accuracy", np.nan),
            "val_precision": record["best_val"].get("precision", np.nan),
            "val_recall": record["best_val"].get("recall", np.nan),
            "val_f1": record["best_val"].get("f1", np.nan),
            "val_roc_auc": record["best_val"].get("roc_auc", np.nan),
            "train_loss_at_best": at_best.get("train_loss", np.nan),
            "gap": record["best_val_loss"] - at_best.get("train_loss", np.nan),
            "final_train_loss": history[-1]["train_loss"] if history else np.nan,
            "final_val_loss": history[-1]["val_loss"] if history else np.nan,
            "grad_norm_mean": at_best.get("grad_norm_mean", np.nan),
            "weight_norm": at_best.get("weight_norm/total", np.nan),
            "update_ratio_epoch": at_best.get("update_ratio/total", np.nan),
            "update_ratio_step": at_best.get("update_ratio/step_mean", np.nan),
            "trainable_parameters": record["trainable_parameters"],
            "train_seconds": record["train_seconds"],
            "seconds_per_epoch": record["train_seconds"] / max(len(history), 1),
            "diverged": record["diverged"],
            "stopped_early": record["stopped_early"],
        }
        for field in config_fields:
            row[field] = record["config"].get(field)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)


def history_table(group: str) -> pd.DataFrame:
    """Every epoch of every run, in long form."""
    frames: List[pd.DataFrame] = []
    for tag, record in load_group(group).items():
        base, seed = _split_tag(tag)
        frame = pd.DataFrame(record["history"])
        frame["tag"] = tag
        frame["base"] = base
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(table: pd.DataFrame, metrics: Sequence[str], by: str = "base") -> pd.DataFrame:
    """Mean and standard deviation across the seeds of each configuration."""
    aggregation = {metric: ["mean", "std"] for metric in metrics}
    out = table.groupby(by).agg(aggregation)
    out.columns = [f"{metric}_{statistic}" for metric, statistic in out.columns]
    out["n_seeds"] = table.groupby(by).size()
    return out.reset_index()


def mean_curve(
    history: pd.DataFrame, base: str, column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and spread of one column against epoch, across seeds.

    Runs stop at different epochs, so the mean is taken over whichever seeds
    are still running at each epoch rather than truncating to the shortest.
    """
    subset = history[(history["base"] == base) & history[column].notna()]
    if subset.empty:
        return np.array([]), np.array([]), np.array([])
    grouped = subset.groupby("epoch")[column]
    epochs = np.array(sorted(subset["epoch"].unique()))
    return epochs, grouped.mean().to_numpy(), grouped.std().fillna(0.0).to_numpy()


def plot_curves(
    history: pd.DataFrame,
    bases: Sequence[str],
    column: str,
    axis: plt.Axes,
    labels: Optional[Dict[str, str]] = None,
    log_y: bool = False,
    band: bool = True,
) -> None:
    labels = labels or {}
    for base in bases:
        epochs, mean, spread = mean_curve(history, base, column)
        if len(epochs) == 0:
            continue
        line, = axis.plot(epochs, mean, label=labels.get(base, base))
        if band and np.any(spread > 0):
            axis.fill_between(
                epochs, mean - spread, mean + spread, alpha=0.15, color=line.get_color()
            )
    axis.set_xlabel("epoch")
    if log_y:
        axis.set_yscale("log")
    axis.grid(alpha=0.3)


def save_figure(figure: plt.Figure, name: str) -> "Path":  # noqa: F821
    path = FIGURES / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {path.relative_to(FIGURES.parents[1])}")
    return path


def save_table(table: pd.DataFrame, name: str, float_format: str = "%.4f") -> "Path":  # noqa: F821
    path = FIGURES.parent / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=float_format)
    print(f"  wrote {path.relative_to(FIGURES.parents[1])}")
    return path


def epochs_to_reach(
    history: pd.DataFrame, base: str, column: str, threshold: float
) -> float:
    """First epoch at which a run's metric crosses a threshold, averaged over seeds.

    Used instead of "final loss" when the question is convergence speed: two
    optimizers can reach the same place while taking very different numbers of
    epochs to get there.
    """
    reached: List[float] = []
    subset = history[history["base"] == base]
    for _, run in subset.groupby("seed"):
        hit = run[run[column] <= threshold]
        reached.append(float(hit["epoch"].min()) if not hit.empty else np.nan)
    if not reached or all(np.isnan(value) for value in reached):
        return np.nan
    return float(np.nanmean(reached))

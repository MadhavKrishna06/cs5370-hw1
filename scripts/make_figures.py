"""Build every figure and table in the report from the saved run records.

    python -m scripts.make_figures            # everything that has results
    python -m scripts.make_figures --only a b # just those experiments

Reading the saved JSON rather than retraining means the figures can be redrawn
without a GPU, and that the numbers in the report are exactly the numbers the
runs produced.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw1.analysis import (
    epochs_to_reach,
    history_table,
    mean_curve,
    plot_curves,
    runs_table,
    save_figure,
    save_table,
    summarize,
)

CORE_METRICS = ("val_loss", "val_accuracy", "val_f1", "gap", "best_epoch")


def _bases(table: pd.DataFrame, prefix: str = "") -> List[str]:
    return sorted(base for base in table["base"].unique() if base.startswith(prefix))


# --------------------------------------------------------------------------- A


def figures_a() -> None:
    group = "a_activations"
    runs = runs_table(group, ("activation", "optimizer", "lr"))
    history = history_table(group)
    if runs.empty:
        print(f"{group}: no results yet")
        return
    print(f"{group}:")

    activations = ["relu", "leaky_relu", "gelu", "silu", "sigmoid"]
    arms = ["momentum", "adam"]

    # Loss curves: rows are train and validation, columns are the two optimizers.
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, arm in enumerate(arms):
        bases = [f"{arm}_{activation}" for activation in activations]
        labels = {f"{arm}_{a}": a for a in activations}
        for row, metric in enumerate(["train_loss", "val_loss"]):
            axis = axes[row][column]
            plot_curves(history, bases, metric, axis, labels)
            axis.set_title(f"{arm}: {metric.replace('_', ' ')}")
            axis.set_ylabel("BCE loss")
    axes[0][0].legend(fontsize=8)
    figure.suptitle("A. Hidden activation, under two optimizers (mean of 3 seeds)")
    save_figure(figure, "a_loss_curves")

    # Gradient reaching the first layer versus the output layer. Under plain
    # momentum this is the mechanism; under Adam the rescaling hides it.
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, arm in zip(axes[:2], arms):
        bases = [f"{arm}_{activation}" for activation in activations]
        labels = {f"{arm}_{a}": a for a in activations}
        plot_curves(history, bases, "grad_norm/linears.0.weight", axis, labels, log_y=True)
        axis.set_title(f"{arm}: gradient norm, hidden layer")
        axis.set_ylabel("L2 norm of gradient")
    axes[0].legend(fontsize=8)

    # Ratio of the two layers' gradient norms at the first probe epoch: a single
    # number for "how much signal survives the activation".
    ratios: Dict[str, List[float]] = {arm: [] for arm in arms}
    for arm in arms:
        for activation in activations:
            subset = history[
                (history["base"] == f"{arm}_{activation}") & (history["epoch"] == 0)
            ]
            hidden = subset["grad_norm/linears.0.weight"].mean()
            output = subset["grad_norm/out.weight"].mean()
            ratios[arm].append(hidden / output if output else np.nan)
    positions = np.arange(len(activations))
    width = 0.38
    for offset, arm in zip((-width / 2, width / 2), arms):
        axes[2].bar(positions + offset, ratios[arm], width, label=arm)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(activations, rotation=20, ha="right")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("hidden grad norm / output grad norm")
    axes[2].set_title("Gradient reaching the hidden layer (epoch 0)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")
    save_figure(figure, "a_gradient_flow")

    # Unit health: inactive rectified units, saturated squashing units.
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    rectified = [a for a in activations if a != "sigmoid"]
    plot_curves(
        history,
        [f"momentum_{a}" for a in rectified],
        "act/layer0/zero_fraction",
        axes[0],
        {f"momentum_{a}": a for a in rectified},
    )
    axes[0].set_title("Fraction of zero outputs (momentum)")
    axes[0].set_ylabel("fraction of activations equal to 0")
    axes[0].legend(fontsize=8)

    plot_curves(
        history,
        [f"momentum_{a}" for a in rectified],
        "act/layer0/dead_unit_fraction",
        axes[1],
        {f"momentum_{a}": a for a in rectified},
    )
    axes[1].set_title("Units inactive for every example in the batch")
    axes[1].set_ylabel("fraction of units")

    for arm in arms:
        epochs, mean, _ = mean_curve(history, f"{arm}_sigmoid", "act/layer0/saturated_fraction")
        if len(epochs):
            axes[2].plot(epochs, mean, label=f"{arm}: |pre-activation| > 4")
        epochs, mean, _ = mean_curve(history, f"{arm}_sigmoid", "act/layer0/mean_local_slope")
        if len(epochs):
            axes[2].plot(epochs, mean, linestyle="--", label=f"{arm}: mean sigmoid slope")
    axes[2].set_xlabel("epoch")
    axes[2].set_title("Sigmoid saturation")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    save_figure(figure, "a_unit_health")

    table = summarize(runs, CORE_METRICS)
    table["grad_ratio_epoch0"] = [
        ratios[base.split("_")[0]][activations.index(base.split("_", 1)[1])]
        if base.split("_", 1)[1] in activations
        else np.nan
        for base in table["base"]
    ]
    save_table(table, "a_activations")


# --------------------------------------------------------------------------- B


def figures_b() -> None:
    group = "b_learning_rate"
    runs = runs_table(group, ("lr",))
    history = history_table(group)
    if runs.empty:
        print(f"{group}: no results yet")
        return
    print(f"{group}:")

    bases = sorted(runs["base"].unique(), key=lambda base: float(base[2:]))
    labels = {base: f"lr = {float(base[2:]):.0e}" for base in bases}

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_curves(history, bases, "train_loss", axes[0], labels, log_y=True)
    axes[0].set_title("Training loss")
    axes[0].set_ylabel("BCE loss (log scale)")
    axes[0].legend(fontsize=8)
    plot_curves(history, bases, "val_loss", axes[1], labels, log_y=True)
    axes[1].set_title("Validation loss")
    plot_curves(history, bases, "update_ratio/step_mean", axes[2], labels, log_y=True)
    axes[2].set_title("Step size relative to weight size")
    axes[2].set_ylabel("mean per-step ||update|| / ||weight||")
    figure.suptitle("B. Learning rate, plain SGD (mean of 3 seeds)")
    save_figure(figure, "b_loss_curves")

    # Best validation loss against learning rate: too small underfits within
    # the epoch budget, too large is unstable, and the middle is flat-ish.
    summary = summarize(runs, CORE_METRICS + ("update_ratio_step", "grad_norm_mean"))
    summary["lr"] = [float(base[2:]) for base in summary["base"]]
    summary = summary.sort_values("lr")
    threshold = 0.35
    summary["epochs_to_val_loss"] = [
        epochs_to_reach(history, base, "val_loss", threshold) for base in summary["base"]
    ]
    diverged = runs.groupby("base")["diverged"].any()
    summary["diverged"] = [bool(diverged.get(base, False)) for base in summary["base"]]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].errorbar(
        summary["lr"], summary["val_loss_mean"], yerr=summary["val_loss_std"], marker="o"
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel("best validation loss")
    axes[0].set_title("Best validation loss reached")
    axes[0].grid(alpha=0.3)

    axes[1].plot(summary["lr"], summary["epochs_to_val_loss"], marker="o")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel(f"epochs to reach val loss {threshold}")
    axes[1].set_title("Convergence speed")
    axes[1].grid(alpha=0.3)

    axes[2].plot(summary["lr"], summary["update_ratio_step_mean"], marker="o")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("learning rate")
    axes[2].set_ylabel("||update|| / ||weight|| per step")
    axes[2].set_title("How far each step moves the weights")
    axes[2].grid(alpha=0.3)
    figure.suptitle("B. Summary against learning rate")
    save_figure(figure, "b_summary")

    save_table(summary, "b_learning_rate")


# --------------------------------------------------------------------------- C


def figures_c() -> None:
    group = "c_optimizers"
    runs = runs_table(group, ("optimizer", "lr"))
    history = history_table(group)
    if runs.empty:
        print(f"{group}: no results yet")
        return
    print(f"{group}:")

    optimizers = ["sgd", "momentum", "rmsprop", "adam"]
    for arm, title in (("shared", "shared learning rate 1e-3"), ("tuned", "tuned learning rate")):
        bases = [f"{arm}_{optimizer}" for optimizer in optimizers]
        if not any(base in set(runs["base"]) for base in bases):
            continue
        chosen = {
            f"{arm}_{optimizer}": runs.loc[runs["base"] == f"{arm}_{optimizer}", "lr"].iloc[0]
            for optimizer in optimizers
            if (runs["base"] == f"{arm}_{optimizer}").any()
        }
        labels = {base: f"{base.split('_', 1)[1]} (lr {lr:.0e})" for base, lr in chosen.items()}

        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_curves(history, bases, "train_loss", axes[0], labels, log_y=True)
        axes[0].set_title("Training loss")
        axes[0].set_ylabel("BCE loss (log scale)")
        axes[0].legend(fontsize=8)
        plot_curves(history, bases, "val_loss", axes[1], labels, log_y=True)
        axes[1].set_title("Validation loss")
        plot_curves(history, bases, "update_ratio/step_mean", axes[2], labels, log_y=True)
        axes[2].set_title("Step size relative to weight size")
        axes[2].set_ylabel("mean per-step ||update|| / ||weight||")
        figure.suptitle(f"C. Optimizers, {title}")
        save_figure(figure, f"c_{arm}_curves")

    # Sensitivity: the whole tuning sweep, which is the real difference between
    # these optimizers on this problem.
    tuning = runs_table("c_optimizers_tuning", ("optimizer", "lr"))
    if not tuning.empty:
        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        for optimizer in optimizers:
            subset = tuning[tuning["optimizer"] == optimizer].sort_values("lr")
            if subset.empty:
                continue
            axis.plot(subset["lr"], subset["val_loss"], marker="o", label=optimizer)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("learning rate")
        axis.set_ylabel("best validation loss (seed 99)")
        axis.set_title("C. Sensitivity to the learning rate")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        save_figure(figure, "c_sensitivity")
        save_table(tuning[["tag", "optimizer", "lr", "val_loss", "val_accuracy", "val_f1"]], "c_tuning")

    summary = summarize(runs, CORE_METRICS + ("update_ratio_step", "train_seconds"))
    summary["epochs_to_val_loss_0.32"] = [
        epochs_to_reach(history, base, "val_loss", 0.32) for base in summary["base"]
    ]
    save_table(summary, "c_optimizers")


# --------------------------------------------------------------------------- D


def figures_d() -> None:
    group = "d_regularization"
    runs = runs_table(group, ("dropout", "weight_decay", "train_subset"))
    history = history_table(group)
    if runs.empty:
        print(f"{group}: no results yet")
        return
    print(f"{group}:")

    full = runs[runs["train_subset"].isna()]
    subset = runs[runs["train_subset"].notna()]

    def strength_panel(frame: pd.DataFrame, field: str, name: str, log_x: bool) -> None:
        summary = summarize(frame, CORE_METRICS + ("weight_norm",))
        summary[field] = [
            frame.loc[frame["base"] == base, field].iloc[0] for base in summary["base"]
        ]
        summary = summary.sort_values(field)
        values = summary[field].to_numpy(dtype=float)
        x = values
        tick_labels = None
        if log_x:
            # Zero cannot be drawn on a log axis; place it a decade below the
            # smallest non-zero value and label that tick "0" so the figure does
            # not imply a value that was never run.
            nonzero = values[values > 0]
            placeholder = nonzero.min() / 10 if len(nonzero) else 1e-6
            x = np.where(values == 0, placeholder, values)
            tick_labels = ["0" if value == 0 else f"$10^{{{np.log10(value):.0f}}}$" for value in values]

        figure, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].errorbar(x, summary["val_loss_mean"], yerr=summary["val_loss_std"], marker="o")
        axes[0].set_ylabel("best validation loss")
        axes[0].set_title("Validation loss")
        axes[1].errorbar(x, summary["gap_mean"], yerr=summary["gap_std"], marker="o", color="tab:red")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("val loss - train loss at best epoch")
        axes[1].set_title("Generalization gap")
        axes[2].errorbar(x, summary["val_f1_mean"], yerr=summary["val_f1_std"], marker="o", color="tab:green")
        axes[2].set_ylabel("best validation F1")
        axes[2].set_title("Validation F1")
        axes[3].plot(x, summary["weight_norm_mean"], marker="o", color="tab:purple")
        axes[3].set_ylabel("L2 norm of weights at best epoch")
        axes[3].set_title("Weight norm")
        for axis in axes:
            axis.set_xlabel(name)
            axis.grid(alpha=0.3)
            if log_x:
                axis.set_xscale("log")
                axis.set_xticks(x)
                axis.set_xticklabels(tick_labels, fontsize=8)
                axis.minorticks_off()
        figure.suptitle(f"D. {name} (mean of 3 seeds)")
        save_figure(figure, f"d_{field}_summary")
        save_table(summary, f"d_{field}")

    # The two arms share a zero point, but it was run once per arm under its own
    # tag; selecting by tag prefix keeps that point from being plotted twice.
    dropout_arm = full[full["base"].str.startswith("dropout")]
    strength_panel(dropout_arm, "dropout", "dropout probability", log_x=False)
    l2_arm = full[full["base"].str.startswith("l2_")]
    strength_panel(l2_arm, "weight_decay", "L2 weight decay", log_x=True)

    # A few curves, to show what the summary points are made of: without
    # regularization the two losses separate, with too much neither moves.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, (bases, title) in zip(
        axes,
        (
            (["dropout0", "dropout0.5", "dropout0.9"], "dropout"),
            (["l2_0", "l2_0.01", "l2_1"], "L2"),
        ),
    ):
        present = [base for base in bases if base in set(full["base"])]
        for base in present:
            epochs, mean, _ = mean_curve(history, base, "train_loss")
            line, = axis.plot(epochs, mean, linestyle="--", label=f"{base} train")
            epochs, mean, _ = mean_curve(history, base, "val_loss")
            axis.plot(epochs, mean, color=line.get_color(), label=f"{base} val")
        axis.set_xlabel("epoch")
        axis.set_ylabel("BCE loss")
        axis.set_title(f"{title}: training (dashed) and validation (solid)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    save_figure(figure, "d_curves")

    if not subset.empty:
        summary = summarize(subset, CORE_METRICS)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        positions = np.arange(len(summary))
        axes[0].bar(positions, summary["gap_mean"], yerr=summary["gap_std"], color="tab:red")
        axes[0].set_ylabel("generalization gap")
        axes[1].bar(positions, summary["val_f1_mean"], yerr=summary["val_f1_std"], color="tab:green")
        axes[1].set_ylabel("best validation F1")
        for axis in axes:
            axis.set_xticks(positions)
            axis.set_xticklabels(summary["base"], rotation=25, ha="right", fontsize=8)
            axis.grid(alpha=0.3, axis="y")
        figure.suptitle("D. The same techniques on a 500-image training subset")
        save_figure(figure, "d_subset")
        save_table(summary, "d_subset")


# --------------------------------------------------------------------------- E


def figures_e() -> None:
    group = "e_finetuning"
    runs = runs_table(group, ("unfreeze_from", "augment"))
    history = history_table(group)
    if runs.empty:
        print(f"{group}: no results yet")
        return
    print(f"{group}:")

    plain = runs[~runs["augment"]]
    order = ["frozen", "unfreeze8", "unfreeze7", "unfreeze6", "unfreeze5"]
    bases = [base for base in order if base in set(plain["base"])]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_curves(history, bases, "train_loss", axes[0], log_y=False)
    axes[0].set_title("Training loss")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend(fontsize=8)
    plot_curves(history, bases, "val_loss", axes[1])
    axes[1].set_title("Validation loss")
    plot_curves(history, bases, "val_f1", axes[2])
    axes[2].set_title("Validation F1")
    figure.suptitle("E. How much of the backbone is trainable (mean of 2 seeds)")
    save_figure(figure, "e_curves")

    summary = summarize(
        plain, CORE_METRICS + ("val_accuracy", "trainable_parameters", "seconds_per_epoch", "train_seconds")
    )
    summary = summary.set_index("base").loc[bases].reset_index()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].errorbar(
        summary["trainable_parameters_mean"],
        summary["val_f1_mean"],
        yerr=summary["val_f1_std"],
        marker="o",
    )
    for _, row in summary.iterrows():
        axes[0].annotate(
            row["base"],
            (row["trainable_parameters_mean"], row["val_f1_mean"]),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=8,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("trainable parameters")
    axes[0].set_ylabel("best validation F1")
    axes[0].set_title("Performance against trainable parameters")

    axes[1].errorbar(
        summary["seconds_per_epoch_mean"],
        summary["val_f1_mean"],
        yerr=summary["val_f1_std"],
        marker="o",
        color="tab:orange",
    )
    for _, row in summary.iterrows():
        axes[1].annotate(
            row["base"],
            (row["seconds_per_epoch_mean"], row["val_f1_mean"]),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=8,
        )
    axes[1].set_xlabel("seconds per epoch")
    axes[1].set_ylabel("best validation F1")
    axes[1].set_title("Performance against cost")

    # Not the usual gap bar: every arm here peaks at epoch 0 or 1, which makes a
    # gap taken at the best epoch an artifact of averaging the training loss
    # over an epoch the model was still improving through. What the arms really
    # differ in is how fast they overfit after the peak, so that is plotted.
    positions = np.arange(len(summary))
    width = 0.38
    LATE_EPOCH = 4
    best_loss, late_loss = [], []
    for base in summary["base"]:
        best_loss.append(plain.loc[plain["base"] == base, "val_loss"].mean())
        late = history[(history["base"] == base) & (history["epoch"] == LATE_EPOCH)]
        late_loss.append(late["val_loss"].mean() if not late.empty else np.nan)
    axes[2].bar(positions - width / 2, best_loss, width, label="at best epoch")
    axes[2].bar(positions + width / 2, late_loss, width, label=f"at epoch {LATE_EPOCH}")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(summary["base"], rotation=25, ha="right", fontsize=8)
    axes[2].set_ylabel("validation loss")
    axes[2].set_title("How fast each arm overfits past its peak")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.suptitle("E. What fine-tuning buys, and what it costs")
    save_figure(figure, "e_tradeoff")
    save_table(summary, "e_finetuning")

    augmented = runs[runs["augment"]]
    if not augmented.empty:
        pairs = [("frozen", "frozen_augment"), ("unfreeze7", "unfreeze7_augment")]
        pairs = [(a, b) for a, b in pairs if b in set(runs["base"]) and a in set(runs["base"])]
        if pairs:
            combined = summarize(runs, CORE_METRICS).set_index("base")
            figure, axes = plt.subplots(1, 2, figsize=(11, 4))
            positions = np.arange(len(pairs))
            width = 0.38
            for offset, which, label in ((-width / 2, 0, "no augmentation"), (width / 2, 1, "augmentation")):
                axes[0].bar(
                    positions + offset,
                    [combined.loc[pair[which], "val_f1_mean"] for pair in pairs],
                    width,
                    yerr=[combined.loc[pair[which], "val_f1_std"] for pair in pairs],
                    label=label,
                )
                axes[1].bar(
                    positions + offset,
                    [combined.loc[pair[which], "val_loss_mean"] for pair in pairs],
                    width,
                    yerr=[combined.loc[pair[which], "val_loss_std"] for pair in pairs],
                    label=label,
                )
            axes[0].set_ylabel("best validation F1")
            axes[1].set_ylabel("best validation loss")
            for axis in axes:
                axis.set_xticks(positions)
                axis.set_xticklabels([pair[0] for pair in pairs])
                axis.grid(alpha=0.3, axis="y")
                axis.legend(fontsize=8)
            figure.suptitle("D/E. Data augmentation, with the backbone frozen and partly unfrozen")
            save_figure(figure, "e_augmentation")
            save_table(combined.reset_index(), "e_augmentation")


BUILDERS = {"a": figures_a, "b": figures_b, "c": figures_c, "d": figures_d, "e": figures_e}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS))
    arguments = parser.parse_args(argv)
    for key in arguments.only:
        BUILDERS[key]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

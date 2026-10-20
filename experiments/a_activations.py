"""Experiment A: how the hidden activation affects training.

Question
    Does replacing the hidden activation change *how* the head trains, not just
    where it ends up? Specifically: does a squashing activation starve the first
    layer of gradient, and do rectified units go permanently inactive?

Design
    One hidden layer of 512 units, everything else held fixed, five activations
    compared: ReLU, Leaky ReLU, GELU, SiLU and sigmoid. Three seeds each.

    The comparison is run twice, under two different optimizers, because the
    optimizer determines whether a gradient-magnitude difference can even show
    up in the weights:

      * SGD with momentum applies the raw gradient, so a layer receiving a
        small gradient takes small steps.
      * Adam divides by a running estimate of the gradient magnitude, which
        rescales small and large gradients towards the same step size.

    Running both separates "the activation changes the gradient" from "the
    activation changes the outcome". Measuring only under Adam would risk
    concluding that the activation does not matter, when what actually happened
    is that the optimizer compensated for it.

Measured
    Per-layer gradient norms, so that layer 0 and the output layer can be
    compared directly; fraction of zero outputs and of permanently inactive
    units for the rectified activations; fraction of saturated pre-activations
    and the mean local slope for sigmoid; update-to-weight ratio; loss curves.
"""

from __future__ import annotations

import sys

from hw1.train import run_group

from .common import CACHED_BASE, seeded

ACTIVATIONS = ("relu", "leaky_relu", "gelu", "silu", "sigmoid")

# Chosen so that both optimizers train the ReLU baseline to a sensible place:
# anything that fails here fails because of the activation, not the step size.
OPTIMIZER_ARMS = {
    "momentum": dict(optimizer="momentum", lr=1e-2, momentum=0.9),
    "adam": dict(optimizer="adam", lr=1e-3),
}


def configs() -> dict:
    out = {}
    for arm, settings in OPTIMIZER_ARMS.items():
        for activation in ACTIVATIONS:
            out.update(
                seeded(
                    CACHED_BASE,
                    f"{arm}_{activation}",
                    activation=activation,
                    **settings,
                )
            )
    return out


def main() -> int:
    run_group("a_activations", configs(), verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

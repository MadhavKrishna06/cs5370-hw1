# CS 5370 Homework 1 — transfer learning with EfficientNet-B0 on Pascal VOC 2007

Binary task: does an image contain a person? The visual representation comes
from EfficientNet-B0 pretrained on ImageNet; its 1000-way classifier is removed
and replaced with a custom head trained with `BCEWithLogitsLoss` on one logit.

## Setup

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch is installed from the CUDA 12.4 index:

```
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Download the two VOC 2007 archives into `data/`:

- `VOCtrainval_06-Nov-2007.tar`
- `VOCtest_06-Nov-2007.tar`

from `https://pjreddie.com/media/files/` (a mirror of
`http://host.robots.ox.ac.uk/pascal/VOC/voc2007/`; the md5 checks in
`scripts/prepare_data.py` confirm the files are identical to the originals).

## Running

```
python -m scripts.prepare_data        # verify checksums, extract, report class balance
python -m scripts.build_cache         # frozen-backbone features + linear-probe sanity check
python -m experiments.a_activations   # A: hidden activation functions
python -m experiments.b_learning_rate # B: learning rate
python -m experiments.c_optimizers    # C: SGD, momentum, RMSProp, Adam
python -m experiments.d_regularization# D: dropout and L2
python -m experiments.e_finetuning    # E: frozen backbone vs partial fine-tuning
python -m scripts.make_figures        # all figures and tables from saved results
python -m scripts.final_model         # train the selected model, evaluate on test once
```

Experiments A–D read the cached features and take minutes in total. Experiment E
and the final model use the image pipeline and take a few hours on a GTX 1650.

## Layout

| path | what it holds |
| --- | --- |
| `hw1/config.py` | `Config`: every setting of a run, frozen so experiments cannot mutate the baseline |
| `hw1/data.py` | VOC 2007 as a binary person-present problem; transforms; split report |
| `hw1/features.py` | the pretrained backbone, and the frozen-feature cache |
| `hw1/model.py` | the custom head, and backbone + head for fine-tuning |
| `hw1/probes.py` | gradient norms, update norms, saturation and inactive-unit statistics |
| `hw1/train.py` | the single training loop every experiment calls |
| `hw1/metrics.py` | accuracy, precision, recall, F1, confusion matrix, ROC-AUC |
| `hw1/analysis.py` | saved runs to tables and figures |
| `experiments/` | one module per investigation, each a sweep over a single field |
| `scripts/` | data preparation, cache building, figures, final evaluation |
| `results/<group>/<tag>.json` | one record per run: config, per-epoch history, best metrics |
| `report/report.md` | the experimental report, source of truth |
| `report/CS5370_Homework1_Report.docx` | the same report with figures embedded, for submission |
| `report/figures`, `report/tables` | generated; nothing is drawn by hand |

## Design notes

**Splits.** VOC's own `train` (2501) fits parameters, `val` (2510) drives model
selection and early stopping, `test` (4952) is read once by
`scripts/final_model.py`. Person-present rate is 0.43, 0.41, 0.42.

**Labels** come from `ImageSets/Main/person_<split>.txt`, VOC's own per-class
annotation lists. Cases flagged difficult are counted as person-present;
`read_person_labels(..., difficult_as_positive=False)` drops them instead.

**The feature cache.** While the backbone is frozen it computes the same
1280-vector for an image on every epoch of every run, so `scripts/build_cache`
computes it once (131 s for all 9963 images, 51 MB) and the head-only
experiments read vectors instead of images. A head run then takes about a second
per epoch instead of fifty, which is what makes a seeded sweep affordable. The
cache is only valid with a frozen backbone and deterministic preprocessing, so
augmentation and fine-tuning bypass it.

**Controls.** Each experiment builds its configs as `BASE.with_(field=value)`,
and every configuration is repeated over three seeds (two for the expensive
image runs). Reported differences are compared against the spread across seeds.

**Frozen BatchNorm.** A frozen block left in `train()` mode would keep updating
its BatchNorm running statistics, so its output would drift even though no
weight changes. `BackboneWithHead.set_train_mode` holds frozen blocks in `eval()`
to prevent that.

**A hardware fault worth knowing about.** On the GTX 1650 (Turing TU117) used
here, the half-precision convolution kernel EfficientNet-B0 selects returns NaN
at batch size 64, while 32, 48 and 96 are all fine and full precision is
unaffected at every size. It is a kernel-selection fault, not an overflow. All
image-pipeline results in this report use batch size 32 and are unaffected.
`train()` now checks one real batch before training and falls back to full
precision if it comes back non-finite, recording `amp_fell_back` in the run
record. That check runs with the model in `eval()` mode on purpose: in `train()`
mode a NaN batch would be folded into the backbone's BatchNorm running
statistics and every later evaluation would return NaN while training still
looked healthy.

**Mixed precision** is used for the image pipeline only. It rescales gradients,
so the probes call `scaler.unscale_` before measuring; the cached pipeline is
already fast and runs in full precision so its diagnostics need no correction.

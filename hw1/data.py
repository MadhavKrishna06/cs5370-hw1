"""Pascal VOC 2007 framed as a binary person-present / person-absent problem.

VOC ships one annotation file per class per split, so the labels used here are
the ones the dataset authors defined rather than something re-derived from the
bounding-box XML.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .paths import VOC_ROOT

VOC2007 = VOC_ROOT / "VOCdevkit" / "VOC2007"
SPLITS = ("train", "val", "test")

# The statistics the pretrained backbone was normalized with. Using anything
# else would feed the network inputs it never saw during pretraining.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def read_person_labels(
    split: str, difficult_as_positive: bool = True
) -> Tuple[List[str], torch.Tensor]:
    """Read ``ImageSets/Main/person_<split>.txt``.

    Each line is ``<image id> <flag>`` where the flag is 1 (person present),
    -1 (person absent) or 0 (present but marked difficult). Difficult cases are
    counted as positive by default; passing ``difficult_as_positive=False``
    drops them instead, which is the stricter VOC evaluation convention.
    """
    path = VOC2007 / "ImageSets" / "Main" / f"person_{split}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m scripts.prepare_data` first."
        )

    ids: List[str] = []
    labels: List[float] = []
    for line in path.read_text().strip().splitlines():
        image_id, raw_flag = line.split()
        flag = int(raw_flag)
        if flag == 0 and not difficult_as_positive:
            continue
        ids.append(image_id)
        labels.append(0.0 if flag == -1 else 1.0)
    return ids, torch.tensor(labels, dtype=torch.float32)


def eval_transform() -> transforms.Compose:
    """Deterministic preprocessing. Used for validation, test, and the cache."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def train_transform(augment: bool) -> transforms.Compose:
    """Training preprocessing. Identical to ``eval_transform`` unless augmenting,
    so that turning augmentation on is the only change between the two arms of
    the augmentation experiment."""
    if not augment:
        return eval_transform()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class VOCPerson(Dataset):
    """VOC 2007 images paired with the binary person label."""

    def __init__(
        self,
        split: str,
        transform: Optional[transforms.Compose] = None,
        difficult_as_positive: bool = True,
        subset: Optional[int] = None,
        subset_seed: int = 0,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.ids, self.labels = read_person_labels(split, difficult_as_positive)
        if subset is not None and subset < len(self.ids):
            generator = torch.Generator().manual_seed(subset_seed)
            keep = torch.randperm(len(self.ids), generator=generator)[:subset]
            self.ids = [self.ids[i] for i in keep.tolist()]
            self.labels = self.labels[keep]
        self.transform = transform if transform is not None else eval_transform()
        self.image_dir = VOC2007 / "JPEGImages"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image = Image.open(self.image_dir / f"{self.ids[index]}.jpg").convert("RGB")
        return self.transform(image), self.labels[index]

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean())


def image_loader(
    split: str,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
    num_workers: int = 4,
    subset: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    dataset = VOCPerson(
        split,
        transform=train_transform(augment) if shuffle else eval_transform(),
        subset=subset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def describe_splits(difficult_as_positive: bool = True) -> None:
    """Print split sizes and class balance. Sanity check before any training."""
    print(f"{'split':<8}{'images':>8}{'person':>9}{'no person':>11}{'pos rate':>10}")
    for split in SPLITS:
        _, labels = read_person_labels(split, difficult_as_positive)
        positives = int(labels.sum())
        total = len(labels)
        print(
            f"{split:<8}{total:>8}{positives:>9}{total - positives:>11}"
            f"{positives / total:>10.3f}"
        )

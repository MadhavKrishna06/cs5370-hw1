"""Verify and extract the Pascal VOC 2007 archives, then report class balance.

Run this once:

    python -m scripts.prepare_data

The two tar files are expected in ``data/``. They are checked against the
checksums published with the dataset, so a truncated or interrupted download is
caught here rather than halfway through training.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

from hw1.data import VOC2007, describe_splits
from hw1.paths import DATA

ARCHIVES = {
    "VOCtrainval_06-Nov-2007.tar": "c52e279531787c972589f7e41ab4ae64",
    "VOCtest_06-Nov-2007.tar": "b6e924de25625d8de591ea690078ad9f",
}

SOURCE = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"


def md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(path: Path) -> None:
    print(f"extracting {path.name} ...")
    with tarfile.open(path) as archive:
        try:
            archive.extractall(DATA, filter="data")
        except TypeError:
            # Older Python without the extraction filter.
            archive.extractall(DATA)


def main() -> int:
    missing = [name for name in ARCHIVES if not (DATA / name).exists()]
    if missing:
        print("missing archive(s):")
        for name in missing:
            print(f"  {name}   download from {SOURCE}{name}")
        return 1

    for name, expected in ARCHIVES.items():
        path = DATA / name
        print(f"checking {name} ({path.stat().st_size / 1e6:.0f} MB) ...", end=" ", flush=True)
        actual = md5(path)
        if actual != expected:
            print(f"FAILED\n  expected md5 {expected}\n  got          {actual}")
            print("  the download is incomplete or corrupt; delete the file and retry")
            return 1
        print("ok")
        extract(path)

    if not VOC2007.exists():
        print(f"extraction did not produce {VOC2007}")
        return 1

    image_count = len(list((VOC2007 / "JPEGImages").glob("*.jpg")))
    print(f"\n{VOC2007} ready, {image_count} images\n")
    describe_splits()
    print("\nLabels come from VOC's own ImageSets/Main/person_<split>.txt files.")
    print("Difficult cases (flag 0) are counted as person-present; see hw1/data.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

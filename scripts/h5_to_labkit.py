"""
Export segmentation result .h5 files for review in Labkit: <stem>.tiff (image) + <stem>.labeling (instances).

    python scripts/h5_to_labkit.py results/Promotion_                 # every *.h5 below the folder
    python scripts/h5_to_labkit.py results/Promotion_/200/batch_02/01.checkpoint_060.h5

Existing .tiff / .labeling files are skipped unless --overwrite is given.
"""

import argparse
import time
from pathlib import Path

from sam2.addons import h5_to_labkit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help=".h5 files or folders (searched recursively)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to each h5)")
    ap.add_argument("--image-key", default="image")
    ap.add_argument("--labels-key", default="labels")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = sorted({f for p in args.paths for f in ([p] if p.is_file() else p.rglob("*.h5"))})
    print(f"{len(files)} h5 file(s)")
    for k, f in enumerate(files, 1):
        t0 = time.time()
        try:
            tif, lab = h5_to_labkit(f, args.out, args.image_key, args.labels_key, args.overwrite)
        except KeyError as e:
            print(f"[{k}/{len(files)}] {f}: skipped, no dataset {e}")
            continue
        print(f"[{k}/{len(files)}] {f} -> {tif.name}, {lab.name}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

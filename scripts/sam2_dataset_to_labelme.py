"""
Write LabelMe files for a SAM 2 dataset folder created by h5_to_sam2_dataset.py.

    python scripts/sam2_dataset_to_labelme.py data/fibre_ds_512

Then in LabelMe: Open Dir → <dataset>/images, File → Change Output Dir → <dataset>/labelme
(or: labelme <dataset>/images --output <dataset>/labelme --nodata)
"""

import argparse
from pathlib import Path

from sam2.addons import sa1b_to_labelme


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="dataset root with images/ and annotations/")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: <dataset>/labelme)")
    parser.add_argument("--label", default="fibre", help="label name of the polygons (default: fibre)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="polygon simplification in px, 0 = every contour pixel (default: 0.5)")
    args = parser.parse_args()

    out = args.out or args.dataset / "labelme"
    n = sa1b_to_labelme(args.dataset, out, label=args.label, tolerance=args.tolerance)
    print(f"Wrote {n} LabelMe files → {out}")


if __name__ == "__main__":
    main()

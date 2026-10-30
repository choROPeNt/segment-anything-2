"""
Convert specimen .h5 results into a SAM 2 training dataset (SA-1B layout).

Each .h5 is cut into square windows; windows with enough ROI coverage are written as
images/{name}.jpg + annotations/{name}.json (COCO RLE) to the output folder, plus a
windows.csv manifest and a {split}.txt file list.  Running the script again on the same
output folder appends (numbering continues), so splits can be built specimen by specimen:

    python scripts/h5_to_sam2_dataset.py results/Promotion/160/batch_02/01.h5 -o data/fibre_ds --split train
    python scripts/h5_to_sam2_dataset.py results/Promotion/200/batch_02/01.h5 -o data/fibre_ds --split val

Analyse only (no files written):

    python scripts/h5_to_sam2_dataset.py results/Promotion/160/batch_02/01.h5 -o data/fibre_ds --dry-run
"""

import argparse
from pathlib import Path

import pandas as pd

from sam2.addons import export_specimen_to_sa1b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", nargs="+", type=Path, help="specimen .h5 file(s)")
    parser.add_argument("-o", "--out", type=Path, required=True, help="dataset root (images/, annotations/)")
    parser.add_argument("--split", default="train", help="split label for these files (default: train)")
    parser.add_argument("--specimen", default=None,
                        help="specimen name (single input only; default: <grandparent>-<parent>-<stem>)")
    parser.add_argument("--size", type=int, default=1024, help="window size in px (default: 1024)")
    parser.add_argument("--stride", type=int, default=512, help="window stride in px (default: 512)")
    parser.add_argument("--min-roi-cov", type=float, default=0.8,
                        help="min. fraction of the window inside the ROI mask (default: 0.8)")
    parser.add_argument("--min-visible-frac", type=float, default=0.3,
                        help="min. visible fraction of an instance cut by the window edge (default: 0.3)")
    parser.add_argument("--min-area", type=int, default=64, help="min. visible instance area in px (default: 64)")
    parser.add_argument("--blank-outside-roi", action="store_true",
                        help="fill pixels outside the ROI with the background colour")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--dry-run", action="store_true", help="only analyse windows, write nothing")
    args = parser.parse_args()

    if args.specimen and len(args.h5) > 1:
        parser.error("--specimen can only be used with a single .h5")

    frames = []
    for h5 in args.h5:
        df = export_specimen_to_sa1b(
            h5, args.out,
            specimen=args.specimen,
            split=args.split,
            size=args.size,
            stride=args.stride,
            min_roi_cov=args.min_roi_cov,
            min_visible_frac=args.min_visible_frac,
            min_area=args.min_area,
            blank_outside_roi=args.blank_outside_roi,
            jpeg_quality=args.jpeg_quality,
            dry_run=args.dry_run,
        )
        frames.append(df)
        if len(df):
            print(f"{h5}: {len(df)} windows, {df['n_kept'].sum():,} instances "
                  f"(dropped: {df['n_dropped_cut'].sum():,} cut by window, {df['n_dropped_roi'].sum():,} cut by ROI), "
                  f"FVG {df['fvg_roi'].median():.3f} (median)")
        else:
            print(f"{h5}: no window with ROI coverage >= {args.min_roi_cov}")

    df = pd.concat(frames, ignore_index=True)
    action = "analysed (dry run)" if args.dry_run else f"written to {args.out}  [split: {args.split}]"
    print(f"\nTotal: {len(df)} windows, {df['n_kept'].sum() if len(df) else 0:,} instances {action}")


if __name__ == "__main__":
    main()

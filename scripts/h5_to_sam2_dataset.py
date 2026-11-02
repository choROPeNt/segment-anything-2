"""
Convert specimen .h5 results into a SAM 2 training dataset (SA-1B layout).

Each .h5 is cut into square windows; windows with enough ROI coverage are written as
images/{name}.jpg + annotations/{name}.json (COCO RLE) to the output folder, plus a
windows.csv manifest and one {split}.txt file list per split.  Running the script again on
the same output folder appends (numbering continues).

Split by specimen (preferred as soon as several specimens are available):

    python scripts/h5_to_sam2_dataset.py results/Promotion/160/batch_02/160-02_01.h5 -o data/fibre_ds --split train
    python scripts/h5_to_sam2_dataset.py results/Promotion/200/batch_02/200-02_01.h5 -o data/fibre_ds --split val

Spatial split within one specimen (blocks along the long axis, guard band in between):

    python scripts/h5_to_sam2_dataset.py results/Promotion/160/batch_02/160-02_01.h5 -o data/fibre_ds --split auto \\
        --val-frac 0.2 --n-blocks 10 --guard 128

Train and val never share pixels or fibres: a specimen can only be exported once per dataset,
and with --split auto every region is tiled on its own.  The whole dataset is checked for
overlapping train / val windows at the end.

Analyse only (no files written):

    python scripts/h5_to_sam2_dataset.py results/Promotion/160/batch_02/160-02_01.h5 -o data/fibre_ds --split auto --dry-run
"""

import argparse
from pathlib import Path

import pandas as pd

from sam2.addons import check_split_disjoint, export_specimen_to_sa1b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", nargs="+", type=Path, help="specimen .h5 file(s)")
    parser.add_argument("-o", "--out", type=Path, required=True, help="dataset root (images/, annotations/)")
    parser.add_argument("--split", default="train",
                        help="split label for these files, or 'auto' for a spatial train/val split (default: train)")
    parser.add_argument("--specimen", default=None,
                        help="specimen name (single input only; default: <grandparent>-<parent>-<stem>)")
    parser.add_argument("--size", type=int, default=1024, help="window size in px (default: 1024)")
    parser.add_argument("--stride", type=int, default=512, help="window stride in px (default: 512)")
    parser.add_argument("--min-roi-cov", type=float, default=0.8,
                        help="min. fraction of the window inside the (padded) ROI mask (default: 0.8)")
    parser.add_argument("--roi-padding", type=int, default=0,
                        help="grow the ROI by this many px for the window selection, so windows also cover the "
                             "roving border / sparse regions; labels are unchanged (default: 0)")
    parser.add_argument("--min-visible-frac", type=float, default=0.3,
                        help="min. visible fraction of an instance cut by the window edge (default: 0.3)")
    parser.add_argument("--min-area", type=int, default=64, help="min. visible instance area in px (default: 64)")
    parser.add_argument("--blank-outside-roi", action="store_true",
                        help="fill pixels outside the ROI with the background colour")
    parser.add_argument("--jpeg-quality", type=int, default=95)

    auto = parser.add_argument_group("spatial split (--split auto)")
    auto.add_argument("--val-frac", type=float, default=0.2, help="fraction of blocks used for val (default: 0.2)")
    auto.add_argument("--n-blocks", type=int, default=10, help="blocks along the long axis (default: 10)")
    auto.add_argument("--guard", type=int, default=128,
                      help="gap in px between train and val regions, > fibre diameter (default: 128)")
    auto.add_argument("--seed", type=int, default=0, help="seed for choosing the val blocks (default: 0)")

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
            roi_padding=args.roi_padding,
            min_visible_frac=args.min_visible_frac,
            min_area=args.min_area,
            blank_outside_roi=args.blank_outside_roi,
            jpeg_quality=args.jpeg_quality,
            val_frac=args.val_frac,
            n_blocks=args.n_blocks,
            guard=args.guard,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        frames.append(df)
        if not len(df):
            print(f"{h5}: no window with ROI coverage >= {args.min_roi_cov}")
            continue
        for split, d in df.groupby("split"):
            print(f"{h5} [{split}]: {len(d)} windows, {d['n_kept'].sum():,} instances "
                  f"(dropped: {d['n_dropped_cut'].sum():,} cut by window, {d['n_dropped_roi'].sum():,} cut by ROI), "
                  f"FVG {d['fvg_roi'].median():.3f} (median)")

    df = pd.concat(frames, ignore_index=True)
    action = "analysed (dry run)" if args.dry_run else f"written to {args.out}"
    print(f"\nTotal: {len(df)} windows, {df['n_kept'].sum() if len(df) else 0:,} instances {action}")
    if len(df):
        print("  " + ", ".join(f"{s}: {n}" for s, n in df["split"].value_counts().items()))

    manifest = df if args.dry_run else args.out / "windows.csv"
    if len(df):
        problems = check_split_disjoint(manifest)
        if problems:
            print(f"\nLEAKAGE between splits ({len(problems)}):")
            for p in problems[:20]:
                print("  " + p)
        else:
            print("Split check: train / val share no windows, pixels or specimens")


if __name__ == "__main__":
    main()

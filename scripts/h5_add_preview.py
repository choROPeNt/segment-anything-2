"""
Add downscaled preview datasets (image_preview, labels_preview, label_map_preview, mask_preview) to existing
.h5 files, so large panoramas can be viewed in H5Web (VS Code) — see sam2.addons.add_previews.

    python scripts/h5_add_preview.py results/Promotion_               # every *.h5 below the folder
    python scripts/h5_add_preview.py results/Promotion/160/batch_02/160-02_01.h5 --factor 2
"""

import argparse
import time
from pathlib import Path

from sam2.addons import add_previews


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help=".h5 files or folders (searched recursively)")
    ap.add_argument("--factor", type=int, default=2, help="downscale factor (default: 2)")
    args = ap.parse_args()

    files = sorted({f for p in args.paths for f in ([p] if p.is_file() else p.rglob("*.h5"))})
    for k, f in enumerate(files, 1):
        t0 = time.time()
        try:
            written = add_previews(f, args.factor)
        except OSError as e:
            print(f"[{k}/{len(files)}] {f}: skipped ({e})")
            continue
        print(f"[{k}/{len(files)}] {f}: {', '.join(written) or 'nothing to preview'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

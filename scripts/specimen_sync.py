"""
Compare and sync the panorama folders with their specimen .h5 (layout: sam2.addons.specimen_store).

    python scripts/specimen_sync.py status  data_phd                    # table: entry / file / state
    python scripts/specimen_sync.py status  data_phd --deep             # also compare content of untracked/changed
    python scripts/specimen_sync.py migrate data_phd                    # flat merged h5 → folder layout (moves only)
    python scripts/specimen_sync.py adopt   data_phd                    # sign entries whose content is identical
    python scripts/specimen_sync.py import  data_phd [--create] [--states ...]   # folder → h5
    python scripts/specimen_sync.py export  data_phd [--force]          # h5 → folder (missing Labkit / tiff files)
    python scripts/specimen_sync.py review-merge data_phd [--review bad] # merge labeling/{s}.bad.labeling back, delete it

Restrict to some specimens with --only 160_01 (prefix match). Every write command accepts --dry-run.
Typical loop: edit labeling/{s}.tiff.labeling in Labkit → `status` shows "folder changed" → `import`
(updates /labeling/labels, instances, previews) → `export --force` refreshes the derived labels.tiff.
"""

import argparse
from collections import Counter

from sam2.addons.specimen_store import (adopt, find_specimens, merge_review, migrate, status, sync_export,
                                         sync_import)

SHORT = {"ok": "ok", "-": "·"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "migrate", "adopt", "import", "export", "review-merge"])
    ap.add_argument("root", help="data root, e.g. data_phd")
    ap.add_argument("--only", nargs="*", default=None, help="specimen name prefixes, e.g. 160_01 245")
    ap.add_argument("--deep", action="store_true", help="status: compare content where signatures are missing/changed")
    ap.add_argument("--create", action="store_true", help="import: create missing h5 files")
    ap.add_argument("--pixel-size", type=float, default=None, help="import --create: pixel size in µm")
    ap.add_argument("--states", nargs="*", default=["new in folder", "folder changed"], help="import: states to import")
    ap.add_argument("--entries", nargs="*", default=None, help="import: only these datasets, e.g. panorama mask")
    ap.add_argument("--force", action="store_true", help="export: rewrite all exportable files")
    ap.add_argument("--review", default="bad", help="review-merge: name of the review file {s}.<name>.labeling")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sps = [s for s in find_specimens(args.root) if not args.only or any(s["specimen"].startswith(o) for o in args.only)]
    total = Counter()
    for sp in sps:
        has_h5 = sp["h5"].exists()
        if args.command == "status" or args.dry_run:
            rows = status(sp, deep=args.deep)
            print(f"\n{sp['specimen']}  ({sp['h5'].name if has_h5 else 'no h5'})")
            for r in rows:
                if r["state"] == "-":
                    continue
                total[r["state"]] += 1
                print(f"  {r['entry'].dataset:28s} {r['files']:42s} {r['state']}")
            continue
        if args.command == "review-merge" and not (sp["dir"] / "labeling" / f"{sp['specimen']}.{args.review}.labeling").exists():
            continue
        if not has_h5 and not (args.command == "import" and args.create):
            print(f"{sp['specimen']}: no h5 (import --create to make one)")
            continue
        log = {"migrate": lambda: migrate(sp),
               "adopt": lambda: adopt(sp, args.root),
               "import": lambda: sync_import(sp, args.root, tuple(args.states), args.create, args.pixel_size, entries=args.entries),
               "export": lambda: sync_export(sp, args.root, force=args.force),
               "review-merge": lambda: [f"{k}: {v}" for k, v in merge_review(sp, args.root, args.review).items()]}[args.command]()
        print(f"{sp['specimen']}: " + ("; ".join(log) if log else "nothing to do"))
    if total:
        print("\n" + ", ".join(f"{k}: {v}" for k, v in total.most_common()))


if __name__ == "__main__":
    main()

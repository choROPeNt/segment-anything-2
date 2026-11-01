"""
Convert specimen ``.h5`` results into a SAM 2 training dataset (SA-1B layout).

Output layout (read by ``training.dataset.vos_raw_dataset.SA1BRawDataset``)::

    out_dir/
        images/{name}.jpg          window crops (RGB)
        annotations/{name}.json    {"image": {...}, "annotations": [{segmentation (COCO RLE), area, bbox, ...}]}
        windows.csv                one row per window: position, ROI coverage, instance counts, FVG, split
        {split}.txt                window names per split (``file_list_txt`` in the training config)

``SA1BRawDataset`` parses ``int(name.split("_")[-1])``, so every name ends in a running number
that is unique across the whole output directory.
"""

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_erosion, find_objects

from .patcher import Sam2Patcher


def load_specimen_h5(h5_path):
    """
    Load image, label map and ROI mask from a specimen ``.h5``.

    Supports the current layout (``/label_map`` + ``/mask``) and the older one (``/labels``, no mask).

    Returns
    -------
    image     : (H, W, 3) uint8
    label_map : (H, W)    integer, 0 = background
    roi       : (H, W)    bool, or None if the file has no ``/mask``
    attrs     : dict      root attributes
    """
    with h5py.File(h5_path, "r") as hf:
        image = hf["image"][()]
        label_key = "label_map" if "label_map" in hf else "labels"
        label_map = hf[label_key][()]
        roi = hf["mask"][()].astype(bool) if "mask" in hf else None
        attrs = dict(hf.attrs)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return image, label_map, roi, attrs


def window_positions(full, size, stride):
    """Window starts along one axis; the last window is flush with the border."""
    return Sam2Patcher._positions(full, size, size - stride)


def roi_clipped_ids(label_map, roi):
    """Label ids touching the ROI boundary — their outline is cut by the mask, not by the fibre edge."""
    if roi is None:
        return set()
    edge = roi & ~binary_erosion(roi)
    ids = np.unique(label_map[edge])
    return set(ids[ids > 0].tolist())


def window_annotations(lab_win, keep_ids, start_id=1):
    """
    SA-1B style annotations for the instances *keep_ids* in a window label map.

    Returns a list of dicts with ``segmentation`` as COCO RLE (``counts`` as str).
    """
    annots = []
    slices = find_objects(lab_win)
    for n, lid in enumerate(sorted(keep_ids), start=start_id):
        sl = slices[lid - 1]
        m = np.zeros(lab_win.shape, dtype=np.uint8)
        m[sl][lab_win[sl] == lid] = 1
        rle = mask_utils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("utf-8")
        sy, sx = sl
        annots.append({
            "id"           : n,
            "instance_id"  : int(lid),
            "segmentation" : rle,
            "area"         : int(m.sum()),
            "bbox"         : [int(sx.start), int(sy.start), int(sx.stop - sx.start), int(sy.stop - sy.start)],
            "iscrowd"      : 0,
            "category_id"  : 1,
            "uncertain_iou": 1.0,
        })
    return annots


def _next_index(out_dir):
    """Next free running number in out_dir/images (names end in ``_<int>``)."""
    imgs = list((Path(out_dir) / "images").glob("*.jpg"))
    nums = [int(p.stem.split("_")[-1]) for p in imgs if p.stem.split("_")[-1].isdigit()]
    return max(nums) + 1 if nums else 0


def spatial_split_regions(length, size, val_frac=0.2, n_blocks=10, guard=128, seed=0):
    """
    Split one image axis into contiguous train / val regions separated by a guard band.

    The axis is divided into *n_blocks* equal blocks; ``round(val_frac * n_blocks)`` of them
    (at least one) are drawn as validation with *seed*.  Train regions are shortened by *guard*
    px on every side that borders a val region, so no pixel — and no fibre smaller than *guard* —
    can end up in both splits.  Regions shorter than *size* cannot hold a window and are dropped.

    Returns
    -------
    list of (start, end, split) — half-open pixel intervals along the axis
    """
    edges = np.linspace(0, length, n_blocks + 1).round().astype(int)
    n_val = min(max(1, int(round(val_frac * n_blocks))), n_blocks - 1)
    val_blocks = set(np.random.default_rng(seed).choice(n_blocks, n_val, replace=False).tolist())

    # merge neighbouring blocks with the same split
    regions = []
    for b in range(n_blocks):
        split = "val" if b in val_blocks else "train"
        if regions and regions[-1][2] == split:
            regions[-1][1] = edges[b + 1]
        else:
            regions.append([edges[b], edges[b + 1], split])

    out = []
    for i, (start, end, split) in enumerate(regions):
        if split == "train":
            if i > 0:
                start += guard
            if i < len(regions) - 1:
                end -= guard
        if end - start >= size:
            out.append((int(start), int(end), split))
        else:
            warnings.warn(f"{split} region [{start}, {end}) is shorter than the window size {size} — dropped")
    return out


def check_split_disjoint(manifest, guard=0):
    """
    Check a ``windows.csv`` manifest for leakage between splits.

    * a specimen exported as whole-file split must not appear in another split
    * within a specimen, no train window may overlap (or come closer than *guard* px to) a val window

    Returns a list of problems (empty = clean).
    """
    df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
    problems = []
    for spec, d in df.groupby("specimen"):
        splits = sorted(d["split"].unique())
        if len(splits) < 2:
            continue
        if "split_mode" in d and (d["split_mode"] != "auto").any():
            problems.append(f"{spec}: exported to several splits {splits} without spatial split")
            continue
        for a, b in [(s, t) for i, s in enumerate(splits) for t in splits[i + 1:]]:
            A, B = d[d["split"] == a], d[d["split"] == b]
            for r in A.itertuples():
                close = ((B["x0"] < r.x0 + r.width + guard) & (r.x0 < B["x0"] + B["width"] + guard)
                         & (B["y0"] < r.y0 + r.height + guard) & (r.y0 < B["y0"] + B["height"] + guard))
                if close.any():
                    problems.append(f"{spec}: {a} window {r.name} overlaps {b} window(s) "
                                    f"{', '.join(B.loc[close, 'name'].head(3))}")
    return problems


def export_specimen_to_sa1b(
    h5_path,
    out_dir,
    specimen=None,
    split="train",
    size=1024,
    stride=512,
    min_roi_cov=0.8,
    min_visible_frac=0.3,
    min_area=64,
    blank_outside_roi=False,
    jpeg_quality=95,
    val_frac=0.2,
    n_blocks=10,
    guard=128,
    seed=0,
    dry_run=False,
):
    """
    Cut a specimen into windows and export them as SA-1B image/annotation pairs.

    Per window
    ----------
    * kept if the ROI covers at least *min_roi_cov* of it (whole image is ROI if the file has no mask)
    * an instance is kept if its visible part is >= *min_visible_frac* of its full area and >= *min_area* px;
      instances cut by the ROI boundary are dropped (artificial straight outline)
    * *blank_outside_roi* fills pixels outside the ROI with the median colour outside the fibres

    Splits
    ------
    * ``split="train"`` / ``"val"`` / ... — the whole specimen goes to that split (split by specimen)
    * ``split="auto"`` — spatial split within the specimen along its long axis
      (:func:`spatial_split_regions` with *val_frac*, *n_blocks*, *guard*, *seed*); every region is
      tiled on its own, so train and val windows never share pixels or fibres

    A specimen already present in ``windows.csv`` is refused — exporting it twice (e.g. once per split)
    would leak it into both splits.

    Parameters
    ----------
    h5_path  : specimen ``.h5``
    out_dir  : dataset root, created if needed; existing files are kept and numbering continues
    specimen : name used in the window names / manifest (default: ``<grandparent>-<parent>-<stem>``)
    split    : split label, or ``"auto"`` for a spatial train / val split
    dry_run  : only analyse windows (manifest is returned, nothing is written)

    Returns
    -------
    pandas.DataFrame — one row per window (also appended to ``windows.csv``)
    """
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    if specimen is None:
        specimen = "-".join(h5_path.with_suffix("").parts[-3:])
    specimen = specimen.replace("_", "-")          # the part after the last "_" must be the running number

    manifest = out_dir / "windows.csv"
    if not dry_run and manifest.exists():
        done = pd.read_csv(manifest, usecols=["specimen", "split"])
        if specimen in set(done["specimen"]):
            prev = sorted(done.loc[done["specimen"] == specimen, "split"].unique())
            raise ValueError(f"specimen {specimen!r} is already in {manifest} (split {prev}) — "
                             "exporting it again would put the same data into several splits")

    image, label_map, roi, attrs = load_specimen_h5(h5_path)
    H, W = label_map.shape
    if roi is None:
        warnings.warn(f"{h5_path} has no /mask — the whole image is used as ROI, "
                      "unlabelled fibres of neighbouring rovings end up in the windows")
    roi_full = roi if roi is not None else np.ones((H, W), dtype=bool)

    full_area = np.bincount(label_map.ravel())
    clipped = np.array(sorted(roi_clipped_ids(label_map, roi)), dtype=label_map.dtype)

    if blank_outside_roi and roi is not None:
        fill = np.median(image[~roi & (label_map == 0)], axis=0).astype(np.uint8)
        image = image.copy()
        image[~roi] = fill

    # summed-area table → ROI coverage of any window in O(1)
    sat = np.pad(roi_full.cumsum(0, dtype=np.int64).cumsum(1), ((1, 0), (1, 0)))

    if not dry_run:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    idx = _next_index(out_dir)

    # windows per split region: (y0, x0, split); "auto" splits along the long axis
    if split == "auto":
        long_axis = 1 if W >= H else 0
        regions = spatial_split_regions((H, W)[long_axis], size, val_frac, n_blocks, guard, seed)
    else:
        long_axis, regions = 1, [(0, W, split)]

    windows = []
    for start, end, reg_split in regions:
        along = [start + p for p in window_positions(end - start, size, stride)]
        across = window_positions((W, H)[long_axis], size, stride)
        for a in along:
            for c in across:
                windows.append((c, a, reg_split) if long_axis == 1 else (a, c, reg_split))

    rows = []
    for y0, x0, win_split in windows:
        y1, x1 = min(y0 + size, H), min(x0 + size, W)
        h, w = y1 - y0, x1 - x0
        roi_px = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
        cov = roi_px / (size * size)
        if cov < min_roi_cov:
            continue

        lab = label_map[y0:y1, x0:x1]
        roi_w = roi_full[y0:y1, x0:x1]
        ids, vis = np.unique(lab, return_counts=True)
        vis, ids = vis[ids > 0], ids[ids > 0]

        frac = vis / full_area[ids]
        cut_by_roi = np.isin(ids, clipped)
        ok = (frac >= min_visible_frac) & (vis >= min_area) & ~cut_by_roi
        keep = ids[ok]

        name = f"{specimen}_{idx:06d}"
        fibre_roi = np.count_nonzero((lab > 0) & roi_w)
        rows.append({
            "name"         : name,
            "specimen"     : specimen,
            "split"        : win_split,
            "split_mode"   : "auto" if split == "auto" else "specimen",
            "h5"           : str(h5_path),
            "x0"           : x0,
            "y0"           : y0,
            "width"        : w,
            "height"       : h,
            "roi_cov"      : round(float(cov), 4),
            "n_instances"  : len(ids),
            "n_kept"       : len(keep),
            "n_complete"   : int(np.count_nonzero(ok & (frac >= 0.999))),
            "n_dropped_cut": int(np.count_nonzero(~ok & ~cut_by_roi)),
            "n_dropped_roi": int(np.count_nonzero(cut_by_roi)),
            "fvg_roi"      : round(fibre_roi / max(int(roi_px), 1), 4),
            "median_area"  : float(np.median(full_area[keep])) if len(keep) else 0.0,
        })

        if not dry_run:
            crop = image[y0:y1, x0:x1]
            if (h, w) != (size, size):                  # pad to a square window (image smaller than size)
                pad = ((0, size - h), (0, size - w))
                crop = np.pad(crop, pad + ((0, 0),), mode="edge")
                lab = np.pad(lab, pad)
            Image.fromarray(crop).save(out_dir / "images" / f"{name}.jpg",
                                       quality=jpeg_quality, subsampling=0)
            payload = {
                "image": {"image_id": idx, "file_name": f"{name}.jpg", "width": size, "height": size,
                          "specimen": specimen, "x0": x0, "y0": y0},
                "annotations": window_annotations(lab, set(keep.tolist())),
            }
            with open(out_dir / "annotations" / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f)
        idx += 1

    df = pd.DataFrame(rows)
    if not dry_run and len(df):
        df.to_csv(manifest, mode="a", header=not manifest.exists(), index=False)
        for s, d in df.groupby("split"):
            with open(out_dir / f"{s}.txt", "a", encoding="utf-8") as f:
                f.writelines(f"{n}\n" for n in d["name"])
    return df


def sa1b_to_labelme(dataset_dir, out_dir=None, label="fibre", tolerance=0.5):
    """
    Write LabelMe JSON files (one per window, polygon shapes) for an SA-1B dataset folder.

    Each annotation mask becomes a polygon of its outer contour (``group_id`` = ``instance_id``),
    simplified with Douglas-Peucker *tolerance* in px (0 = keep every contour pixel).
    The files reference ``images/{name}.jpg`` relative to *out_dir*, default ``dataset_dir/labelme``.

    Open in LabelMe: *Open Dir* → ``images/``, then *File → Change Output Dir* → *out_dir*.

    Returns the number of files written.
    """
    import os

    import cv2

    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir) if out_dir is not None else dataset_dir / "labelme"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = 0
    for ann_path in sorted((dataset_dir / "annotations").glob("*.json")):
        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)

        shapes = []
        for a in data["annotations"]:
            m = mask_utils.decode(a["segmentation"])
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            if tolerance > 0:
                c = cv2.approxPolyDP(c, tolerance, True)
            if len(c) < 3:
                continue
            shapes.append({
                "label"      : label,
                "points"     : c[:, 0, :].astype(float).tolist(),
                "group_id"   : a.get("instance_id"),
                "description": "",
                "shape_type" : "polygon",
                "flags"      : {},
                "mask"       : None,
            })

        h, w = data["annotations"][0]["segmentation"]["size"] if data["annotations"] else (
            data["image"]["height"], data["image"]["width"])
        image_path = dataset_dir / "images" / f"{ann_path.stem}.jpg"
        payload = {
            "version"    : "5.5.0",
            "flags"      : {},
            "shapes"     : shapes,
            "imagePath"  : os.path.relpath(image_path, out_dir),
            "imageData"  : None,
            "imageHeight": int(h),
            "imageWidth" : int(w),
        }
        with open(out_dir / f"{ann_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
        n_files += 1

    return n_files

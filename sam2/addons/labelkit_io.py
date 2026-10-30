import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import find_objects

from .labelmap_tools import reindex_labelmap


def load_labkit_json(
    json_path: Path | str,
    label_map: dict[str, int] | None = None,
    dtype: type = np.int32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a LabelKit sparse-label JSON and return a dense 2-D label image.

    Coordinates are stored as ``[x, y]`` in the file; the output array uses
    row-major layout ``(row=y, col=x)``.

    Parameters
    ----------
    json_path : Path | str
        Path to the ``.labeling`` / ``.json`` file exported by LabelKit.
    label_map : dict[str, int], optional
        Mapping from JSON label key to output integer value,
        e.g. ``{"g-fiber": 1, "matrix": 2}``.  Keys not present in the
        dict fall back to ``int(key)``.
    dtype : np.dtype
        Integer dtype for the output label image.  Defaults to ``np.int32``.

    Returns
    -------
    out : np.ndarray (H, W)
        Dense label map.  Each pixel holds the mapped integer value.
    data : dict
        Raw parsed JSON payload.
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    interval = data["interval"]
    xmin, ymin = interval["min"]
    xmax, ymax = interval["max"]

    width  = xmax - xmin + 1
    height = ymax - ymin + 1

    out = np.zeros((height, width), dtype=dtype)

    for key, coords in data["labels"].items():
        print(key)
        if not coords:
            continue
        if label_map is not None:
            if key not in label_map:
                continue
            value = label_map[key]
        else:
            try:
                value = int(key)
            except ValueError:
                continue
        xy = np.asarray(coords, dtype=np.int64)  # (N, 2)
        out[xy[:, 1] - ymin, xy[:, 0] - xmin] = value

    return out, data


def read_labelkit_json(json_file, dtype=np.uint16):
    """
    Read a LabelKit-style sparse label JSON and reconstruct a dense 2D label map.

    Labels are assigned purely by instance order:
        first label  -> 1
        second label -> 2
        ...

    Returns
    -------
    label_map : np.ndarray
        Dense 2D label map of shape (H, W)
    meta : dict
        Parsed metadata
    """
    json_file = Path(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    interval = data["interval"]
    mins = interval["min"]
    maxs = interval["max"]
    ndim = interval["n"]

    if ndim != 2:
        raise ValueError(f"Expected 2D data, got n={ndim}")

    width  = maxs[0] - mins[0] + 1  # inclusive max
    height = maxs[1] - mins[1] + 1

    label_map = np.zeros((height, width), dtype=dtype)
    label_name_to_id = {}

    for instance_id, (label_name, coords) in enumerate(data["labels"].items(), start=1):
        label_name_to_id[label_name] = instance_id

        coords = np.asarray(coords, dtype=np.int64)
        if coords.size == 0:
            continue

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Label {label_name} has invalid coordinate shape: {coords.shape}")

        x = coords[:, 0] - mins[0]  # coordinates are [x, y]
        y = coords[:, 1] - mins[1]

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        label_map[y[valid], x[valid]] = instance_id

    meta = {
        "interval": interval,
        "pixelSizes": data.get("pixelSizes"),
        "num_labels": len(data.get("labels", {})),
        "shape": label_map.shape,
        "label_name_to_id": label_name_to_id,
    }

    return label_map, meta


def export_labels_to_labelkit(label_map, ids, output_path, meta):
    """
    Export a subset of labels as a LabelKit-style sparse JSON.

    The output can be opened in LabelKit on top of the original image for
    manual correction.  Coordinate system matches the original file exactly
    so re-importing is seamless.

    Parameters
    ----------
    label_map   : np.ndarray (H, W)
    ids         : list of int  — label IDs to export
    output_path : str or Path
    meta        : dict returned by read_labelkit_json  (carries interval / pixelSizes)
    """
    interval = meta["interval"]
    ox, oy   = interval["min"]   # origin offset (pixels → absolute coords)

    labels_dict = {}
    for label_id, sl in enumerate(find_objects(label_map), start=1):
        if label_id not in ids or sl is None:
            continue
        sy, sx = sl
        ys, xs = np.where(label_map[sy, sx] == label_id)
        # shift back to absolute coordinates
        coords = np.stack([xs + sx.start + ox,
                           ys + sy.start + oy], axis=1).tolist()
        labels_dict[str(label_id)] = coords

    payload = {
        "interval"  : interval,
        "pixelSizes": meta.get("pixelSizes"),
        "labels"    : labels_dict,
    }

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Exported {len(labels_dict)} labels → {output_path}")


def merge_corrected_labels(label_map, corrected_json, ids_to_replace, meta):
    """
    Replace specific labels in *label_map* with a corrected LabelKit JSON
    and reindex the whole map.

    Workflow
    --------
    1. export_labels_to_labelkit(label_map, bad_ids, 'fix.labeling', meta)
    2. open fix.labeling in LabelKit, correct manually, save
    3. label_map_, id_map = merge_corrected_labels(label_map_, 'fix.labeling',
                                                    bad_ids, meta)

    Parameters
    ----------
    label_map       : np.ndarray (H, W)  — current full label map
    corrected_json  : str or Path        — .labeling file from LabelKit
    ids_to_replace  : list of int        — old IDs to remove before merging
    meta            : dict from read_labelkit_json  (for coordinate origin)

    Returns
    -------
    out    : reindexed label map
    id_map : {new_id -> original_id}
    """
    # 1. zero out the labels being replaced
    base = label_map.copy()
    for i in ids_to_replace:
        base[base == i] = 0

    # 2. read corrected labels — coordinates already in absolute space
    corrected_map, _ = read_labelkit_json(corrected_json,
                                          dtype=label_map.dtype)

    # 3. paste corrected labels above the current max to avoid collisions
    offset  = int(base.max())
    new_ids = np.unique(corrected_map)
    new_ids = new_ids[new_ids > 0]

    # clip corrected map to base shape (labeling may cover same full canvas)
    h, w = base.shape
    corrected_map = corrected_map[:h, :w]

    for nid in new_ids:
        base[corrected_map == nid] = offset + int(nid)

    return reindex_labelmap(base)

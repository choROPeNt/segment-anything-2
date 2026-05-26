import json
from pathlib import Path
from typing import Any

import numpy as np


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

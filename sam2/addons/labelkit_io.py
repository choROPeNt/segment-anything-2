import json
from pathlib import Path
from typing import Any

import numpy as np


def load_labkit_json(
    json_path: Path | str,
    dtype: type = np.int32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a LabelKit sparse-label JSON and return a dense 2-D label image.

    Labels are assigned by enumeration order (first entry → 0, second → 1, …).
    Coordinates are stored as ``[x, y]`` in the file; the output array uses
    row-major layout ``(row=y, col=x)``.

    Parameters
    ----------
    json_path : Path | str
        Path to the ``.labeling`` / ``.json`` file exported by LabelKit.
    dtype : np.dtype
        Integer dtype for the output label image.  Defaults to ``np.int32``.

    Returns
    -------
    label_map : np.ndarray (H, W)
        Dense label map.  Each pixel holds the class id (0-based).
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

    label_map = np.zeros((height, width), dtype=dtype)

    for class_id, (_, coords) in enumerate(data["labels"].items()):
        if not coords:
            continue
        xy = np.asarray(coords, dtype=np.int64)  # (N, 2)
        # shift to local origin, then assign — row=y, col=x
        label_map[xy[:, 1] - ymin, xy[:, 0] - xmin] = class_id + 1

    return label_map, data

import os
from pathlib import Path

import h5py 
import numpy as np
from PIL import Image

from .labelmap_tools import apply_mask_to_labelmap, iter_sam_masks, get_segmentation


def delete_h5_if_exists(h5_path: str) -> None:
    if os.path.exists(h5_path):
        try:
            os.remove(h5_path)
            print(f"Deleted existing HDF5: {h5_path}")
        except PermissionError as e:
            raise RuntimeError(f"Cannot delete {h5_path} (file may be open elsewhere): {e}")
    else:
        print(f"No existing HDF5 at {h5_path} — nothing to delete.")

def _write_h5_item(parent: h5py.Group, key: str, value) -> None:
    """Recursively write a value into an h5py group under the given key.

    - list of dicts → group of zero-padded sub-groups (keyed by "id" if present)
    - dict          → nested group
    - array/scalar  → dataset (gzip-compressed when ndim >= 2)
    """
    if isinstance(value, list) and value and isinstance(value[0], dict):
        grp = parent.create_group(key)
        for i, record in enumerate(value):
            sub_key = str(record.get("id", i + 1)).zfill(4)
            g = grp.create_group(sub_key)
            for k, v in record.items():
                _write_h5_item(g, k, v)
    elif isinstance(value, dict):
        grp = parent.create_group(key)
        for k, v in value.items():
            _write_h5_item(grp, k, v)
    else:
        arr = np.asarray(value)
        if arr.ndim >= 2:
            parent.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        else:
            parent.create_dataset(key, data=arr)


def write_h5(path: str, dict_out: dict, overwrite: bool = True):
    """
    Write any dict to an HDF5 file.

    - numpy arrays / scalars → datasets (gzip-compressed when ndim >= 2)
    - list of dicts → group of indexed sub-groups (keyed by zero-padded "id"
      field if present, otherwise by position)
    - nested dicts → nested groups

    Parameters
    ----------
    path : str
        Output HDF5 file path.
    dict_out : dict
        Data to write.
    overwrite : bool
        If True, overwrites existing file.
    """
    if overwrite and os.path.exists(path):
        os.remove(path)

    with h5py.File(path, "w") as h5f:
        for key, value in dict_out.items():
            _write_h5_item(h5f, key, value)

    keys = sorted(dict_out)
    print(f"✅ Saved: {path}")
    for i, k in enumerate(keys):
        prefix = "  └─" if i == len(keys) - 1 else "  ├─"
        v = dict_out[k]
        if isinstance(v, np.ndarray):
            info = str(v.shape)
        elif isinstance(v, list):
            info = f"{len(v)} items"
        else:
            arr = np.asarray(v)
            info = str(arr.shape) if arr.ndim > 0 else repr(v)
        print(f"{prefix} {k:<20}: {info}")


def _h5_scalar_or_array(v):
    arr = np.asarray(v)
    return arr.item() if arr.ndim == 0 else arr


def _load_h5_item(item):
    """Recursively load an h5py Dataset or Group into Python/numpy types.

    Groups whose children are all sub-groups are returned as a list of dicts
    (attrs take priority over same-named datasets within each sub-group).
    All other groups are returned as nested dicts.
    """
    if isinstance(item, h5py.Dataset):
        arr = np.array(item)
        return arr.item() if arr.ndim == 0 else arr

    children = list(item.keys())
    if children and all(isinstance(item[k], h5py.Group) for k in children):
        records = []
        for k in sorted(children):
            g = item[k]
            rec = {a: _h5_scalar_or_array(v) for a, v in g.attrs.items()}
            for ds in g:
                if ds not in rec:
                    rec[ds] = _load_h5_item(g[ds])
            records.append(rec)
        return records

    result = {a: _h5_scalar_or_array(v) for a, v in item.attrs.items()}
    for k in children:
        if k not in result:
            result[k] = _load_h5_item(item[k])
    return result


def read_h5(path: str) -> dict:
    """
    Generically read all datasets and groups from an HDF5 file.

    - Datasets → numpy arrays (0-d arrays unwrapped to Python scalars).
    - Groups whose children are all sub-groups → list of dicts (attrs take
      priority over same-named datasets within each record).
    - Other groups → nested dicts.
    - "label_map" is aliased to "labels" for backward compatibility.

    Returns
    -------
    dict
        Keys mirror the HDF5 root structure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    _ALIASES = {"label_map": "labels"}

    out = {}
    with h5py.File(path, "r") as h5f:
        for key in h5f:
            key = str(key)
            out[_ALIASES.get(key, key)] = _load_h5_item(h5f[key])

    keys = sorted(out)
    print(f"📂 Loaded: {path}")
    for i, k in enumerate(keys):
        prefix = "  └─" if i == len(keys) - 1 else "  ├─"
        v = out[k]
        if isinstance(v, np.ndarray):
            info = str(v.shape)
        elif isinstance(v, list):
            info = f"{len(v)} items"
        else:
            info = repr(v)
        print(f"{prefix} {k:<20}: {info}")

    return out

def specimen_to_h5(specimen, files, out_dir, pixel_size=None, pixel_unit="µm"):
    """
    Load image + fixed label map + mask for one specimen,
    apply mask, extract SAM2 instance crops, save to HDF5.

    Parameters
    ----------
    specimen   : str   — specimen name, output is ``out_dir / f"{specimen}.h5"``
    files      : dict  — paths with keys ``"image"``, ``"labels"``, ``"mask"``
    out_dir    : Path
    pixel_size : float, optional — isotropic pixel size; omitted from attrs if None
    pixel_unit : str

    HDF5 layout
    -----------
    /   attrs: specimen, pixel_size_x, pixel_size_y, pixel_size_unit,
               image_shape, n_instances
    /image                      (H, W, 3) uint8
    /label_map                  (H, W)    uint16  — masked & reindexed
    /mask                       (H, W)    uint8   — binary ROI mask (0/1)
    /instances/<id>/
        segmentation_crop       (h, w)    bool
        bbox                    (4,)      int32   [x, y, w, h]
        attrs: area, bbox_x, bbox_y, bbox_w, bbox_h
    """
    image     = np.asarray(Image.open(files["image"]))
    label_raw = np.asarray(Image.open(files["labels"]))
    mask      = np.asarray(Image.open(files["mask"])).astype(bool)

    label_map_, _ = apply_mask_to_labelmap(label_raw, mask)

    px_x = px_y = pixel_size
    px_unit = pixel_unit

    out_path = Path(out_dir) / f"{specimen}.h5"
    with h5py.File(out_path, "w") as hf:

        # --- root attributes ---
        hf.attrs["specimen"]        = specimen
        hf.attrs["image_shape"]     = image.shape
        if px_x is not None:
            hf.attrs["pixel_size_x"]    = px_x
            hf.attrs["pixel_size_y"]    = px_y
            hf.attrs["pixel_size_unit"] = px_unit

        # --- datasets ---
        ds_img  = hf.create_dataset("image",     data=image,                 compression="gzip")
        ds_lm   = hf.create_dataset("label_map", data=label_map_,            compression="gzip")
        ds_mask = hf.create_dataset("mask",      data=mask.astype(np.uint8), compression="gzip")

        for ds in (ds_img, ds_lm, ds_mask):
            ds.attrs["axes"] = "YXC" if ds.ndim == 3 else "YX"
            if px_x is not None:
                ds.attrs["pixel_size_x"]    = px_x
                ds.attrs["pixel_size_y"]    = px_y
                ds.attrs["pixel_size_unit"] = px_unit

        grp = hf.create_group("instances")
        instance_list = list(iter_sam_masks(label_map_))
        hf.attrs["n_instances"] = len(instance_list)

        for m in instance_list:
            ig = grp.create_group(f"{m['id']:05d}")
            ig.create_dataset("segmentation_crop",
                              data=get_segmentation(m, label_map_),
                              compression="gzip")
            x, y, w, h = m["bbox"]
            ig.create_dataset("bbox", data=np.array(m["bbox"], dtype=np.int32))
            ig.attrs["area"]   = m["area"]
            ig.attrs["bbox_x"] = x
            ig.attrs["bbox_y"] = y
            ig.attrs["bbox_w"] = w
            ig.attrs["bbox_h"] = h
            if px_x is not None:
                ig.attrs["area_um2"] = m["area"] * px_x * px_y

    return out_path

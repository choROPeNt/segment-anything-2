import os
import h5py 
import numpy as np


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
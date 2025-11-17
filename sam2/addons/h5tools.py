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

def write_h5(path: str, dict_out: dict, overwrite: bool = True):
    """
    Write image/label/mask/instance data to an HDF5 file.

    Expected structure:
    {
        "image": np.ndarray [H, W] or [H, W, C],
        "labels": np.ndarray [H, W],
        "mask": np.ndarray [H, W],
        "binary": np.ndarray [H, W],
        "instances": [
            {
                "id": int,
                "area": float,
                "bbox": [x0, y0, w, h],
                "segmentation_cropped": np.ndarray (bool)
            },
            ...
        ]
    }

    Parameters
    ----------
    path : str
        Output HDF5 file path.
    dict_out : dict
        Data dictionary as above.
    overwrite : bool
        If True, overwrites existing file.
    """
    if overwrite and os.path.exists(path):
        os.remove(path)

    with h5py.File(path, "w") as h5f:
        # --- scalar / array datasets ---
        for key in ["image", "labels", "mask", "binary"]:
            if key not in dict_out:
                continue
            data = np.asarray(dict_out[key])
            h5f.create_dataset(
                key,
                data=data,
                compression="gzip",
                compression_opts=4
            )

        # --- instances ---
        if "instances" in dict_out and dict_out["instances"]:
            grp_inst = h5f.create_group("instances")
            for inst in dict_out["instances"]:
                inst_id = str(inst.get("id", len(grp_inst) + 1)).zfill(4)
                g = grp_inst.create_group(inst_id)

                # Store metadata
                g.create_dataset("id", data=np.int32(inst.get("id", -1)))
                g.create_dataset("area", data=np.float32(inst.get("area", 0.0)))

                bbox = np.asarray(inst.get("bbox", [0, 0, 0, 0]), dtype=np.int32)
                g.create_dataset("bbox", data=bbox)

                seg = np.asarray(inst.get("segmentation_crop", []), dtype=bool)
                g.create_dataset(
                    "segmentation_crop",
                    data=seg,
                    compression="gzip",
                    compression_opts=4
                )

        # Summary
        n_inst = len(dict_out.get("instances", []))
        print(f"✅ Saved: {path}")
        print(f"  ├─ image shape   : {dict_out['image'].shape if 'image' in dict_out else None}")
        print(f"  ├─ labels shape  : {dict_out['labels'].shape if 'labels' in dict_out else None}")
        print(f"  ├─ binary shape  : {dict_out['binary'].shape if 'binary' in dict_out else None}")
        print(f"  ├─ mask shape    : {dict_out['mask'].shape if 'mask' in dict_out else None}")
        print(f"  └─ instances     : {n_inst}")



def read_h5(path: str) -> dict:
    """
    Read image/label/mask/instance data from an HDF5 file 
    created with `write_h5`.

    Returns
    -------
    dict
        {
            "image": np.ndarray [H, W] or [H, W, C],
            "labels": np.ndarray [H, W],
            "mask": np.ndarray [H, W],
            "binary": np.ndarray [H, W],
            "instances": [
                {
                    "id": int,
                    "area": float,
                    "bbox": [x0, y0, w, h],
                    "segmentation_crop": np.ndarray (bool)
                },
                ...
            ]
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    out = {}
    with h5py.File(path, "r") as h5f:
        # --- load standard datasets ---
        for key in ["image", "labels", "mask", "binary"]:
            if key in h5f:
                out[key] = np.array(h5f[key])

        # --- load instances ---
        if "instances" in h5f:
            instances = []
            grp_inst = h5f["instances"]
            for inst_id in grp_inst:
                g = grp_inst[inst_id]
                inst = {
                    "id": int(np.array(g["id"])),
                    "area": float(np.array(g["area"])),
                    "bbox": np.array(g["bbox"], dtype=int).tolist(),
                    "segmentation_crop": np.array(g["segmentation_crop"], dtype=bool)
                }
                instances.append(inst)
            out["instances"] = instances

    # --- summary ---
    n_inst = len(out.get("instances", []))
    print(f"📂 Loaded: {path}")
    print(f"  ├─ image shape   : {out['image'].shape if 'image' in out else None}")
    print(f"  ├─ labels shape  : {out['labels'].shape if 'labels' in out else None}")
    print(f"  ├─ binary shape  : {out['binary'].shape if 'binary' in out else None}")
    print(f"  ├─ mask shape    : {out['mask'].shape if 'mask' in out else None}")
    print(f"  └─ instances     : {n_inst}")

    return out
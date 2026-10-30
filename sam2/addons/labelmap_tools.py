import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label as sk_label


def iter_sam_masks(label_map):
    """
    Yield lightweight SAM-style mask dicts — no bool array stored.

    Each dict carries only the bounding-box slice and scalar metadata so the
    full list can be materialised cheaply.  Call ``get_segmentation`` to
    reconstruct the boolean crop on demand.

    Parameters
    ----------
    label_map : np.ndarray (H, W), integer
        Label map — 0 is background.

    Yields
    ------
    dict with keys:
        ``_sl``   : (slice_y, slice_x) into label_map  — private, for get_segmentation
        ``_id``   : int label id                        — private, for get_segmentation
        ``area``  : int
        ``bbox``  : [x0, y0, w, h]  — COCO convention (xmin, ymin, width, height)
        ``id``    : int
    """
    for label_id, sl in enumerate(find_objects(label_map), start=1):
        if sl is None:
            continue

        sy, sx = sl
        area = int(np.sum(label_map[sy, sx] == label_id))
        if area == 0:
            continue

        yield {
            "_sl" : (sy, sx),
            "_id" : label_id,
            "area": area,
            "bbox": [sx.start, sy.start, sx.stop - sx.start, sy.stop - sy.start],  # [x, y, w, h]
            "id"  : label_id,
        }


def get_segmentation(mask, label_map, packed=False):
    """
    Materialise the boolean crop for one mask dict.

    Parameters
    ----------
    mask      : dict yielded by iter_sam_masks
    label_map : the same array passed to iter_sam_masks
    packed    : if True, return (np.packbits(seg), seg.shape) — 8x smaller for
                storage; reconstruct with np.unpackbits(p)[:h*w].reshape(h,w).view(bool)

    Returns
    -------
    bool ndarray (h_box, w_box)  — or (packed_bytes, shape) when packed=True
    """
    sy, sx = mask["_sl"]
    seg = label_map[sy, sx] == mask["_id"]

    if packed:
        return np.packbits(seg), seg.shape

    return seg


def apply_mask_to_labelmap(label_map, binary_mask):
    """
    Zero out labels that fall outside *binary_mask* and reindex survivors from 1.

    Uses ``find_objects`` so only small bbox-sized crops are ever allocated —
    no full (H, W) boolean arrays.

    A label is kept if **any** of its pixels overlap with the mask.
    Pixels of a surviving label that lie outside the mask are also zeroed.

    Parameters
    ----------
    label_map   : np.ndarray (H, W), integer
        Input label map — 0 is background.
    binary_mask : np.ndarray (H, W), bool or uint8
        Pixels that are True/1 define the region of interest.

    Returns
    -------
    out    : np.ndarray (H, W), same dtype as label_map
    id_map : dict  {new_id -> original_id}
    """
    binary_mask = binary_mask.astype(bool)
    out = np.zeros_like(label_map)

    new_id = 1
    id_map = {}

    for old_id, sl in enumerate(find_objects(label_map), start=1):
        if sl is None:
            continue

        sy, sx = sl
        crop      = label_map[sy, sx] == old_id  # small bbox-sized bool crop
        mask_crop = binary_mask[sy, sx]           # matching crop of binary mask

        overlap = crop & mask_crop                # pixels to keep
        if not np.any(overlap):
            continue

        view = out[sy, sx]   # view into output — writes back in-place
        view[overlap] = new_id
        id_map[new_id] = old_id
        new_id += 1

    return out, id_map


def reindex_labelmap(label_map):
    """Renumber all non-zero labels to 1..N preserving spatial layout."""
    out    = np.zeros_like(label_map)
    id_map = {}
    new_id = 1
    for old_id, sl in enumerate(find_objects(label_map), start=1):
        if sl is None:
            continue
        mask = label_map[sl] == old_id
        if not np.any(mask):
            continue
        view = out[sl]
        view[mask] = new_id
        id_map[new_id] = old_id
        new_id += 1
    return out, id_map


def remove_labels(label_map, ids):
    """Zero out the given label IDs and reindex."""
    ids = set([ids] if np.isscalar(ids) else ids)
    out = label_map.copy()
    for i in ids:
        out[out == i] = 0
    return reindex_labelmap(out)


def split_label(label_map, label_id):
    """Split one label into its connected components and reindex the whole map."""
    sl = find_objects(label_map)[label_id - 1]
    if sl is None:
        raise ValueError(f"Label {label_id} not found in label_map")

    out = label_map.copy()
    out[out == label_id] = 0

    binary_crop = (label_map[sl] == label_id)
    components  = sk_label(binary_crop, connectivity=2)
    n_parts     = components.max()

    next_id = int(out.max()) + 1
    view = out[sl]
    for comp_id in range(1, n_parts + 1):
        view[components == comp_id] = next_id + comp_id - 1

    return reindex_labelmap(out) + (n_parts,)


import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence



@dataclass
class Sam2Patcher:
    """
    Tiling & stitching helper for SAM2-style instance dicts.

    Parameters
    ----------
    patch_h, patch_w : int
        Patch (tile) height/width in pixels.
    overlap_h, overlap_w : int
        Overlap (stride = patch - overlap). Must satisfy 0 <= overlap < patch.
    pad_mode : {"reflect", "constant"}
        How to pad on the bottom/right if the image is not a multiple of the patch size.
    pad_value : int or float
        Value used for 'constant' padding.
    merge_iou : float
        IoU threshold to merge touching/overlapping instances across tiles.
    """
    patch_h: int
    patch_w: int
    overlap_h: int = 0
    overlap_w: int = 0
    pad_mode: str = "reflect"
    pad_value: float = 0.0
    merge_iou: float = 0.5

    # ----------------------------- public API ---------------------------------

    def tile_numpy(self, img: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int, int]]:
        """
        Split an (H, W, 3) image into overlapping patches. Patches at the right/bottom
        edge automatically fall back to the minimum available size (no padding).

        Returns
        -------
        patches : list[np.ndarray]
            List of patches; each has shape (h_i, w_i, 3), where h_i<=patch_h and w_i<=patch_w.
        offsets : list[tuple[int,int]]
            Top-left (y0, x0) offsets of each patch in ORIGINAL image coordinates.
        canvas_shape : tuple[int,int,int]
            The original image shape (H, W, 3); useful for stitching later.
        """
        assert img.ndim == 3 and img.shape[-1] == 3, "Expected (H, W, 3) image"
        H, W, _ = img.shape

        ys = self._positions(H, self.patch_h, self.overlap_h)
        xs = self._positions(W, self.patch_w, self.overlap_w)

        patches: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []

        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + self.patch_h, H)
                x1 = min(x0 + self.patch_w, W)
                # copy() so downstream transforms don’t mutate the original backing memory
                patch = img[y0:y1, x0:x1, :].copy()
                patches.append(patch)
                offsets.append((y0, x0))

        return patches, offsets, img.shape



    def stitch_sam2_instances(
        self,
        padded_shape_hw: Tuple[int, int],
        offsets: Sequence[Tuple[int, int]],
        per_tile_sam2: Sequence[List[Dict[str, Any]]],
        debug: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Fast merge of per-tile SAM2 masks into a global instance map.
        - No confidence map (keep-first policy)
        - Compact uniques with return_counts (no giant bincount)
        - searchsorted alignment (no Python dict)
        """
        H, W = padded_shape_hw
        inst_map = np.zeros((H, W), dtype=np.uint32)  # lean dtype
        next_id = np.uint32(1)
        total_written = 0
        merge_iou = float(self.merge_iou)

        for (tile_y0, tile_x0), detections in zip(offsets, per_tile_sam2):
            if not detections:
                continue

            for d in detections:
                m = np.asarray(d["segmentation"], dtype=bool, order="C")
                if not m.any():
                    continue

                # bbox in patch coords [x, y, w, h]
                bx, by, bw, bh = d.get("bbox", [0, 0, m.shape[1], m.shape[0]])
                if bw <= 0 or bh <= 0:
                    # cheap bbox from mask
                    ys = np.flatnonzero(m.any(axis=1))
                    if ys.size == 0:
                        continue
                    xs = np.flatnonzero(m.any(axis=0))
                    by, bx = int(ys[0]), int(xs[0])
                    bh = int(ys[-1] - ys[0] + 1)
                    bw = int(xs[-1] - xs[0] + 1)

                by0 = int(by); bx0 = int(bx)
                by1 = by0 + int(bh); bx1 = bx0 + int(bw)

                # global coords (exclusive high)
                gy0 = tile_y0 + by0; gx0 = tile_x0 + bx0
                gy1 = tile_y0 + by1; gx1 = tile_x0 + bx1

                # clamp to canvas
                if gy0 < 0: gy0 = 0
                if gx0 < 0: gx0 = 0
                if gy1 > H: gy1 = H
                if gx1 > W: gx1 = W
                if gy0 >= gy1 or gx0 >= gx1:
                    continue

                # crop tile mask to bbox (tile-local)
                m_sub = m[by0:by1, bx0:bx1]
                if not m_sub.any():
                    continue

                # quick skip: tiny masks → just assign new id (saves IoU work)
                area_m_int = int(m_sub.sum())
                if area_m_int < 8:
                    tgt = next_id
                    next_id += 1
                    im_sub = inst_map[gy0:gy1, gx0:gx1]
                    write = m_sub & (im_sub == 0)
                    if write.any():
                        im_sub[write] = tgt
                        total_written += int(write.sum())
                    continue

                im_sub = inst_map[gy0:gy1, gx0:gx1]

                # ---- Compact overlaps (no huge minlength) ----
                # intersection labels (only where mask is True)
                labels_inter = im_sub[m_sub].ravel()
                if labels_inter.size == 0:
                    overlapping_oids = np.array([], dtype=np.uint32)
                    inter_counts = np.array([], dtype=np.int32)
                else:
                    u_inter, inter_counts = np.unique(labels_inter, return_counts=True)
                    # drop background 0
                    nz = (u_inter != 0)
                    overlapping_oids = u_inter[nz].astype(np.uint32, copy=False)
                    inter_counts = inter_counts[nz].astype(np.int32, copy=False)

                best_iou = 0.0
                tgt = None

                if overlapping_oids.size:
                    # region label counts anywhere in bbox
                    u_reg, reg_counts = np.unique(im_sub.ravel(), return_counts=True)
                    # u_reg is sorted; map overlapping_oids -> indices via searchsorted
                    idx = np.searchsorted(u_reg, overlapping_oids)
                    # Safety: overlapping_oids are subset of u_reg
                    reg_area = reg_counts[idx].astype(np.float32, copy=False)

                    inter = inter_counts.astype(np.float32, copy=False)
                    area_m = float(area_m_int)
                    denom = (area_m + reg_area - inter)
                    denom[denom == 0.0] = 1.0
                    ious = inter / denom

                    j = int(ious.argmax())
                    best_iou = float(ious[j])
                    tgt = int(overlapping_oids[j])

                # assign new id if no good match
                if tgt is None or best_iou < merge_iou:
                    tgt = int(next_id)
                    next_id += 1

                # keep-first write rule (no conf map)
                write = m_sub & (im_sub == 0)
                if write.any():
                    im_sub[write] = np.uint32(tgt)
                    total_written += int(write.sum())

        if debug:
            print(f"[stitch] written_px={total_written}, instances={int(inst_map.max())}")

        # Build instances list efficiently (optional)
        instances: List[Dict[str, Any]] = []
        ids = np.unique(inst_map)
        ids = ids[ids != 0]
        for oid in ids.tolist():
            ys, xs = np.where(inst_map == oid)
            if ys.size == 0:
                continue
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())

            seg_crop = (inst_map[ymin:ymax + 1, xmin:xmax + 1] == oid)

            instances.append({
                "id": int(oid),
                "area": int(ys.size),
                "segmentation_crop": seg_crop.astype(bool),
                "bbox": [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1],  # xywh exclusive
            })

        return inst_map, instances

    # --------------------------- private helpers ------------------------------

    @staticmethod
    def _positions(full: int, win: int, overlap: int) -> List[int]:
        if not (0 <= overlap < win):
            raise ValueError("0 <= overlap < win required")
        if full <= win:
            return [0]
        stride = win - overlap
        pos = [0]
        x = 0
        while True:
            x += stride
            if x + win >= full:
                last = max(0, full - win)
                if last != pos[-1]:
                    pos.append(last)
                break
            pos.append(x)
        return pos

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union)

    @staticmethod
    def _bbox2d(mask_bool: np.ndarray) -> Tuple[List[int], List[int]]:
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            return [0, 0, 0, 0], [0, 0, 0, 0]
        y0, x0 = int(ys.min()), int(xs.min())
        y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
        w, h = x1 - x0, y1 - y0
        return [x0, y0, w, h], [x0, y0, x1, y1]

    def _instances_from_maps(self, inst_map: np.ndarray, conf_map: np.ndarray) -> List[Dict[str, Any]]:
        """Build SAM2-style dicts for each global instance id (>0)."""
        instances: List[Dict[str, Any]] = []
        ids = np.unique(inst_map)
        ids = ids[ids > 0]
        for iid in ids:
            seg = (inst_map == iid)
            area = int(seg.sum())
            if area == 0:
                continue
            bbox, crop_box = self._bbox2d(seg)
            scores = conf_map[seg]
            predicted_iou = float(scores.mean()) if scores.size else 0.0
            instances.append({
                "segmentation": seg,          # (H,W) bool, global coords
                "area": area,
                "bbox": bbox,                 # [x, y, w, h], global coords
                "predicted_iou": predicted_iou,
                "point_coords": [],           # optional: fill if prompts used
                "stability_score": None,      # optional: fill if computed
                "crop_box": crop_box,         # [x0, y0, x1, y1], global coords
            })
        return instances
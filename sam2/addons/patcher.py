
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Any, Sequence



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
    def tile_numpy(
        self,
        img: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, ...]]:
        """
        Split an image into overlapping patches. Supports:

        - grayscale: (H, W)
        - RGB:       (H, W, 3)

        Patches at the right/bottom edge automatically fall back to the minimum
        available size (no padding).

        Returns
        -------
        patches : list[np.ndarray]
            List of patches; each has shape:
            - (h_i, w_i) for grayscale
            - (h_i, w_i, 3) for RGB
        offsets : list[tuple[int, int]]
            Top-left (y0, x0) offsets of each patch in original image coordinates.
        canvas_shape : tuple[int, ...]
            Original image shape; useful for stitching later.
        """
        if img.ndim == 2:
            H, W = img.shape
            is_rgb = False
        elif img.ndim == 3 and img.shape[-1] == 3:
            H, W, _ = img.shape
            is_rgb = True
        else:
            raise ValueError("Expected image of shape (H, W) or (H, W, 3)")

        ys = self._positions(H, self.patch_h, self.overlap_h)
        xs = self._positions(W, self.patch_w, self.overlap_w)

        patches: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []

        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + self.patch_h, H)
                x1 = min(x0 + self.patch_w, W)

                if is_rgb:
                    patch = img[y0:y1, x0:x1, :].copy()
                else:
                    patch = img[y0:y1, x0:x1].copy()

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
        Merge per-tile SAM2 masks into a global instance map.

        Speedups over the naive approach:
        - ``inst_area`` dict: O(1) lookup for existing region sizes, avoids
          ``np.unique(im_sub.ravel())`` per mask.
        - ``inst_bbox`` dict: tracks tight bounding boxes incrementally so the
          final instance-list build only scans each instance's own bbox instead
          of the full canvas (avoids O(N * H * W) ``np.where`` loop).
        - Neighbour-only IoU: a patch grid maps (row, col) → written instance
          IDs; each tile only considers IDs placed by its 4 direct neighbours
          (top, left, top-left, top-right), avoiding spurious merges with
          distant instances that happen to overlap the bbox.
        """
        H, W = padded_shape_hw
        inst_map = np.zeros((H, W), dtype=np.uint32)
        next_id = 1
        merge_iou = float(self.merge_iou)

        # incremental instance metadata: id → [gy0, gx0, gy1, gx1, area_written]
        inst_area: Dict[int, int] = {}          # id → pixel count in inst_map
        inst_bbox: Dict[int, List[int]] = {}    # id → [gy0, gx0, gy1, gx1]

        # patch grid for neighbour lookup
        ys_sorted = sorted(set(o[0] for o in offsets))
        xs_sorted = sorted(set(o[1] for o in offsets))
        ys_idx = {y: i for i, y in enumerate(ys_sorted)}
        xs_idx = {x: i for i, x in enumerate(xs_sorted)}
        # (row, col) → set of instance IDs placed by that tile
        tile_ids: Dict[Tuple[int, int], Set[int]] = {}

        for (tile_y0, tile_x0), detections in zip(offsets, per_tile_sam2):
            if not detections:
                continue

            ri, ci = ys_idx[tile_y0], xs_idx[tile_x0]
            tile_key = (ri, ci)
            tile_ids[tile_key] = set()

            # IDs written by the 4 spatially adjacent tiles that were already processed
            neighbour_ids: Set[int] = set()
            for dr, dc in ((-1, 0), (0, -1), (-1, -1), (-1, 1)):
                nk = (ri + dr, ci + dc)
                if nk in tile_ids:
                    neighbour_ids.update(tile_ids[nk])

            for d in detections:
                m = np.asarray(d["segmentation"], dtype=bool, order="C")
                if not m.any():
                    continue

                bx, by, bw, bh = d.get("bbox", [0, 0, m.shape[1], m.shape[0]])
                if bw <= 0 or bh <= 0:
                    ys_m = np.flatnonzero(m.any(axis=1))
                    if ys_m.size == 0:
                        continue
                    xs_m = np.flatnonzero(m.any(axis=0))
                    by, bx = int(ys_m[0]), int(xs_m[0])
                    bh = int(ys_m[-1] - ys_m[0] + 1)
                    bw = int(xs_m[-1] - xs_m[0] + 1)

                by0, bx0 = int(by), int(bx)
                by1, bx1 = by0 + int(bh), bx0 + int(bw)

                gy0 = max(tile_y0 + by0, 0); gx0 = max(tile_x0 + bx0, 0)
                gy1 = min(tile_y0 + by1, H); gx1 = min(tile_x0 + bx1, W)

                if gy0 >= gy1 or gx0 >= gx1:
                    continue

                m_sub = m[by0:by1, bx0:bx1]
                if not m_sub.any():
                    continue

                area_m = int(m_sub.sum())
                im_sub = inst_map[gy0:gy1, gx0:gx1]

                # tiny masks: skip IoU, just assign new id
                if area_m < 8:
                    tgt = next_id; next_id += 1
                    write = m_sub & (im_sub == 0)
                    if write.any():
                        n = int(write.sum())
                        im_sub[write] = np.uint32(tgt)
                        inst_area[tgt] = n
                        inst_bbox[tgt] = [gy0, gx0, gy1, gx1]
                        tile_ids[tile_key].add(tgt)
                    continue

                # ---- IoU against neighbour instances only ----
                labels_inter = im_sub[m_sub].ravel()
                best_iou = 0.0
                tgt = None

                if labels_inter.size:
                    u_inter, inter_counts = np.unique(labels_inter, return_counts=True)
                    nz = u_inter != 0
                    if nz.any():
                        oids = u_inter[nz]
                        cnts = inter_counts[nz].astype(np.float32)
                        # restrict to neighbours (or same tile if merging within tile)
                        valid = np.fromiter(
                            (int(o) in neighbour_ids or int(o) in tile_ids[tile_key]
                             for o in oids),
                            dtype=bool, count=len(oids)
                        )
                        if valid.any():
                            oids = oids[valid]; cnts = cnts[valid]
                            reg_areas = np.fromiter(
                                (inst_area.get(int(o), 1) for o in oids),
                                dtype=np.float32, count=len(oids)
                            )
                            denom = area_m + reg_areas - cnts
                            denom[denom == 0.0] = 1.0
                            ious = cnts / denom
                            j = int(ious.argmax())
                            best_iou = float(ious[j])
                            tgt = int(oids[j])

                if tgt is None or best_iou < merge_iou:
                    tgt = next_id; next_id += 1

                write = m_sub & (im_sub == 0)
                if write.any():
                    n = int(write.sum())
                    im_sub[write] = np.uint32(tgt)
                    if tgt in inst_bbox:
                        bb = inst_bbox[tgt]
                        bb[0] = min(bb[0], gy0); bb[1] = min(bb[1], gx0)
                        bb[2] = max(bb[2], gy1); bb[3] = max(bb[3], gx1)
                        inst_area[tgt] += n
                    else:
                        inst_bbox[tgt] = [gy0, gx0, gy1, gx1]
                        inst_area[tgt] = n
                    tile_ids[tile_key].add(tgt)

        if debug:
            print(f"[stitch] instances={len(inst_bbox)}, unique_ids={int(inst_map.max())}")

        # ---- Build instance list using tracked bboxes (avoids O(N*H*W) scan) ----
        instances: List[Dict[str, Any]] = []
        for oid, (gy0, gx0, gy1, gx1) in inst_bbox.items():
            crop = inst_map[gy0:gy1, gx0:gx1] == oid
            ys_c, xs_c = np.where(crop)
            if ys_c.size == 0:
                continue
            # tight bbox within the tracked window
            rgy0 = gy0 + int(ys_c.min()); rgx0 = gx0 + int(xs_c.min())
            rgy1 = gy0 + int(ys_c.max()) + 1; rgx1 = gx0 + int(xs_c.max()) + 1
            seg_crop = (inst_map[rgy0:rgy1, rgx0:rgx1] == oid)
            instances.append({
                "id": int(oid),
                "area": int(ys_c.size),
                "segmentation_crop": seg_crop,
                "bbox": [rgx0, rgy0, rgx1 - rgx0, rgy1 - rgy0],
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
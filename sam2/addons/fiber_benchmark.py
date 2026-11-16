"""
Fibre-detection benchmark: SAM 2 detections as circles / ellipses and the metrics of the
composites fibre-detection benchmark

    Yuksel, Broggi, ..., Caglar (2026). Benchmarking and sensitivity analysis of segmentation
    methods for image-based fiber detection in composites. Compos. Part A 211, 110150.
    doi:10.1016/j.compositesa.2026.110150 -- dataset doi:10.5281/zenodo.17186135

Coordinates follow the Fiji / ImageJ convention used by the expert annotators: pixel ``(i, j)``
covers ``[j, j+1) x [i, i+1)``, so a pixel centre is at ``+0.5``.  Radii are in pixels unless a
column name ends in ``_um``.

Detection works on patches enlarged by ``scale`` (fibres at the size the model was trained at);
centroid and moment ellipse of every mask are measured in the enlarged patch and divided by
``scale``, so the geometry keeps sub-pixel precision instead of being re-rasterised at the (coarse)
original pixel size.  Masks touching an interior patch border are dropped (the fibre is whole in
a neighbouring patch); a fibre seen by several overlapping patches is kept once, from the patch that
gives it the most context (:func:`suppress_duplicates`).  A hard core-ownership rule is avoided on
purpose: the centre estimates of two patches differ by ~0.1 px, so a fibre on the core boundary
could fall outside both cores.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree


# ----------------------------------------------------------------------------------------------
# geometry of one mask
# ----------------------------------------------------------------------------------------------
def mask_ellipse(seg: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Centroid and moment ellipse of a boolean mask (pixel-corner coordinates, i.e. pixel centres at
    +0.5).  ``a`` / ``b`` are the semi-axes of the ellipse with the mask's second moments (a disk of
    radius R gives a = b = R), ``angle`` the major-axis angle in degrees (image x axis, y down).
    ``r_eq`` = sqrt(area / pi), ``r_ell`` = (a + b) / 2 (the annotators' r = (Major + Minor) / 4).
    """
    m = cv2.moments(np.asarray(seg, dtype=np.uint8), binaryImage=True)
    area = m["m00"]
    if area <= 0:
        return None
    cx, cy = m["m10"] / area + 0.5, m["m01"] / area + 0.5
    # + 1/12: second moment of a unit pixel about its own centre (continuous pixel area)
    mu20, mu02, mu11 = m["mu20"] / area + 1 / 12, m["mu02"] / area + 1 / 12, m["mu11"] / area
    half_diff = np.hypot((mu20 - mu02) / 2, mu11)
    l1, l2 = (mu20 + mu02) / 2 + half_diff, max((mu20 + mu02) / 2 - half_diff, 0.0)
    a, b = 2 * np.sqrt(l1), 2 * np.sqrt(l2)
    angle = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    return {"x": cx, "y": cy, "area": float(area), "a": a, "b": b, "angle": angle,
            "r_eq": float(np.sqrt(area / np.pi)), "r_ell": (a + b) / 2}


def _border_flags(seg: np.ndarray) -> Tuple[bool, bool, bool, bool]:
    """Mask touches (top, bottom, left, right) border of its patch."""
    return bool(seg[0].any()), bool(seg[-1].any()), bool(seg[:, 0].any()), bool(seg[:, -1].any())


# ----------------------------------------------------------------------------------------------
# tiled detection
# ----------------------------------------------------------------------------------------------
def _positions(full: int, win: int, overlap: int) -> List[int]:
    if full <= win:
        return [0]
    stride = win - overlap
    pos = list(range(0, full - win + 1, stride))
    if pos[-1] + win < full:
        pos.append(full - win)
    return pos


def patch_grid(shape_hw: Tuple[int, int], patch: int, overlap: int,
               roi: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
    """
    Top-left / bottom-right ``(y0, x0, y1, x1)`` of overlapping patches covering *roi*
    (``(y0, x0, y1, x1)``, default: the whole image).
    """
    H, W = shape_hw
    ry0, rx0, ry1, rx1 = roi if roi is not None else (0, 0, H, W)
    ys = [ry0 + p for p in _positions(ry1 - ry0, patch, overlap)]
    xs = [rx0 + p for p in _positions(rx1 - rx0, patch, overlap)]
    return [(y, x, min(y + patch, ry1), min(x + patch, rx1)) for y in ys for x in xs]


def detect_fibres(
    image: np.ndarray,
    generate: Callable[[np.ndarray], List[dict]],
    scale: float,
    patch: int,
    overlap: int,
    roi: Optional[Tuple[int, int, int, int]] = None,
    progress: Optional[Callable[[Iterable], Iterable]] = None,
) -> pd.DataFrame:
    """
    Run *generate* (e.g. ``SAM2AutomaticMaskGenerator.generate``) on overlapping *patch* x *patch*
    tiles of *image* (H, W, 3) enlarged by *scale*, and return one row per fibre:
    ``x, y, a, b, angle, r_eq, r_ell, area`` (original pixels), ``score`` (predicted IoU),
    ``stability``, ``edge_dist`` (centre to the nearest interior patch border), ``patch``.

    *overlap* should exceed two fibre diameters so that every fibre lies whole, with context, inside
    one patch.  Masks touching an interior patch border are dropped (the fibre is complete in the
    neighbouring patch); masks touching the image / ROI border are kept.  The result still holds
    every fibre once per patch that saw it whole: pass it through :func:`suppress_duplicates`.
    """
    H, W = image.shape[:2]
    ry0, rx0, ry1, rx1 = roi if roi is not None else (0, 0, H, W)
    grid = patch_grid((H, W), patch, overlap, roi)
    it = progress(grid) if progress is not None else grid
    rows = []
    for k, (y0, x0, y1, x1) in enumerate(it):
        tile = np.ascontiguousarray(image[y0:y1, x0:x1])
        h, w = tile.shape[:2]
        sh, sw = int(round(h * scale)), int(round(w * scale))
        sy, sx = sh / h, sw / w
        up = cv2.resize(tile, (sw, sh), interpolation=cv2.INTER_CUBIC) if scale != 1 else tile
        interior = (y0 > ry0, y1 < ry1, x0 > rx0, x1 < rx1)       # top, bottom, left, right
        for d in generate(up):
            seg = np.asarray(d["segmentation"], dtype=bool)
            if any(t and i for t, i in zip(_border_flags(seg), interior)):
                continue
            g = mask_ellipse(seg)
            if g is None:
                continue
            X, Y = x0 + g["x"] / sx, y0 + g["y"] / sy
            # context on the tightest side: distance of the centre to the nearest interior border
            edge = min([v for v, i in zip((Y - y0, y1 - Y, X - x0, x1 - X), interior) if i] or [np.inf])
            s = np.sqrt(sx * sy)
            rows.append({"x": X, "y": Y, "a": g["a"] / s, "b": g["b"] / s, "angle": g["angle"],
                         "r_eq": g["r_eq"] / s, "r_ell": g["r_ell"] / s, "area": g["area"] / (sx * sy),
                         "score": float(d.get("predicted_iou", np.nan)),
                         "stability": float(d.get("stability_score", np.nan) or np.nan),
                         "edge_dist": float(min(edge, 1e6)), "patch": k})
    cols = ["x", "y", "a", "b", "angle", "r_eq", "r_ell", "area", "score", "stability", "edge_dist", "patch"]
    return pd.DataFrame(rows, columns=cols)


def suppress_duplicates(df: pd.DataFrame, r_col: str = "r_ell", frac: float = 0.5,
                        by: Sequence[str] = ("edge_dist", "score"), edge_step: float = 2.0) -> pd.DataFrame:
    """
    Remove detections whose centre lies within ``frac * r`` of a preferred one (the same fibre seen
    by several overlapping patches, or a fibre and a fragment of it).  Preference: larger *by*
    columns, in order; ``edge_dist`` (context to the patch border) is binned to *edge_step* px so
    that the score decides between detections with about the same context.
    """
    if len(df) < 2:
        return df.copy()
    keys = []
    for c in by:
        if c not in df:
            continue
        v = df[c].fillna(-np.inf).to_numpy(float)
        keys.append(np.floor(v / edge_step) if c == "edge_dist" else v)
    order = np.lexsort(keys[::-1])[::-1]               # first column = primary key, descending
    xy = df[["x", "y"]].to_numpy()
    r = df[r_col].to_numpy()
    tree = cKDTree(xy)
    keep = np.ones(len(df), bool)
    for i in order:
        if not keep[i]:
            continue
        for j in tree.query_ball_point(xy[i], frac * r[i]):
            if j != i and keep[j]:
                keep[j] = False
    return df.iloc[np.flatnonzero(keep)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# expert reference
# ----------------------------------------------------------------------------------------------
def read_annotator(csv_path) -> pd.DataFrame:
    """One expert's Fiji ellipse fits -> ``x, y, r`` (r = (Major + Minor) / 4), px, subframe origin."""
    d = pd.read_csv(csv_path)
    return pd.DataFrame({"x": d["X"].to_numpy(float), "y": d["Y"].to_numpy(float),
                         "r": (d["Major"].to_numpy(float) + d["Minor"].to_numpy(float)) / 4})


def _nearest_within(src: pd.DataFrame, dst: pd.DataFrame, frac: float = 0.5) -> np.ndarray:
    """Index into *dst* of the nearest fibre within ``frac * r_src`` of each *src* fibre, -1 if none."""
    d, j = cKDTree(dst[["x", "y"]].to_numpy()).query(src[["x", "y"]].to_numpy())
    return np.where(d < frac * src["r"].to_numpy(), j, -1)


def expert_consensus(annotators: Sequence[pd.DataFrame], frac: float = 0.5) -> pd.DataFrame:
    """
    Fibres marked by *all* annotators: pairwise nearest-neighbour matching with a ``frac * r``
    (half-radius) proximity criterion, kept only if the matches are consistent across every pair.
    Consensus ``x, y, r`` = mean over annotators; ``r_sd`` = their standard deviation.
    """
    base = annotators[0]
    idx = [np.arange(len(base))] + [_nearest_within(base, o, frac) for o in annotators[1:]]
    ok = np.all(np.stack(idx) >= 0, axis=0)
    # pairwise consistency among the other annotators
    for a in range(1, len(annotators)):
        for b in range(a + 1, len(annotators)):
            ab = _nearest_within(annotators[a], annotators[b], frac)
            ok &= np.where(idx[a] >= 0, ab[np.clip(idx[a], 0, None)] == idx[b], False)
    sel = np.flatnonzero(ok)
    stack = {c: np.stack([annotators[k][c].to_numpy()[idx[k][sel]] for k in range(len(annotators))])
             for c in ("x", "y", "r")}
    return pd.DataFrame({"x": stack["x"].mean(0), "y": stack["y"].mean(0), "r": stack["r"].mean(0),
                         "r_sd": stack["r"].std(0, ddof=1), "xy_sd": np.hypot(stack["x"].std(0, ddof=1),
                                                                             stack["y"].std(0, ddof=1))})


def match_reference(ref: pd.DataFrame, det: pd.DataFrame, r_col: str = "r") -> Tuple[pd.DataFrame, Dict]:
    """
    Paper sec. 2.3.1: each reference fibre is matched to the detections whose centre lies within its
    radius; none -> unmatched, more than one -> multi-matched, the nearest is kept.

    Returns the per-reference-fibre table (``n_cand``, ``r_det``, ``r_error`` = (r_det - r_ref)/r_ref,
    ``p_error`` = centre distance, ``dx``, ``dy``) and a summary dict.  Detections not matched by any
    reference fibre are counted as ``unmatched_det`` (only meaningful away from the subframe border,
    where the consensus is incomplete).
    """
    dxy = det[["x", "y"]].to_numpy()
    tree = cKDTree(dxy) if len(det) else None
    rows, used = [], set()
    for x, y, r in ref[["x", "y", "r"]].to_numpy():
        cand = tree.query_ball_point((x, y), r) if tree is not None else []
        row = {"x": x, "y": y, "r": r, "n_cand": len(cand)}
        if cand:
            dist = np.hypot(dxy[cand, 0] - x, dxy[cand, 1] - y)
            j = cand[int(np.argmin(dist))]
            used.add(j)
            rd = float(det[r_col].iloc[j])
            row.update(r_det=rd, r_error=(rd - r) / r, p_error=float(dist.min()),
                       dx=float(dxy[j, 0] - x), dy=float(dxy[j, 1] - y), det_idx=j)
        rows.append(row)
    m = pd.DataFrame(rows)
    matched = m["n_cand"] > 0
    summary = {"n_ref": len(ref), "unmatched": int((~matched).sum()), "multi_matched": int((m["n_cand"] > 1).sum()),
               "n_matched": int(matched.sum()),
               "r_error_mean": float(m.loc[matched, "r_error"].mean()),
               "r_error_abs_mean": float(m.loc[matched, "r_error"].abs().mean()),
               "p_error_mean_px": float(m.loc[matched, "p_error"].mean()),
               "p_error_median_px": float(m.loc[matched, "p_error"].median()),
               "dx_mean_px": float(m.loc[matched, "dx"].mean()), "dy_mean_px": float(m.loc[matched, "dy"].mean()),
               "unmatched_det_idx": sorted(set(range(len(det))) - used)}
    return m, summary


# ----------------------------------------------------------------------------------------------
# paper metrics
# ----------------------------------------------------------------------------------------------
def vf_overlap(x, y, r, roi_hw: Tuple[float, float]) -> float:
    """Homogenised Vf summing every fibre's disk area (paper Fig. 3b) over the ROI ``(H, W)``."""
    return float(np.pi * np.sum(np.square(r)) / (roi_hw[0] * roi_hw[1]))


def vf_union(x, y, r, roi_hw: Tuple[int, int], supersample: int = 4) -> float:
    """Homogenised Vf of the boolean union of the fibre disks (paper Fig. 3a), rasterised with
    *supersample* sub-pixels per pixel side: a sub-pixel is covered if its centre lies inside a disk
    (unbiased, unlike cv2.circle, whose filled circles include the boundary pixels)."""
    H, W = int(np.ceil(roi_hw[0])), int(np.ceil(roi_hw[1]))
    s = int(supersample)
    canvas = np.zeros((H * s, W * s), bool)
    x, y, r = (np.asarray(v, float) * s for v in (x, y, r))
    m = int(np.ceil(r.max())) + 1
    off = np.arange(-m, m + 1)
    for cx, cy, rr in zip(x, y, r):
        ix, iy = int(np.floor(cx)), int(np.floor(cy))
        xs, ys = ix + off, iy + off
        inside = (xs[None, :] + 0.5 - cx) ** 2 + (ys[:, None] + 0.5 - cy) ** 2 <= rr * rr
        vy, vx = (ys >= 0) & (ys < H * s), (xs >= 0) & (xs < W * s)
        if not (vy.any() and vx.any()):
            continue
        sub = inside[np.ix_(vy, vx)]
        canvas[ys[vy][0]:ys[vy][-1] + 1, xs[vx][0]:xs[vx][-1] + 1] |= sub
    return float(canvas.mean())


def local_vf(x, y, r, roi_hw: Tuple[float, float]) -> np.ndarray:
    """
    Per-fibre local Vf = disk area / Voronoi cell area (paper Fig. 4).  Cells that are unbounded or
    reach outside the ROI ``(H, W)`` are NaN.
    """
    xy = np.column_stack([x, y])
    out = np.full(len(xy), np.nan)
    if len(xy) < 4:
        return out
    vor = Voronoi(xy)
    H, W = roi_hw
    V = vor.vertices
    for i, reg in enumerate(vor.point_region):
        idx = vor.regions[reg]
        if not idx or -1 in idx:
            continue
        P = V[idx]
        if P[:, 0].min() < 0 or P[:, 1].min() < 0 or P[:, 0].max() > W or P[:, 1].max() > H:
            continue
        area = 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))
        if area > 0:
            out[i] = np.pi * r[i] ** 2 / area
    return out


def first_neighbour_distance(x, y, r, k: int = 8) -> np.ndarray:
    """Boundary-to-boundary distance to the closest fibre (min over the *k* nearest centres);
    negative = overlap (paper Fig. 4)."""
    xy = np.column_stack([x, y])
    r = np.asarray(r, float)
    k = min(k + 1, len(xy))
    d, j = cKDTree(xy).query(xy, k=k)
    gap = d[:, 1:] - r[:, None] - r[j[:, 1:]]
    return gap.min(axis=1)


def summary_metrics(x, y, r, roi_hw: Tuple[int, int], px_um: float, union: bool = True) -> Dict[str, float]:
    """Macroscale descriptors of one fibre set: count, mean / sd radius (um), Vf (union, overlap)."""
    r = np.asarray(r, float)
    out = {"n": int(len(r)), "r_mean_um": float(r.mean() * px_um), "r_sd_um": float(r.std(ddof=1) * px_um),
           "vf_overlap": vf_overlap(x, y, r, roi_hw)}
    if union:
        out["vf_union"] = vf_union(x, y, r, roi_hw)
    return out

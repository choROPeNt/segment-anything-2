"""
Evaluate SAM 2 checkpoints on SA-1B style windows (as written by :mod:`sam2.addons.sam2_dataset`).

* point prompts: one click per ground-truth instance → per-instance IoU (the task SAM 2 is fine-tuned on)
* automatic mask generation: match predicted to ground-truth instances → precision / recall / F1
* leakage check: how much of an evaluation window was covered by the training windows
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt


def load_sa1b_window(dataset_dir, name):
    """
    Load one window of an SA-1B dataset folder.

    Returns
    -------
    image : (H, W, 3) uint8
    masks : (N, H, W) bool — ground-truth instances
    ids   : (N,) int       — ``instance_id`` of each mask
    """
    dataset_dir = Path(dataset_dir)
    image = np.array(Image.open(dataset_dir / "images" / f"{name}.jpg").convert("RGB"))   # writable copy
    with open(dataset_dir / "annotations" / f"{name}.json", encoding="utf-8") as f:
        annots = json.load(f)["annotations"]
    if not annots:
        return image, np.zeros((0,) + image.shape[:2], dtype=bool), np.zeros(0, dtype=int)
    masks = np.stack([mask_utils.decode(a["segmentation"]).astype(bool) for a in annots])
    ids = np.array([a.get("instance_id", i + 1) for i, a in enumerate(annots)])
    return image, masks, ids


def interior_points(masks):
    """One click per mask at its most interior pixel (max. distance to the boundary) → (N, 2) [x, y]."""
    pts = np.zeros((len(masks), 2), dtype=np.float32)
    for i, m in enumerate(masks):
        d = np.asarray(distance_transform_edt(np.pad(m, 1)))[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(d), d.shape)
        pts[i] = x, y
    return pts


def predict_from_points(predictor, image, points, batch_size=64):
    """
    Single positive click per point with a :class:`SAM2ImagePredictor` → (N, H, W) bool, (N,) scores.

    The image embedding is computed once; points are decoded in batches.
    """
    predictor.set_image(image)
    out_masks, out_scores = [], []
    for s in range(0, len(points), batch_size):
        p = points[s:s + batch_size]
        masks, scores, _ = predictor.predict(
            point_coords=p[:, None, :],
            point_labels=np.ones((len(p), 1), dtype=np.int32),
            multimask_output=False,
        )
        masks = np.asarray(masks).reshape(len(p), *image.shape[:2])
        out_masks.append(masks > 0)
        out_scores.append(np.asarray(scores).reshape(len(p)))
    if not out_masks:
        return np.zeros((0,) + image.shape[:2], dtype=bool), np.zeros(0)
    return np.concatenate(out_masks), np.concatenate(out_scores)


def mask_iou(a, b):
    """IoU of two equally long stacks of masks, pairwise per index → (N,)."""
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    inter = (a & b).sum(1)
    union = (a | b).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def iou_matrix(pred, gt):
    """IoU between every predicted and every ground-truth mask → (P, G)."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = pred.reshape(len(pred), -1).astype(np.float32)
    g = gt.reshape(len(gt), -1).astype(np.float32)
    inter = p @ g.T
    union = p.sum(1)[:, None] + g.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)


def match_instances(pred, gt, iou_thr=0.5):
    """
    Greedy one-to-one matching of predicted to ground-truth instances at *iou_thr*.

    Returns dict with precision, recall, f1, mean IoU of the matches and the counts.
    """
    ious = iou_matrix(pred, gt)
    matched_p, matched_g, matched_iou = set(), set(), []
    for p, g in sorted(zip(*np.nonzero(ious >= iou_thr)), key=lambda pg: -ious[pg]):
        if p in matched_p or g in matched_g:
            continue
        matched_p.add(p); matched_g.add(g); matched_iou.append(ious[p, g])
    tp = len(matched_iou)
    precision = tp / len(pred) if len(pred) else 0.0
    recall = tp / len(gt) if len(gt) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched_iou": float(np.mean(matched_iou)) if matched_iou else 0.0,
            "n_pred": len(pred), "n_gt": len(gt), "tp": tp}


def semantic_metrics(pred, gt):
    """
    Fibre / background comparison of the instance unions (class level, instances ignored).

    Returns dict with pixel IoU, Dice, FVG of ground truth and prediction and their difference
    (FVG = foreground fraction of the window).
    """
    p = pred.any(0) if len(pred) else np.zeros(gt.shape[1:], dtype=bool)
    g = gt.any(0) if len(gt) else np.zeros(pred.shape[1:], dtype=bool)
    inter, union = np.count_nonzero(p & g), np.count_nonzero(p | g)
    total = p.size
    return {
        "pixel_iou": inter / union if union else 1.0,
        "dice"     : 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0,
        "fvg_gt"   : g.sum() / total,
        "fvg_pred" : p.sum() / total,
        "fvg_error": (p.sum() - g.sum()) / total,
    }


def train_overlap(eval_manifest, train_manifest, train_names=None):
    """
    Fraction of each evaluation window covered by training windows of the same specimen.

    Parameters
    ----------
    eval_manifest  : DataFrame (``windows.csv`` rows) of the evaluated windows
    train_manifest : DataFrame of the training dataset's ``windows.csv``
    train_names    : optional set of window names actually used for training (e.g. ``train.txt``)

    Returns
    -------
    numpy array (one value per evaluation window): max. covered fraction, 0 = no overlap
    """
    tr = train_manifest
    if train_names is not None:
        tr = tr[tr["name"].isin(train_names)]
    out = np.zeros(len(eval_manifest))
    for k, r in enumerate(eval_manifest.itertuples()):
        t = tr[tr["specimen"] == r.specimen]
        if not len(t):
            continue
        ix = np.clip(np.minimum(t["x0"] + t["width"], r.x0 + r.width) - np.maximum(t["x0"], r.x0), 0, None)
        iy = np.clip(np.minimum(t["y0"] + t["height"], r.y0 + r.height) - np.maximum(t["y0"], r.y0), 0, None)
        out[k] = float((ix * iy).max()) / (r.width * r.height)
    return out


def masks_from_amg(anns, shape):
    """Stack the ``segmentation`` masks of :class:`SAM2AutomaticMaskGenerator` output → (N, H, W) bool."""
    if not anns:
        return np.zeros((0,) + tuple(shape), dtype=bool)
    return np.stack([np.asarray(a["segmentation"], dtype=bool) for a in anns])

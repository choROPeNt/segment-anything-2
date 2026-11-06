"""
Tuning the SAM 2 automatic mask generator (AMG) against labelled windows.

The AMG settings fall into two groups:

* **generation** settings change what the model predicts and need a model run:
  ``points_per_side``, ``multimask_output``, ``use_m2m``, ``min_mask_region_area``, patch upscaling
* **filter** settings are thresholds on what every candidate mask already carries:
  ``pred_iou_thresh``, ``stability_score_thresh``, ``box_nms_thresh`` and the area filter of
  ``scripts/analyse.py``

:func:`raw_mask_generator` runs the AMG once with the loosest filters of a sweep; :func:`filter_candidates`
then reproduces any stricter filter setting offline, in the order the AMG + ``analyse.py`` apply them
(predicted IoU and stability → box NMS by predicted IoU → area filter).  Exact for the IoU / stability / area
filters; the NMS is exact when the generation ran without NMS (``nms_ceiling=1``) and near-exact otherwise
(greedy NMS at a lower threshold on an already NMS-thinned set).

:func:`window_candidates` + :func:`score_selection` evaluate a selection against ground truth with an
*ignore region* (outside the ROI, unlabelled fibres such as instances cut at the window border): an
unmatched prediction lying mostly there is neither a hit nor a false positive.
"""

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms


def raw_mask_generator(model, points_per_side=32, multimask_output=False, use_m2m=False,
                       min_mask_region_area=150, pred_iou_floor=0.0, stability_floor=0.0, nms_ceiling=1.0,
                       points_per_batch=64):
    """
    :class:`SAM2AutomaticMaskGenerator` with the loosest filters of a sweep: everything with predicted
    IoU > *pred_iou_floor* and stability ≥ *stability_floor* is kept, box NMS only above *nms_ceiling*.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_floor,
        stability_score_thresh=stability_floor,
        box_nms_thresh=nms_ceiling,
        min_mask_region_area=min_mask_region_area,
        multimask_output=multimask_output,
        use_m2m=use_m2m,
    )


def window_candidates(anns, gt_masks, roi=None, ignore=None, fibre=None):
    """
    Candidate masks of one window (AMG output) with everything needed to score any filter setting.

    Parameters
    ----------
    anns : list of AMG detections (``segmentation``, ``predicted_iou``, ``stability_score``, ``bbox`` XYWH)
    gt_masks : (G, H, W) bool, disjoint ground-truth instances
    roi : (H, W) bool evaluation region for the phase fraction (default: whole window)
    ignore : (H, W) bool pixels where predictions are not counted as false positives (default: none)
    fibre : (H, W) bool true fibre pixels for the phase fraction, e.g. all labelled fibres incl. the ones not
        used as ground truth (default: union of *gt_masks*)

    Returns
    -------
    dict with ``piou``, ``stab``, ``area``, ``boxes`` (N, 4) XYXY, ``pix`` (list of flat pixel indices),
    ``iou`` (N, G) IoU with every ground-truth mask, ``ignore_frac`` (N,), ``pairs`` (candidate, gt, IoU)
    with IoU ≥ 0.5 sorted by IoU (for greedy matching), ``n_gt``, ``roi`` (flat bool), ``gt_px`` (number of
    fibre pixels inside the ROI)
    """
    G = len(gt_masks)
    H, W = gt_masks.shape[1:] if G else np.asarray(anns[0]["segmentation"]).shape
    gt_id = np.zeros(H * W, dtype=np.int32)
    for k, m in enumerate(gt_masks):
        gt_id[m.ravel()] = k + 1
    gt_area = np.bincount(gt_id, minlength=G + 1)[1:]
    roi = np.ones(H * W, dtype=bool) if roi is None else np.asarray(roi, dtype=bool).ravel()
    ign = np.zeros(H * W, dtype=bool) if ignore is None else np.asarray(ignore, dtype=bool).ravel()
    fib = gt_id > 0 if fibre is None else np.asarray(fibre, dtype=bool).ravel()

    N = len(anns)
    pix, iou, ign_frac = [], np.zeros((N, G)), np.zeros(N)
    for i, a in enumerate(anns):
        p = np.flatnonzero(np.asarray(a["segmentation"], dtype=bool))
        pix.append(p)
        if len(p):
            inter = np.bincount(gt_id[p], minlength=G + 1)[1:]
            iou[i] = inter / np.maximum(len(p) + gt_area - inter, 1)
            ign_frac[i] = ign[p].mean()
    area = np.array([len(p) for p in pix], dtype=float)
    boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                      for a in anns], dtype=np.float32).reshape(N, 4)
    cp, cg = np.nonzero(iou >= 0.5)
    order = np.argsort(-iou[cp, cg], kind="stable")
    return {
        "piou": np.array([a["predicted_iou"] for a in anns], dtype=float),
        "stab": np.array([a["stability_score"] for a in anns], dtype=float),
        "area": area, "boxes": boxes, "pix": pix, "iou": iou, "ignore_frac": ign_frac,
        "pairs": np.stack([cp[order], cg[order]], 1) if len(cp) else np.zeros((0, 2), dtype=int),
        "n_gt": G, "roi": roi, "gt_px": int(np.count_nonzero(fib & roi)), "shape": (H, W),
    }


def filter_candidates(c, pred_iou_thresh, stability_score_thresh, box_nms_thresh, min_area=0, max_area=None):
    """Indices of the candidates that survive one filter setting (same order of steps as AMG + analyse.py)."""
    idx = np.flatnonzero((c["piou"] > pred_iou_thresh) & (c["stab"] >= stability_score_thresh))
    if len(idx) == 0:
        return idx
    if box_nms_thresh < 1.0:
        keep = batched_nms(torch.from_numpy(c["boxes"][idx]), torch.from_numpy(c["piou"][idx]).float(),
                           torch.zeros(len(idx), dtype=torch.int64), iou_threshold=float(box_nms_thresh))
        idx = idx[keep.numpy()]
    area = c["area"][idx]
    ok = area >= min_area
    if max_area is not None:
        ok &= area < max_area
    return idx[ok]


def score_selection(c, keep, ignore_thr=0.5):
    """
    Counts of one selection: greedy one-to-one matching at IoU ≥ 0.5 (as
    :func:`sam2.addons.evaluation.match_instances`), unmatched predictions with ≥ *ignore_thr* of their area in
    the ignore region are ``ignored``.

    Returns dict with tp, fp, fn, ignored, n_pred, iou_sum (sum of matched IoUs), pred_px / gt_px / roi_px
    (fibre pixels of the prediction and the truth inside the ROI, ROI size).
    """
    in_keep = np.zeros(len(c["piou"]), dtype=bool)
    in_keep[keep] = True
    used_p, used_g, iou_sum = set(), set(), 0.0
    for p, g in c["pairs"]:
        if in_keep[p] and p not in used_p and g not in used_g:
            used_p.add(p); used_g.add(g); iou_sum += c["iou"][p, g]
    tp = len(used_p)
    unmatched = np.array([k for k in keep if k not in used_p], dtype=int)
    ignored = int(np.count_nonzero(c["ignore_frac"][unmatched] >= ignore_thr)) if len(unmatched) else 0

    H, W = c["shape"]
    union = np.zeros(H * W, dtype=bool)
    for k in keep:
        union[c["pix"][k]] = True
    roi = c["roi"]
    return {"tp": tp, "fp": len(keep) - tp - ignored, "fn": c["n_gt"] - tp, "ignored": ignored,
            "n_pred": len(keep), "iou_sum": iou_sum, "pred_px": int(np.count_nonzero(union & roi)),
            "gt_px": c["gt_px"], "roi_px": int(np.count_nonzero(roi))}


def summarize_counts(df, by):
    """
    Pool the per-window counts of :func:`score_selection` (a DataFrame) per setting → precision, recall, F1,
    matched IoU, phase-fraction bias (pooled over all ROI pixels) and mean absolute per-window error.
    """
    import pandas as pd

    d = df.assign(fvg_err=(df["pred_px"] - df["gt_px"]) / df["roi_px"].clip(lower=1))
    g = d.groupby(by, sort=False)
    s = g[["tp", "fp", "fn", "ignored", "n_pred", "iou_sum", "pred_px", "gt_px", "roi_px"]].sum()
    out = pd.DataFrame(index=s.index)
    out["precision"] = s["tp"] / (s["tp"] + s["fp"]).clip(lower=1)
    out["recall"] = s["tp"] / (s["tp"] + s["fn"]).clip(lower=1)
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"]).where(
        out["precision"] + out["recall"] > 0, 1)
    out["matched_iou"] = s["iou_sum"] / s["tp"].clip(lower=1)
    out["fvg_bias"] = (s["pred_px"] - s["gt_px"]) / s["roi_px"].clip(lower=1)
    out["fvg_mae"] = g["fvg_err"].apply(lambda v: v.abs().mean())
    out[["tp", "fp", "fn", "ignored"]] = s[["tp", "fp", "fn", "ignored"]]
    return out.reset_index()

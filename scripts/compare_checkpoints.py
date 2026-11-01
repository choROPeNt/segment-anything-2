"""
Compare a fine-tuned SAM 2 checkpoint with the base checkpoint on validation windows.

For every window of the chosen split (dataset folder written by h5_to_sam2_dataset.py):

* points — one click per ground-truth fibre → per-fibre IoU (the task the model is fine-tuned on)
* amg    — automatic mask generation → precision / recall / F1 of the matched fibres (IoU >= 0.5)
* class  — automatic mask generation, fibre / background only (instances ignored): pixel IoU, Dice
           and FVG of the window; figures show error maps in plasma colours (yellow = extra fibre, indigo = missed fibre)

Class metrics (class_*) are also computed in the amg and both modes.

Writes metrics.csv (one row per window and model), summary.csv and comparison PNGs
(image + ground truth | base | fine-tuned) to --out.

    python scripts/compare_checkpoints.py \\
        --dataset data/fibre_ds_512_train-val --split val \\
        --ckpt /Volumes/warm_SD/Projects/segement-anything-2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_fibre_ds_512.yaml/checkpoints/checkpoint.pt

If the training run's config_resolved.yaml is found next to the checkpoint, the evaluated windows are
checked against the training windows and a warning is printed when they overlap (leakage).
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from sam2.addons.evaluation import (
    interior_points,
    load_sa1b_window,
    mask_iou,
    masks_from_amg,
    match_instances,
    predict_from_points,
    semantic_metrics,
    train_overlap,
)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def overlay(ax, image, masks, title, seed=0):
    """Image with every instance in a random plasma colour and a white outline."""
    import cv2
    from matplotlib import colormaps

    ax.imshow(image)
    rng = np.random.default_rng(seed)
    cmap = colormaps["plasma"]
    rgba = np.zeros(image.shape[:2] + (4,), dtype=np.float32)
    for m in masks:
        rgba[m] = (*cmap(int(rng.integers(0, 256)))[:3], 0.5)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color="white", lw=0.5)
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# residual colours from the plasma colormap: (label, plasma position, alpha)
RESIDUAL_COLORS = {
    "correct": ("correct fibre", 0.50, 0.35),
    "extra"  : ("extra (pred, not GT)", 0.95, 0.85),
    "missed" : ("missed (GT, not pred)", 0.05, 0.85),
}


def residual_rgba(key):
    from matplotlib import colormaps

    _, pos, alpha = RESIDUAL_COLORS[key]
    return (*colormaps["plasma"](pos)[:3], alpha)


def error_map(ax, image, pred, gt, title):
    """Fibre / background errors on the image in plasma colours: magenta = correct, yellow = extra, indigo = missed."""
    p = pred.any(0) if len(pred) else np.zeros(image.shape[:2], dtype=bool)
    g = gt.any(0) if len(gt) else np.zeros(image.shape[:2], dtype=bool)
    rgba = np.zeros(image.shape[:2] + (4,), dtype=np.float32)
    rgba[p & g] = residual_rgba("correct")
    rgba[p & ~g] = residual_rgba("extra")
    rgba[~p & g] = residual_rgba("missed")
    ax.imshow(image)
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_summary(df, path):
    """Paired plot per metric: one line per window from base to fine-tuned, mean as a bold marker."""
    metrics = [(c, t) for c, t in (("point_miou", "mIoU — 1 click / fibre"), ("point_iou_ge_0_9", "share IoU ≥ 0.9"),
                                   ("amg_f1", "F1 — automatic"), ("amg_matched_iou", "matched IoU — automatic"),
                                   ("class_pixel_iou", "pixel IoU — fibre / background"),
                                   ("class_fvg_error", "FVG error (pred − GT)"))
               if c in df]
    colors = {"base": "0.55", "fine-tuned": "steelblue"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.8), squeeze=False)
    for ax, (col, title) in zip(axes[0], metrics):
        wide = df.pivot(index="name", columns="model", values=col)[["base", "fine-tuned"]]
        for _, r in wide.iterrows():
            ax.plot([0, 1], r.values, color="0.8", lw=1, zorder=1)
        for x, model in enumerate(wide.columns):
            ax.scatter(np.full(len(wide), x), wide[model], s=14, color=colors[model], zorder=2)
            ax.scatter(x, wide[model].mean(), s=90, marker="D", color=colors[model], edgecolor="black", zorder=3)
            ax.annotate(f"{wide[model].mean():.3f}", (x, wide[model].mean()), xytext=(10 if x else -10, 0),
                        textcoords="offset points", ha="left" if x else "right", va="center", fontsize=8)
        better = int((wide["fine-tuned"] > wide["base"]).sum())
        ax.set_xticks([0, 1], ["base", "fine-tuned"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{title}\nfine-tuned better: {better}/{len(wide)}", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Base vs. fine-tuned on {df['name'].nunique()} windows (◆ = mean)", fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def training_windows(ckpt):
    """(train manifest, train names) of the run that produced *ckpt*, from its config_resolved.yaml."""
    cfg = Path(ckpt).resolve().parent.parent / "config_resolved.yaml"
    if not cfg.exists():
        return None, None
    ds = yaml.safe_load(cfg.read_text()).get("dataset", {})
    img_folder = Path(ds.get("img_folder", ""))
    manifest = img_folder.parent / "windows.csv"
    if not manifest.exists():
        return None, None
    names = None
    if ds.get("file_list_txt") and Path(ds["file_list_txt"]).exists():
        names = set(Path(ds["file_list_txt"]).read_text().split())
    return pd.read_csv(manifest), names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset root (images/, annotations/, windows.csv)")
    parser.add_argument("--split", default="val", help="split to evaluate: reads <dataset>/<split>.txt (default: val)")
    parser.add_argument("--ckpt", type=Path, required=True, help="fine-tuned checkpoint (.pt)")
    parser.add_argument("--base-ckpt", type=Path, default=Path("checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml", help="model config (both checkpoints)")
    parser.add_argument("--mode", choices=["points", "amg", "both", "class"], default="both")
    parser.add_argument("--max-windows", type=int, default=None, help="evaluate at most N windows (default: all)")
    parser.add_argument("--n-vis", type=int, default=6, help="number of windows to plot (default: 6)")
    parser.add_argument("--points-per-side", type=int, default=16,
                        help="AMG grid (default: 16 — enough for ~60 px fibres in 512 px windows; "
                             "32 needs about twice the GPU memory and 3-4x the time)")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: <run>/eval_<split> next to the checkpoint)")
    args = parser.parse_args()

    out = args.out or args.ckpt.resolve().parent.parent / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    manifest = pd.read_csv(args.dataset / "windows.csv")
    names = (args.dataset / f"{args.split}.txt").read_text().split()
    rows = manifest.set_index("name").loc[names].reset_index()
    if args.max_windows:
        rows = rows.sample(min(args.max_windows, len(rows)), random_state=args.seed).reset_index(drop=True)
    vis_names = set(rows.sample(min(args.n_vis, len(rows)), random_state=args.seed)["name"])

    # --- leakage check against the training windows of this run ---
    train_manifest, train_names = training_windows(args.ckpt)
    if train_manifest is not None:
        cov = train_overlap(rows, train_manifest, train_names)
        rows["train_overlap"] = cov
        if (cov > 0).any():
            print(f"WARNING: {np.count_nonzero(cov > 0)}/{len(rows)} evaluated windows overlap the training windows "
                  f"of this checkpoint (median coverage {np.median(cov[cov > 0]):.0%}) — the fine-tuned scores "
                  "are optimistic (data leakage).")
        else:
            print("Leakage check: no evaluated window overlaps the training windows of this checkpoint.")
    else:
        print("Leakage check skipped: training dataset of this checkpoint not found.")

    models = {}
    for label, ckpt in (("base", args.base_ckpt), ("fine-tuned", args.ckpt)):
        sam = build_sam2(args.config, str(ckpt), device=device)
        models[label] = {
            "predictor": SAM2ImagePredictor(sam),
            "amg": SAM2AutomaticMaskGenerator(sam, points_per_side=args.points_per_side) if args.mode != "points" else None,
        }
    print(f"Evaluating {len(rows)} windows of '{args.split}' on {device} ({args.mode})")

    records = []
    t0 = time.time()
    for k, r in enumerate(rows.itertuples()):
        image, gt, _ = load_sa1b_window(args.dataset, r.name)
        pts = interior_points(gt)
        preds = {}
        for label, m in models.items():
            rec = {"name": r.name, "model": label, "n_gt": len(gt)}
            if args.mode in ("points", "both"):
                with torch.inference_mode():
                    pm, _ = predict_from_points(m["predictor"], image, pts)
                ious = mask_iou(pm, gt) if len(gt) else np.zeros(0)
                rec.update(point_miou=float(ious.mean()) if len(ious) else np.nan,
                           point_iou_ge_0_9=float((ious >= 0.9).mean()) if len(ious) else np.nan)
                preds[(label, "points")] = (pm, ious)
            if args.mode in ("amg", "both", "class"):
                with torch.inference_mode():
                    am = masks_from_amg(m["amg"].generate(image), image.shape[:2])
                sem = semantic_metrics(am, gt)
                rec.update({f"class_{key}": v for key, v in sem.items()})
                if args.mode == "class":
                    preds[(label, "class")] = (am, sem)
                else:
                    met = match_instances(am, gt)
                    rec.update({f"amg_{key}": v for key, v in met.items() if key not in ("n_gt",)})
                    preds[(label, "amg")] = (am, met)
            records.append(rec)

        if r.name in vis_names:
            cols = [("ground truth", None)] + [(lab, mode) for mode in ("points", "amg", "class")
                                               for lab in models if (lab, mode) in preds]
            fvg_gt = gt.any(0).mean() if len(gt) else 0.0
            if args.mode == "class":
                fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.3), squeeze=False)
                error_map(axes[0, 0], image, gt, gt, f"{r.name}\nground truth — FVG {fvg_gt:.3f}")
                for ax, (lab, mode) in zip(axes[0, 1:], cols[1:]):
                    masks, info = preds[(lab, mode)]
                    error_map(ax, image, masks, gt,
                              f"{lab} — fibre / background\nIoU {info['pixel_iou']:.3f}, "
                              f"FVG {info['fvg_pred']:.3f} ({info['fvg_error']:+.3f})")
            else:
                # row 1: instances (random plasma colours), row 2: residuals against the ground truth
                fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 8.6))
                overlay(axes[0, 0], image, gt, f"{r.name}\nground truth ({len(gt)} fibres)")
                axes[0, 0].plot(pts[:, 0], pts[:, 1], "o", ms=2.5, mfc="white", mec="black", mew=0.3)
                error_map(axes[1, 0], image, gt, gt, f"residual legend — GT FVG {fvg_gt:.3f}")
                axes[1, 0].legend(
                    handles=[plt.matplotlib.patches.Patch(color=residual_rgba(k)[:3], label=RESIDUAL_COLORS[k][0])
                             for k in ("correct", "extra", "missed")],
                    loc="lower left", fontsize=7, framealpha=0.9)
                for c, (lab, mode) in enumerate(cols[1:], start=1):
                    masks, info = preds[(lab, mode)]
                    if mode == "points":
                        title = f"{lab} — 1 click / fibre\nmIoU {info.mean():.3f}, IoU≥0.9: {(info >= 0.9).mean():.0%}"
                    else:
                        title = f"{lab} — automatic\nF1 {info['f1']:.3f}  ({info['n_pred']} pred / {info['n_gt']} gt)"
                    overlay(axes[0, c], image, masks, title)
                    sem = semantic_metrics(masks, gt)
                    p = masks.any(0) if len(masks) else np.zeros(image.shape[:2], dtype=bool)
                    g = gt.any(0) if len(gt) else np.zeros(image.shape[:2], dtype=bool)
                    error_map(axes[1, c], image, masks, gt,
                              f"residual — pixel IoU {sem['pixel_iou']:.3f}\n"
                              f"+{np.count_nonzero(p & ~g):,} / −{np.count_nonzero(~p & g):,} px, "
                              f"ΔFVG {sem['fvg_error']:+.3f}")
            plt.tight_layout()
            fig.savefig(out / f"{r.name}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)

        # free GPU memory of the automatic mask generation before the next window
        del preds
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        print(f"  [{k + 1}/{len(rows)}] {r.name}  ({time.time() - t0:.0f} s)", end="\r", flush=True)
    print()

    df = pd.DataFrame(records)
    df.to_csv(out / "metrics.csv", index=False)
    metric_cols = [c for c in ("point_miou", "point_iou_ge_0_9", "amg_f1", "amg_precision", "amg_recall",
                               "amg_matched_iou", "class_pixel_iou", "class_dice", "class_fvg_gt",
                               "class_fvg_pred", "class_fvg_error") if c in df]
    summary = df.groupby("model")[metric_cols].mean().loc[["base", "fine-tuned"]]
    summary.to_csv(out / "summary.csv")
    plot_summary(df, out / "summary.png")

    print(f"\nMean over {len(rows)} windows ({args.split}):")
    print(summary.round(4).to_string())
    print(f"\nFigures ({len(vis_names)}), summary.png, metrics.csv, summary.csv → {out}")


if __name__ == "__main__":
    main()

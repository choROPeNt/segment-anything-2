"""
Sweep the automatic mask generator (AMG) settings of scripts/analyse.py on labelled windows.

The model runs once per window and *generation* setting (checkpoint, --points-per-side, --multimask,
--use-m2m, --upscale-to) with the loosest filters of the sweep; every combination of the *filter* settings
(--pred-iou × --stability × --nms × --min-area) is then evaluated offline on those candidates, so a grid of
hundreds of filter settings costs about as much as one AMG run (see sam2.addons.amg_tuning).

Scores per setting, pooled over all windows: precision / recall / F1 of the fibres matched at IoU ≥ 0.5,
mean IoU of the matches, and the fibre volume fraction error inside the ROI (bias and mean absolute
per-window error).  Predictions that lie mostly outside the ROI or on labelled fibres that are not part of
the window's ground truth (e.g. cut at the window border) are counted as "ignored", not as false positives.
The current analyse.py setting (--reference) is always evaluated for comparison.

    python scripts/sweep_amg.py --dataset data/fibre_ds_512_pad256_train-val --split val --max-windows 30 \\
        --ckpt base=checkpoints/sam2.1_hiera_tiny.pt \\
        --ckpt e60=<run>/checkpoints/checkpoint_060.pt \\
        --points-per-side 16 32

Writes to --out: summary.csv (one row per setting, sorted by F1), windows.csv (per window and setting),
pr.png (precision vs. recall of all settings), f1_heatmap.png (F1 over predicted-IoU × stability threshold
for the best setting), vis_<window>.png (current vs. best setting) and best.txt (the analyse.py options).
"""

import argparse
import itertools
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion

from sam2.addons import generate_upscaled
from sam2.addons.amg_tuning import (
    filter_candidates,
    raw_mask_generator,
    score_selection,
    summarize_counts,
    window_candidates,
)
from sam2.addons.evaluation import load_sa1b_window
from sam2.build_sam import build_sam2

# reference palette (dataviz skill): categorical slots in fixed order, status colours for TP / FP / missed
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, CRITICAL, WARNING, MUTED, INK, INK2 = "#0ca30c", "#d03b3b", "#fab219", "#898781", "#0b0b0b", "#52514e"
GEN = ["model", "points_per_side", "multimask", "use_m2m", "upscale_to"]
FILT = ["pred_iou_thresh", "stability_score_thresh", "box_nms_thresh", "min_area"]


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_h5(path, root):
    """Source h5 of a window; follows results/**/rename_log_*.csv when the file was renamed after the export."""
    p = Path(path)
    if (root / p).exists():
        return root / p
    for log in sorted((root / "results").glob("**/rename_log_*.csv")):
        m = pd.read_csv(log)
        hit = m.loc[m["old"] == str(p), "new"]
        if len(hit) and (root / hit.iloc[0]).exists():
            return root / hit.iloc[0]
    raise FileNotFoundError(f"source h5 '{path}' of the dataset not found (no rename log entry either)")


def window_regions(h5f, row, ids):
    """ROI, ignore region and true fibre pixels of one window, cropped from the source h5."""
    ys, xs = slice(row.y0, row.y0 + row.height), slice(row.x0, row.x0 + row.width)
    lab = h5f["label_map"][ys, xs]
    roi = h5f["mask"][ys, xs] > 0 if "mask" in h5f else np.ones(lab.shape, dtype=bool)
    unlabelled_gt = (lab > 0) & ~np.isin(lab, ids)          # labelled fibres that are not ground truth here
    return roi, ~roi | unlabelled_gt, lab > 0


def gen_label(r):
    s = f"{r['model']} · {r['points_per_side']} pts"
    if r["multimask"]:
        s += " · multimask"
    if r["use_m2m"]:
        s += " · m2m"
    if r["upscale_to"]:
        s += f" · ↑{r['upscale_to']}"
    return s


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)


def plot_pr(summary, ref, best, path):
    gens = summary[GEN].drop_duplicates().to_dict("records")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    style(ax)
    # iso-F1 curves (recessive)
    r = np.linspace(0.01, 1, 200)
    for f in (0.8, 0.85, 0.9, 0.95):
        p = f * r / (2 * r - f)
        ok = (p > 0) & (p <= 1)
        ax.plot(r[ok], p[ok], color="#d9d8d3", lw=0.8, zorder=0)
        ax.annotate(f"F1 {f}", (r[ok][-1], p[ok][-1]), fontsize=7, color=MUTED, ha="right", va="bottom")
    handles = []
    for k, g in enumerate(gens):
        sel = np.logical_and.reduce([summary[c] == g[c] for c in GEN])
        col = SERIES[k % len(SERIES)] if k < len(SERIES) else MUTED
        ax.scatter(summary.loc[sel, "recall"], summary.loc[sel, "precision"], s=9, color=col, alpha=0.55,
                   lw=0, zorder=2)
        handles.append(Line2D([], [], marker="o", ls="", color=col, label=gen_label(g)))
    ax.scatter(ref["recall"], ref["precision"], marker="X", s=70, color=INK, edgecolor="white", lw=1, zorder=4)
    ax.scatter([best["recall"]], [best["precision"]], s=110, facecolor="none", edgecolor=INK, lw=1.6, zorder=4)
    ax.annotate("best F1", (best["recall"], best["precision"]), xytext=(8, -12), textcoords="offset points",
                fontsize=8, color=INK)
    handles += [Line2D([], [], marker="X", ls="", color=INK, label="current analyse.py setting"),
                Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="best F1")]
    # zoom on the better half of the settings (always including the current one)
    lo = min(summary["recall"].quantile(0.5), summary["precision"].quantile(0.5), ref["recall"].min(),
             ref["precision"].min())
    ax.set_xlim(max(lo - 0.02, 0), 1.0)
    ax.set_ylim(max(lo - 0.02, 0), 1.0)
    ax.set_xlabel("recall (fibres found)", color=INK2, fontsize=9)
    ax.set_ylabel("precision (predictions that are fibres)", color=INK2, fontsize=9)
    ax.set_title("AMG settings: precision vs. recall", color=INK, fontsize=10, loc="left")
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(summary, best, path):
    sel = np.logical_and.reduce([summary[c] == best[c] for c in GEN + ["box_nms_thresh", "min_area"]])
    piv = summary[sel].pivot_table(index="stability_score_thresh", columns="pred_iou_thresh", values="f1")
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * piv.shape[1], 1.4 + 0.45 * piv.shape[0]))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns], fontsize=8, color=INK2)
    ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index], fontsize=8, color=INK2)
    bi = list(piv.columns).index(best["pred_iou_thresh"]) if best["pred_iou_thresh"] in piv.columns else None
    bj = list(piv.index).index(best["stability_score_thresh"]) if best["stability_score_thresh"] in piv.index else None
    if bi is not None and bj is not None:
        ax.add_patch(plt.Rectangle((bi - 0.5, bj - 0.5), 1, 1, fill=False, ec="white", lw=2))
        ax.annotate(f"{best['f1']:.3f}", (bi, bj), ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xlabel("pred_iou_thresh", color=INK2, fontsize=9)
    ax.set_ylabel("stability_score_thresh", color=INK2, fontsize=9)
    ax.set_title(f"F1 — {gen_label(best)}, NMS {best['box_nms_thresh']:g}, min_area {best['min_area']:g}",
                 color=INK, fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def outline(m):
    return m & ~binary_erosion(m, iterations=2)


def plot_window(image, gt, c, sets, path, name):
    """Image with predictions coloured by outcome (TP / FP / ignored) and missed fibres outlined."""
    H, W = c["shape"]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.9), squeeze=False)
    for ax, (title, keep) in zip(axes[0], sets):
        s = score_selection(c, keep)
        in_keep = np.zeros(len(c["piou"]), dtype=bool)
        in_keep[keep] = True
        matched_p, matched_g = set(), set()
        for p, g in c["pairs"]:
            if in_keep[p] and p not in matched_p and g not in matched_g:
                matched_p.add(p); matched_g.add(g)
        rgba = np.zeros((H * W, 4))
        for k in keep:
            col = GOOD if k in matched_p else (MUTED if c["ignore_frac"][k] >= 0.5 else CRITICAL)
            rgba[c["pix"][k]] = (*matplotlib.colors.to_rgb(col), 0.45)
        ax.imshow(image)
        ax.imshow(rgba.reshape(H, W, 4))
        miss = np.zeros((H, W), dtype=bool)
        for g in range(len(gt)):
            if g not in matched_g:
                miss |= outline(gt[g])
        over = np.zeros((H, W, 4))
        over[miss] = (*matplotlib.colors.to_rgb(WARNING), 1.0)
        ax.imshow(over)
        ax.set_title(f"{title}\n{s['tp']} found · {s['fp']} false · {s['fn']} missed · {s['ignored']} ignored",
                     fontsize=8.5, color=INK, loc="left")
        ax.axis("off")
    fig.legend(handles=[Patch(color=GOOD, label="matched fibre"), Patch(color=CRITICAL, label="false positive"),
                        Patch(color=MUTED, label="ignored (outside ROI / unlabelled)"),
                        Patch(color=WARNING, label="missed fibre (outline)")],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(name, fontsize=9, color=INK2, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="dataset root (images/, annotations/, windows.csv)")
    ap.add_argument("--split", default="val", help="split to evaluate: <dataset>/<split>.txt (default: val)")
    ap.add_argument("--max-windows", type=int, default=30, help="random sample of N windows (default: 30, 0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", action="append", default=None, metavar="NAME=PATH",
                    help="checkpoint to sweep, repeatable (default: base=checkpoints/sam2.1_hiera_tiny.pt)")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml", help="model config (all checkpoints)")
    # generation settings (one model run each)
    ap.add_argument("--points-per-side", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--multimask", type=int, nargs="+", default=[0], choices=[0, 1],
                    help="0 = one mask per point (analyse.py), 1 = three candidates per point")
    ap.add_argument("--use-m2m", type=int, nargs="+", default=[0], choices=[0, 1],
                    help="1 = one mask-to-mask refinement step per candidate")
    ap.add_argument("--upscale-to", type=int, nargs="+", default=[0], help="patch upscaling as analyse.py (0 = off)")
    ap.add_argument("--min-mask-region-area", type=int, default=150, help="hole / island removal (fixed)")
    # filter settings (evaluated offline)
    ap.add_argument("--pred-iou", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95])
    ap.add_argument("--stability", type=float, nargs="+", default=[0.8, 0.85, 0.9, 0.92, 0.95])
    ap.add_argument("--nms", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7])
    ap.add_argument("--min-area", type=float, nargs="+", default=[0, 500, 1000, 1500])
    ap.add_argument("--reference", default="0.1,0.9,0.1,0",
                    help="current analyse.py setting pred_iou,stability,nms,min_area (default: its defaults)")
    ap.add_argument("--exact-nms", action="store_true",
                    help="generate without NMS so every NMS value is exact (more memory and time per window)")
    ap.add_argument("--rank-by", default="f1", choices=["f1", "fvg_mae"], help="what 'best' means (default: f1)")
    ap.add_argument("--n-vis", type=int, default=4, help="windows to plot current vs. best (default: 4)")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--out", type=Path, default=Path("results/amg_sweep"))
    args = ap.parse_args()

    root = Path.cwd()
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpts = [s.split("=", 1) if "=" in s else (Path(s).stem, s)
             for s in (args.ckpt or ["base=checkpoints/sam2.1_hiera_tiny.pt"])]

    ref = tuple(float(v) for v in args.reference.split(","))
    combos = sorted(set(itertools.product(args.pred_iou, args.stability, args.nms, args.min_area)) | {ref})
    gens = list(itertools.product(args.points_per_side, args.multimask, args.use_m2m, args.upscale_to))
    floors = dict(pred_iou_floor=min(c[0] for c in combos), stability_floor=min(c[1] for c in combos),
                  nms_ceiling=1.0 if args.exact_nms else max(c[2] for c in combos))

    manifest = pd.read_csv(args.dataset / "windows.csv")
    names = (args.dataset / f"{args.split}.txt").read_text().split()
    rows = manifest.set_index("name").loc[names].reset_index()
    if args.max_windows:
        rows = rows.sample(min(args.max_windows, len(rows)), random_state=args.seed).reset_index(drop=True)
    vis_names = list(rows["name"][: args.n_vis])
    h5s = {p: h5py.File(resolve_h5(p, root), "r") for p in rows["h5"].unique()}

    models = {}
    for name, path in ckpts:
        models[name] = build_sam2(args.config, str(path), device=device)
    print(f"{len(rows)} windows of '{args.split}' · {len(models)} checkpoint(s) × {len(gens)} generation setting(s) "
          f"× {len(combos)} filter settings · device {device}")
    print(f"generation filters: pred_iou > {floors['pred_iou_floor']}, stability ≥ {floors['stability_floor']}, "
          f"NMS {floors['nms_ceiling']}")

    records, vis = [], {}
    t0 = time.time()
    for w, r in enumerate(rows.itertuples()):
        image, gt, ids = load_sa1b_window(args.dataset, r.name)
        roi, ignore, fibre = window_regions(h5s[r.h5], r, ids)
        for (mname, sam), (pps, mm, m2m, up) in itertools.product(models.items(), gens):
            amg = raw_mask_generator(sam, points_per_side=pps, multimask_output=bool(mm), use_m2m=bool(m2m),
                                     min_mask_region_area=args.min_mask_region_area, **floors)
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        anns = generate_upscaled(amg, image, up)
                else:
                    anns = generate_upscaled(amg, image, up)
            amg.predictor.reset_predictor()
            c = window_candidates(anns, gt, roi=roi, ignore=ignore, fibre=fibre) if anns else None
            gen = dict(model=mname, points_per_side=pps, multimask=mm, use_m2m=m2m, upscale_to=up)
            for pi, st, nms, amin in combos:
                if c is None:
                    s = {"tp": 0, "fp": 0, "fn": len(gt), "ignored": 0, "n_pred": 0, "iou_sum": 0.0,
                         "pred_px": 0, "gt_px": int((fibre & roi).sum()), "roi_px": int(roi.sum())}
                else:
                    s = score_selection(c, filter_candidates(c, pi, st, nms, amin))
                records.append({**gen, "pred_iou_thresh": pi, "stability_score_thresh": st,
                                "box_nms_thresh": nms, "min_area": amin, "window": r.name, **s})
            if r.name in vis_names and c is not None:
                vis[(r.name,) + tuple(gen.values())] = (image, gt, c)
            del anns
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"  [{w + 1}/{len(rows)}] {r.name}  {el:.0f} s, ~{el / (w + 1) * (len(rows) - w - 1):.0f} s left",
              end="\r", flush=True)
    print()

    df = pd.DataFrame(records)
    df.to_csv(args.out / "windows.csv", index=False)
    summary = summarize_counts(df, GEN + FILT)
    # ties (common on few windows) are broken by the V_f error, then by the IoU of the matched fibres
    keys = [args.rank_by] + [k for k in ("f1", "fvg_mae", "matched_iou") if k != args.rank_by]
    asc = [k == "fvg_mae" for k in keys]
    summary = summary.sort_values(keys, ascending=asc).reset_index(drop=True)
    summary.to_csv(args.out / "summary.csv", index=False)

    is_ref = np.logical_and.reduce([np.isclose(summary[c], v) for c, v in zip(FILT, ref)])
    refs = summary[is_ref]
    best = summary.iloc[0]
    cols = GEN + FILT + ["precision", "recall", "f1", "matched_iou", "fvg_bias", "fvg_mae", "fp", "fn"]
    pd.set_option("display.width", 200)
    print(f"\nCurrent analyse.py setting {args.reference} (pred_iou, stability, nms, min_area):")
    print(refs[cols].round(4).to_string(index=False))
    print(f"\nTop 10 by {args.rank_by}:")
    print(summary[cols].head(10).round(4).to_string(index=False))
    for mname in models:
        b = summary[summary["model"] == mname].iloc[0]
        rr = refs[refs["model"] == mname].sort_values(keys, ascending=asc).iloc[0]
        print(f"\n{mname}: best {args.rank_by} {b[args.rank_by]:.4f} (current setting at best generation "
              f"{rr[args.rank_by]:.4f}) — precision {b['precision']:.3f}, recall {b['recall']:.3f}, "
              f"V_f bias {b['fvg_bias']:+.4f}")

    ck = dict(ckpts)[best["model"]]
    cmd = (f"python scripts/analyse.py --input <image> --checkpoint {ck} "
           f"--points_per_side {int(best['points_per_side'])} --pred_iou_thresh {best['pred_iou_thresh']:g} "
           f"--stability_score_thresh {best['stability_score_thresh']:g} --box_nms_thresh {best['box_nms_thresh']:g} "
           f"--min_mask_region_area {args.min_mask_region_area}"
           + (f" --min_area {best['min_area']:g}" if best["min_area"] else "")
           + (f" --upscale_to {int(best['upscale_to'])}" if best["upscale_to"] else ""))
    notes = []
    if best["multimask"] or best["use_m2m"]:
        notes.append("note: the best setting uses multimask / m2m, which analyse.py does not expose yet "
                     "(it runs multimask_output=False, use_m2m=False)")
    (args.out / "best.txt").write_text("\n".join([cmd] + notes) + "\n")
    print(f"\nanalyse.py with the best setting:\n  {cmd}")
    for n in notes:
        print(f"  {n}")

    plot_pr(summary, refs, best, args.out / "pr.png")
    plot_heatmap(summary, best, args.out / "f1_heatmap.png")
    best_gen = tuple(best[c] for c in GEN)
    for name in vis_names:
        key = (name,) + best_gen
        if key not in vis:
            continue
        image, gt, c = vis[key]
        keep_ref = filter_candidates(c, *ref)
        keep_best = filter_candidates(c, best["pred_iou_thresh"], best["stability_score_thresh"],
                                      best["box_nms_thresh"], best["min_area"])
        plot_window(image, gt, c, [(f"current setting {args.reference}", keep_ref),
                                   (f"best: {best['pred_iou_thresh']:g}, {best['stability_score_thresh']:g}, "
                                    f"{best['box_nms_thresh']:g}, {best['min_area']:g}", keep_best)],
                    args.out / f"vis_{name}.png", f"{name} — {gen_label(best)}")
    for f in h5s.values():
        f.close()
    print(f"\nwrote {args.out}/summary.csv, windows.csv, pr.png, f1_heatmap.png, best.txt, vis_*.png")


if __name__ == "__main__":
    main()

"""
SAM 2 on the composites fibre-detection benchmark (Yuksel et al. 2026, Compos. Part A 211:110150,
dataset doi:10.5281/zenodo.17186135, data/Benchmark_fiber-dataset).

detect   : fibre centres / radii with a SAM 2 checkpoint, on the expert subframe (with context) or
           the full micrograph -> <data>/sam2_results/<image>_<roi>_<tag>.csv
evaluate : expert-reference matching (paper sec. 2.3.1 / Table 3) on the subframe and, if full-image
           detections exist, the macroscale / microscale metrics next to the 11 participants
           -> <data>/sam2_results/eval_<tag>/

    python scripts/benchmark_fiber_dataset.py detect --image all --roi subframe
    python scripts/benchmark_fiber_dataset.py detect --image all --roi full
    python scripts/benchmark_fiber_dataset.py evaluate

Patches are enlarged so the fibres have the size of the fine-tune training data (50x ILK images,
0.116887 um/px, 512 px patches): scale = px_um(image) / 0.116887, patch = 512 / scale.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.addons import fiber_benchmark as fb  # noqa: E402

DATA = "data/Benchmark_fiber-dataset"
IMAGES = {"50X": 1 / 3.615, "20X_Plan": 1 / 1.446, "20X_ELWD": 1 / 1.446}      # um / px
TRAIN_PX_UM = 0.116887                                                       # fine-tune data (50x ILK)
TRAIN_PATCH = 512
BASE_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
FT_CHECKPOINT = ("sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_fibre_ds_512_pad256_train-val_diag_e60.yaml/"
                 "checkpoints/checkpoint_060.pt")


def load_image(name):
    im = tifffile.imread(f"{DATA}/data/UD_TC1225_{name}.tif")
    return np.ascontiguousarray(im[..., :3])


def locate_subframe(image, name):
    """Top-left (x, y) and (w, h) of the expert subframe crop in the full image (template match)."""
    t = cv2.imread(f"{DATA}/annotated_data/UD_TC1225_{name}_subframe_11_17.png", cv2.IMREAD_GRAYSCALE)
    from PIL import Image
    g = np.asarray(Image.fromarray(image).convert("L"))           # Rec. 601, as the crops
    res = cv2.matchTemplate(g, t, cv2.TM_SQDIFF)
    _, _, loc, _ = cv2.minMaxLoc(res)
    return loc, (t.shape[1], t.shape[0])


def out_dir():
    d = f"{DATA}/sam2_results"
    os.makedirs(d, exist_ok=True)
    return d


def tag_of(args):
    if args.tag:
        return args.tag
    return "base" if os.path.normpath(args.checkpoint) == os.path.normpath(BASE_CHECKPOINT) else "e60"


# ----------------------------------------------------------------------------------------------
def cmd_detect(args):
    import torch
    from tqdm import tqdm
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    sam2 = build_sam2(args.config, args.checkpoint, device=device, apply_postprocessing=True)
    amg = SAM2AutomaticMaskGenerator(
        model=sam2, points_per_side=args.points_per_side, points_per_batch=64,
        pred_iou_thresh=args.pred_iou_thresh, stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh, min_mask_region_area=args.min_mask_region_area,
        multimask_output=False)

    def generate(img):
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = amg.generate(img)
            else:
                out = amg.generate(img)
        amg.predictor.reset_predictor()
        return out

    tag = tag_of(args)
    names = list(IMAGES) if args.image == "all" else [args.image]
    for name in names:
        image = load_image(name)
        H, W = image.shape[:2]
        scale = args.scale or IMAGES[name] / TRAIN_PX_UM * args.scale_factor
        patch = int(round(TRAIN_PATCH / scale))
        d_px = 7.0 / IMAGES[name]
        overlap = args.overlap or int(np.ceil(2.5 * d_px))
        if args.roi == "subframe":
            (sx, sy), (sw, sh) = locate_subframe(image, name)
            m = int(np.ceil(3 * d_px))                                   # context around the subframe
            roi = (max(sy - m, 0), max(sx - m, 0), min(sy + sh + m, H), min(sx + sw + m, W))
        else:
            roi = None
        print(f"{name}: {W}x{H} px, scale {scale:.2f}, patch {patch} px, overlap {overlap} px, "
              f"roi {roi}, device {device}, checkpoint {args.checkpoint}")
        t0 = time.time()
        df = fb.detect_fibres(image, generate, scale=scale, patch=patch, overlap=overlap, roi=roi,
                              progress=lambda g: tqdm(g, desc=name, unit="patch"))
        n0 = len(df)
        df = fb.suppress_duplicates(df)
        print(f"{name}: {len(df)} fibres ({n0 - len(df)} duplicates removed) in {time.time() - t0:.0f} s")
        df.attrs = {}
        path = f"{out_dir()}/{name}_{args.roi}_{tag}.csv"
        df.to_csv(path, index=False)
        meta = {"checkpoint": os.path.abspath(args.checkpoint), "config": args.config, "scale": scale,
                "patch": patch, "overlap": overlap, "roi_yxyx": roi, "points_per_side": args.points_per_side,
                "pred_iou_thresh": args.pred_iou_thresh, "stability_score_thresh": args.stability_score_thresh,
                "box_nms_thresh": args.box_nms_thresh, "min_mask_region_area": args.min_mask_region_area,
                "runtime_s": time.time() - t0, "device": device}
        with open(path.replace(".csv", ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"saved {path}")


# ----------------------------------------------------------------------------------------------
def filter_fibres(df, name, args):
    """Plausibility filter: fibres are round-ish, nominal r = 3.5 um."""
    r_um = df[args.r_col] * IMAGES[name]
    ok = (r_um > args.r_min_um) & (r_um < args.r_max_um) & (df["b"] / df["a"] > args.min_axis_ratio)
    return df[ok].reset_index(drop=True)


def subframe_eval(name, det, args):
    image = load_image(name)
    (sx, sy), (sw, sh) = locate_subframe(image, name)
    A = [fb.read_annotator(f"{DATA}/annotated_data/UD_TC1225_{name}_subframe_11_17_annotator_{k}.csv")
         for k in range(3)]
    ref = fb.expert_consensus(A)
    d = det.copy()
    d["x"] -= sx
    d["y"] -= sy
    d = d[(d.x > -10) & (d.y > -10) & (d.x < sw + 10) & (d.y < sh + 10)].reset_index(drop=True)
    m, s = fb.match_reference(ref, d, r_col=args.r_col)
    # detections well inside the subframe that no consensus fibre claims
    um = d.iloc[s.pop("unmatched_det_idx")]
    rr = ref.r.mean()
    inner = um[(um.x > 2 * rr) & (um.y > 2 * rr) & (um.x < sw - 2 * rr) & (um.y < sh - 2 * rr)]
    s["extra_det_interior"] = int(len(inner))
    s["r_mean_det_um"] = float(m["r_det"].mean() * IMAGES[name])
    s["r_mean_ref_um"] = float(m["r"].mean() * IMAGES[name])
    s["p_error_mean_um"] = s["p_error_mean_px"] * IMAGES[name]
    return m, s, ref, d, (sx, sy, sw, sh)


def participants_subframe(name, ref, sub):
    """The paper's Table 3 recomputed for the 11 participants with the same matching."""
    sx, sy, sw, sh = sub
    P = pd.read_csv(f"{DATA}/benchmark_results/detected_fibers.csv")
    P = P[P.image_name == name]
    rows = []
    for p, g in P.groupby("participant"):
        g = g.assign(x=g.x - sx, y=g.y - sy)
        g = g[(g.x > -10) & (g.y > -10) & (g.x < sw + 10) & (g.y < sh + 10)].reset_index(drop=True)
        _, s = fb.match_reference(ref, g, r_col="radius")
        s.pop("unmatched_det_idx")
        s["participant"] = str(p)
        rows.append(s)
    return pd.DataFrame(rows)


def cmd_evaluate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = tag_of(args)
    od = f"{out_dir()}/eval_{tag}"
    os.makedirs(od, exist_ok=True)
    P = pd.read_csv(f"{DATA}/benchmark_results/detected_fibers.csv")

    sub_rows, part_rows = [], []
    for name in IMAGES:
        path = f"{out_dir()}/{name}_subframe_{tag}.csv"
        fpath = f"{out_dir()}/{name}_full_{tag}.csv"
        src = path if os.path.exists(path) else fpath if os.path.exists(fpath) else None
        if src is None:
            print(f"no detections for {name} ({tag})")
            continue
        det = filter_fibres(pd.read_csv(src), name, args)
        m, s, ref, d, sub = subframe_eval(name, det, args)
        s.update(image=name, participant=f"SAM2 {tag}")
        sub_rows.append(s)
        pr = participants_subframe(name, ref, sub)
        pr["image"] = name
        part_rows.append(pr)
        m.to_csv(f"{od}/{name}_subframe_matches.csv", index=False)

        # overlay of the subframe
        image = load_image(name)
        sx, sy, sw, sh = sub
        crop = image[sy:sy + sh, sx:sx + sw]
        fig, ax = plt.subplots(figsize=(10, 10 * sh / sw))
        ax.imshow(crop)
        for x, y, r in ref[["x", "y", "r"]].to_numpy():
            ax.add_patch(plt.Circle((x - 0.5, y - 0.5), r, fill=False, color="cyan", lw=0.8))
        for x, y, r in d[["x", "y", args.r_col]].to_numpy():
            ax.add_patch(plt.Circle((x - 0.5, y - 0.5), r, fill=False, color="magenta", lw=0.8, ls="--"))
        um = m[m.n_cand == 0]
        ax.plot(um.x - 0.5, um.y - 0.5, "yx", ms=8)
        ax.set_xlim(-0.5, sw - 0.5)
        ax.set_ylim(sh - 0.5, -0.5)
        ax.set_axis_off()
        ax.set_title(f"{name}: expert consensus (cyan), SAM 2 {tag} (magenta), unmatched (x)")
        fig.tight_layout()
        fig.savefig(f"{od}/{name}_subframe_overlay.png", dpi=200)
        plt.close(fig)

    sub_df = pd.concat([pd.DataFrame(sub_rows)] + part_rows, ignore_index=True)
    cols = ["image", "participant", "n_ref", "n_matched", "unmatched", "multi_matched", "extra_det_interior",
            "r_error_mean", "r_error_abs_mean", "p_error_mean_px", "p_error_median_px", "dx_mean_px", "dy_mean_px",
            "r_mean_det_um", "r_mean_ref_um"]
    sub_df = sub_df[[c for c in cols if c in sub_df]]
    sub_df.to_csv(f"{od}/subframe_reference.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(sub_df)

    # ---- full-image metrics -------------------------------------------------------------------
    macro, dist = [], {}
    for name, px in IMAGES.items():
        fpath = f"{out_dir()}/{name}_full_{tag}.csv"
        if not os.path.exists(fpath):
            continue
        H, W = tifffile.TiffFile(f"{DATA}/data/UD_TC1225_{name}.tif").pages[0].shape[:2]
        det = filter_fibres(pd.read_csv(fpath), name, args)
        sets = {f"SAM2 {tag}": (det.x.to_numpy(), det.y.to_numpy(), det[args.r_col].to_numpy())}
        for p, g in P[P.image_name == name].groupby("participant"):
            sets[str(p)] = (g.x.to_numpy(), g.y.to_numpy(), g.radius.to_numpy())
        for who, (x, y, r) in sets.items():
            s = fb.summary_metrics(x, y, r, (H, W), px, union=not args.no_union)
            lv = fb.local_vf(x, y, r, (H, W))
            nn = fb.first_neighbour_distance(x, y, r) * px
            s.update(image=name, participant=who, local_vf_mean=float(np.nanmean(lv)),
                     local_vf_sd=float(np.nanstd(lv)), fnd_mean_um=float(nn.mean()),
                     fnd_overlap_frac=float((nn < 0).mean()))
            macro.append(s)
            dist[(name, who)] = (r * px, lv[np.isfinite(lv)], nn)
            print(f"{name:9s} {who:10s} n={s['n']:6d} r={s['r_mean_um']:.3f} um  "
                  f"Vf(union)={s.get('vf_union', np.nan):.3f}  Vf(overlap)={s['vf_overlap']:.3f}")
    if macro:
        mdf = pd.DataFrame(macro)
        mdf.to_csv(f"{od}/full_image_metrics.csv", index=False)
        plot_full(mdf, dist, tag, od, plt)
    print(f"results in {od}")


def plot_full(mdf, dist, tag, od, plt):
    names = list(IMAGES)
    who = [f"SAM2 {tag}"] + [str(p) for p in range(1, 12)]
    metrics = [("vf_union" if "vf_union" in mdf else "vf_overlap", "Vf (union)"), ("n", "fibre count"),
               ("r_mean_um", "mean radius [um]")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    w = 0.27
    for ax, (col, lab) in zip(axes, metrics):
        for k, name in enumerate(names):
            v = [mdf[(mdf.image == name) & (mdf.participant == p)][col].squeeze() if
                 ((mdf.image == name) & (mdf.participant == p)).any() else np.nan for p in who]
            ax.bar(np.arange(len(who)) + (k - 1) * w, v, w, label=name)
        ax.set_xticks(np.arange(len(who)))
        ax.set_xticklabels(["SAM2"] + who[1:])
        ax.set_ylabel(lab)
        if col == "n":
            ax.set_ylim(mdf.n.min() * 0.95, mdf.n.max() * 1.02)
        elif col.startswith("vf"):
            ax.set_ylim(0.35, 0.7)
        else:
            ax.set_ylim(2.8, 3.9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{od}/macroscale.png", dpi=200)
    plt.close(fig)

    for idx, (lab, bins) in enumerate([("radius [um]", np.arange(2.0, 5.0, 0.05)),
                                       ("local Vf (Voronoi)", np.arange(0.2, 1.2, 0.025)),
                                       ("first-neighbour distance [um]", np.arange(-1.5, 4.0, 0.05))]):
        present = [n for n in names if any((n, p) in dist for p in who)]
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), sharey=True, squeeze=False)
        axes = axes[0]
        ymax = 0.0
        for ax, name in zip(axes, present):
            for p in who:
                if (name, p) not in dist:
                    continue
                v = dist[(name, p)][idx]
                sam = p.startswith("SAM2")
                h, _, _ = ax.hist(v, bins=bins, density=True, histtype="step", lw=2.2 if sam else 0.7,
                                  color="k" if sam else None, alpha=1 if sam else 0.6, label="SAM2" if sam else p)
                if sam:
                    ymax = max(ymax, h.max())
            ax.set_title(name)
            ax.set_xlabel(lab)
        if ymax > 0:                       # fixed-radius participants give spikes: scale to SAM 2
            axes[0].set_ylim(0, 1.6 * ymax)
        axes[0].set_ylabel("density")
        axes[-1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{od}/dist_{['radius', 'local_vf', 'fnd'][idx]}.png", dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--checkpoint", default=FT_CHECKPOINT)
        p.add_argument("--tag", default=None, help="result name tag (default: e60 / base)")

    d = sub.add_parser("detect")
    common(d)
    d.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    d.add_argument("--image", default="all", choices=["all", *IMAGES])
    d.add_argument("--roi", default="subframe", choices=["subframe", "full"])
    d.add_argument("--scale", type=float, default=0.0, help="patch enlargement (default: from pixel size)")
    d.add_argument("--scale_factor", type=float, default=1.0, help="multiplies the default scale")
    d.add_argument("--overlap", type=int, default=0, help="patch overlap in px (default 2.5 fibre diameters)")
    d.add_argument("--points_per_side", type=int, default=16)
    d.add_argument("--pred_iou_thresh", type=float, default=0.7,
                   help="AMG predicted-IoU filter (0.88 of the in-domain sweep misses ~2%% of the fibres here)")
    d.add_argument("--stability_score_thresh", type=float, default=0.8)
    d.add_argument("--box_nms_thresh", type=float, default=0.2)
    d.add_argument("--min_mask_region_area", type=int, default=150)

    e = sub.add_parser("evaluate")
    common(e)
    e.add_argument("--r_col", default="r_ell", choices=["r_ell", "r_eq"],
                   help="radius of a detection: moment ellipse (a+b)/2 (as the experts) or sqrt(A/pi)")
    e.add_argument("--r_min_um", type=float, default=2.0)
    e.add_argument("--r_max_um", type=float, default=5.5)
    e.add_argument("--min_axis_ratio", type=float, default=0.5)
    e.add_argument("--no_union", action="store_true", help="skip the rasterised union Vf (slow on 50X)")

    a = ap.parse_args()
    {"detect": cmd_detect, "evaluate": cmd_evaluate}[a.cmd](a)

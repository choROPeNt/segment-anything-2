#!/usr/bin/env python
"""
Per-panorama yarn statistics (fibre shape, Delaunay local FVC, window FVC) and their aggregation per type.

    # 1) per panorama → {type}/batch_XX/{PP}/stats/{specimen}.{fibres,triangles,edges,windows}.csv + .summary.json
    python scripts/yarn_stats.py run data_phd [--only 160_01_00 ...] [--force]

    # 1b) only redo the per-panorama figures ({specimen}.fig_*.png in the stats folder)
    python scripts/yarn_stats.py plot data_phd [--only ...]

    # 1c) ParaView export: stats/{specimen}.fvc.xdmf (+ .fvc.h5): image, triangle mesh, Delaunay edges
    python scripts/yarn_stats.py xdmf data_phd [--only ...] [--step 2] [--labels]

    # 2) pool all panoramas → tables + figures (e.g. into the thesis vault)
    python scripts/yarn_stats.py aggregate data_phd --out ~/obsidian/phd/01_Thesis/03_chapter/assets/yarn_stats

    # 3) fibre shape incl. Fourier harmonics compared across batches and types (needs `run` for both batches)
    python scripts/yarn_stats.py compare data_phd --out ~/obsidian/phd/01_Thesis/03_chapter/assets/yarn_stats/compare

    # 4) local FVC (Delaunay triangles, windows, gaps) compared across batches and types
    python scripts/yarn_stats.py compare-fvc data_phd --out ~/obsidian/phd/01_Thesis/03_chapter/assets/yarn_stats/fvc

Every connected component of /mask is one yarn.  Shape statistics use the complete fibres inside a yarn
(:func:`sam2.addons.yarn_stats.valid_fibres`).  Local FVC is reported both number-weighted (every triangle
counts once) and area-weighted (every µm² of yarn counts once); the area-weighted mean equals the pixel FVC
of the triangulated region.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam2.addons.specimen_store import find_specimens  # noqa: E402
from sam2.addons.yarn_stats import (  # noqa: E402
    load_panorama, yarn_regions, fibre_table, add_yarn_position, valid_fibres,
    delaunay_fvc, delaunay_edges, window_fvc, weighted_summary, export_fvc_xdmf, hull_outline,
)

N_HARM = 5
IOU_TARGET = 0.95
WINDOWS_UM = (5, 10, 20, 40, 80)
FIBRES_PER_ROVING = 3000
HULL_EDGE = 1.5          # concave hull: peel boundary triangles with an edge > HULL_EDGE × median Delaunay edge
TYPE_NAME = {"160": "I", "200": "II", "245": "III"}
HEX_PACKING = np.pi / (2 * np.sqrt(3))


def stats_dir(sp):
    return Path(sp["dir"]) / "stats"


# --------------------------------------------------------------------------------------------- per panorama

def run_one(sp, force=False, hull_edge=HULL_EDGE):
    out = stats_dir(sp)
    name = sp["specimen"]
    if (out / f"{name}.summary.json").exists() and not force:
        print(f"{name}: exists, skip (--force to redo)")
        return
    t0 = time.time()
    d = load_panorama(sp["h5"])
    px, labels = d["px"], d["labels"]
    yarn_lab, n_yarn = yarn_regions(d["mask"])
    fibre = labels > 0
    print(f"{name}: {labels.shape[1]}×{labels.shape[0]} px, {n_yarn} yarn(s)", flush=True)

    fib = fibre_table(labels, px, yarn_lab, n_harm=N_HARM, iou_target=IOU_TARGET,
                      progress=lambda n: print(f"  fibres {n}", flush=True))
    fib = add_yarn_position(fib, yarn_lab, px, n_yarn)
    fib["valid"] = valid_fibres(fib)

    tris, edges, index_map, off, hinfo = [], [], None, 0, []
    for y in range(1, n_yarn + 1):
        sel = (fib["yarn"] == y).to_numpy() & (fib["frac_in_yarn"] >= 0.5).to_numpy()
        f_y = fib[sel]
        xy = f_y[["x_um", "y_um"]].to_numpy() / px
        tdf, simp, index_map, inf = delaunay_fvc(xy, fibre, px, hull_edge=hull_edge, index_map=index_map,
                                                 id_offset=off)
        hinfo.append({"yarn": y, **inf, "max_edge_um": inf["max_edge_px"] * px})
        tdf["yarn"] = y
        ids = f_y["id"].to_numpy()
        tdf[["v0", "v1", "v2"]] = ids[simp]
        off = int(index_map.max()) + 1
        tris.append(tdf)
        e = delaunay_edges(simp, xy, f_y["d_eq_um"].to_numpy(), px)
        e["i"], e["j"] = f_y["id"].to_numpy()[e["i"]], f_y["id"].to_numpy()[e["j"]]
        e["yarn"] = y
        edges.append(e)
    tri = pd.concat(tris, ignore_index=True)
    # yarn region = concave hull of the fibre centres (union of the kept triangles, no holes)
    hull_lab = np.zeros(index_map.shape, np.int32)
    tid_yarn = np.zeros(int(index_map.max()) + 2, np.int32)
    tid_yarn[tri["tri"].to_numpy()] = tri["yarn"].to_numpy()
    covered = index_map >= 0
    hull_lab[covered] = tid_yarn[index_map[covered]]
    del index_map
    tri = add_yarn_position(tri, hull_lab, px, n_yarn)
    fib = add_yarn_position(fib, hull_lab, px, n_yarn)
    edg = pd.concat(edges, ignore_index=True)
    win = window_fvc(fibre, covered, px, WINDOWS_UM)

    v = fib[fib["valid"]]
    inner = tri[~tri["boundary"]]
    summ = {
        "specimen": name, "type": d["type"], "type_name": TYPE_NAME.get(d["type"], d["type"]),
        "pixel_size_um": px, "n_yarn": n_yarn,
        "yarn_area_mm2": float((yarn_lab > 0).sum() * px ** 2 / 1e6),
        "n_fibres": int(len(fib)), "n_fibres_in_yarn": int((fib["yarn"] > 0).sum()), "n_valid": int(len(v)),
        "n_rovings_est": float((fib["yarn"] > 0).sum() / FIBRES_PER_ROVING),
        "fvc_yarn_mask": float(fibre[yarn_lab > 0].mean()),
        "fvc_triangulated": float(fibre[covered].mean()),
        "hull_area_mm2": float(covered.sum() * px ** 2 / 1e6), "hull_edge_factor": hull_edge, "hull": hinfo,
        "n_triangles": int(len(tri)), "n_triangles_interior": int(len(inner)),
        "tri_number": weighted_summary(inner["fvc"]),
        "tri_area": weighted_summary(inner["fvc"], inner["area_um2"]),
        "tri_frac_above_hex": float((inner["fvc"] > HEX_PACKING).mean()),
        "tri_corr_area_fvc": float(np.corrcoef(inner["area_um2"], inner["fvc"])[0, 1]),
        "windows": {str(L): {"n": int(len(g)), **weighted_summary(g["fvc"])}
                    for L, g in win.groupby("L_um")},
        "gap_um": weighted_summary(edg["g_um"]), "spacing_um": weighted_summary(edg["l_um"]),
        "shape": {c: weighted_summary(v[c]) for c in ("d_eq_um", "area_um2", "aspect", "psi1_deg",
                                                      "solidity", "circularity", "r2", "r3")},
        "iou_mean": {h: float(v[f"iou{h}"].mean()) for h in range(1, N_HARM + 1)},
        "n_min_hist": {int(k): int(c) for k, c in v["n_min"].value_counts().sort_index().items()},
        "iou_target": IOU_TARGET, "n_harm": N_HARM, "source": str(sp["h5"]), "runtime_s": round(time.time() - t0, 1),
    }
    out.mkdir(exist_ok=True)
    fib.to_csv(out / f"{name}.fibres.csv", index=False, float_format="%.5g")
    tri.to_csv(out / f"{name}.triangles.csv", index=False, float_format="%.5g")
    edg.to_csv(out / f"{name}.edges.csv", index=False, float_format="%.5g")
    win.to_csv(out / f"{name}.windows.csv", index=False, float_format="%.5g")
    (out / f"{name}.summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"  {len(fib)} fibres ({len(v)} valid), {len(tri)} triangles ({sum(h['n_peeled'] for h in hinfo)} peeled, "
          f"edge > {hinfo[0]['max_edge_um']:.1f} µm), hull {summ['hull_area_mm2']:.3f} vs mask {summ['yarn_area_mm2']:.3f} mm², "
          f"φ mask {summ['fvc_yarn_mask']:.3f} hull {summ['fvc_triangulated']:.3f} area-w. {summ['tri_area']['mean']:.3f} "
          f"number-w. {summ['tri_number']['mean']:.3f}, {summ['runtime_s']} s", flush=True)
    plot_one(sp)
    xdmf_one(sp)


# -------------------------------------------------------------------------------------------------- plots

CMAP = "plasma"
# type colours sampled from plasma (dark violet, magenta, orange; the yellow end is too light on white)
# thesis palette: plasma sampled at 10 steps (tex/user-colormap.tex), type1/2/3 = plasma01/04/07
PLASMA10 = ("#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26",
            "#f0f921")
COL = {"I": PLASMA10[1], "II": PLASMA10[4], "III": PLASMA10[7]}
ARROW = " →"
INK, GRID = "#52514e", "#e4e3df"


def unit_ticks(ax, axis, ticks, unit="−", fmt="{:g}"):
    """
    Thesis axis convention: explicit ticks, the second-to-last tick label is replaced by the unit
    ("−" for dimensionless axes), no unit in the axis label.
    """
    ticks = list(ticks)
    labels = [fmt.format(t) for t in ticks]
    if len(labels) >= 2:
        labels[-2] = unit
    if axis == "x":
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.minorticks_off()


def nice_ticks(lo, hi, n=5):
    """About *n* round tick positions inside [lo, hi]."""
    from matplotlib.ticker import MaxNLocator
    t = MaxNLocator(n).tick_values(lo, hi)
    return [x for x in t if lo - 1e-12 <= x <= hi + 1e-12]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b",
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
                         "savefig.bbox": "tight", "figure.facecolor": "white"})
    return plt


def make_figures(groups, colors, out, prefix="fig_"):
    """
    Figures of fibre shape, FVC weighting, profiles, window FVC and gaps for one or more groups.

    groups : {label: {"V": valid fibres, "Ti": interior triangles, "E": edges, "W": windows}}
    Writes {prefix}shape_kde.png, {prefix}fvc_weighting.png, {prefix}fvc_profiles.png, {prefix}window_gap.png.
    """
    from scipy.stats import gaussian_kde
    plt = _plt()
    out = Path(out)
    one = len(groups) == 1

    # ---------------------------------------------------------------- fibre shape
    fig, ax = plt.subplots(1, 4, figsize=(13, 2.8))
    specs = (("d_eq_um", r"$d_\mathrm{eq}$ [µm]", (5.0, 9.0)), ("aspect", r"$\varsigma_1/\varsigma_2$", (1.0, 1.8)),
             ("psi1_deg", r"$\psi_1$ [°]", (-90, 90)))
    for a, (c, lab, lim) in zip(ax, specs):
        xs = np.linspace(*lim, 300)
        for k, g in groups.items():
            x = g["V"][c].to_numpy()
            if one:
                a.hist(x, bins=np.linspace(*lim, 61), density=True, color=colors[k], alpha=0.2, lw=0)
            a.plot(xs, gaussian_kde(x)(xs), color=colors[k], label=k)
        a.set_xlabel(lab); a.set_ylabel("density"); a.set_xlim(*lim)
    if not one:
        ax[0].legend()
    a = ax[3]
    Hs = np.arange(1, N_HARM + 1)
    for k, g in groups.items():
        q = np.array([np.percentile(g["V"][f"iou{h}"], [5, 50, 95]) for h in Hs])
        a.fill_between(Hs, q[:, 0], q[:, 2], color=colors[k], alpha=0.12, lw=0)
        a.plot(Hs, q[:, 1], "-o", ms=4, color=colors[k], label=k)
    a.axhline(IOU_TARGET, color=INK, lw=1, ls=":")
    a.set_xticks(Hs); a.set_xlabel("harmonics $H$"); a.set_ylabel(r"$\mathrm{IoU}_H$ (median, 5–95 %)")
    if one:
        fig.suptitle(f"{next(iter(groups))}: fibre shape ({len(next(iter(groups.values()))['V'])} fibres)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"{prefix}shape_kde.png"); plt.close(fig)

    # ---------------------------------------------------------------- number vs area weighting
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.0), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    xs = np.linspace(0.2, 1.0, 400)
    for k, g in groups.items():
        ti = g["Ti"]
        mA, mN = np.average(ti["fvc"], weights=ti["area_um2"]), ti["fvc"].mean()
        ax[0].plot(xs, gaussian_kde(ti["fvc"], weights=ti["area_um2"])(xs), color=colors[k],
                   label=f"{k}, area-weighted (mean {mA:.3f})")
        ax[0].plot(xs, gaussian_kde(ti["fvc"])(xs), color=colors[k], ls="--", lw=1.2,
                   label=f"{k}, number-weighted (mean {mN:.3f})")
    ax[0].axvline(HEX_PACKING, color=INK, lw=1, ls=":")
    ax[0].text(HEX_PACKING, ax[0].get_ylim()[1] * 0.95, " hex.", color=INK, fontsize=8, va="top")
    ax[0].set_xlabel(r"local FVC $\phi^{(\mathrm{f})}_\mu(\mathcal{T}_k)$"); ax[0].set_ylabel("density")
    ax[0].legend(fontsize=7)
    k0 = next(iter(groups))
    ti = groups[k0]["Ti"]
    hb = ax[1].hexbin(ti["area_um2"], ti["fvc"], gridsize=60, bins="log", cmap=CMAP, mincnt=1,
                      extent=(0, np.percentile(ti["area_um2"], 99.5), 0.2, 1.0))
    r = np.corrcoef(ti["area_um2"], ti["fvc"])[0, 1]
    ax[1].set_xlabel(r"triangle area $|\mathcal{T}_k|$ [µm²]"); ax[1].set_ylabel("local FVC")
    ax[1].set_title(f"{k0}: corr = {r:.2f}", fontsize=9)
    fig.colorbar(hb, ax=ax[1], label="count", fraction=0.05)
    for k, g in groups.items():
        ti = g["Ti"]
        q = pd.qcut(ti["area_um2"], 20)
        gg = ti.groupby(q, observed=True)
        ax[2].plot(gg["area_um2"].sum().cumsum() / ti["area_um2"].sum(), gg["fvc"].mean(), "-o", ms=3,
                   color=colors[k], label=k)
    ax[2].set_xlabel("cumulative area share (triangles sorted by size)"); ax[2].set_ylabel("mean local FVC per size class")
    if not one:
        ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"{prefix}fvc_weighting.png"); plt.close(fig)

    # ---------------------------------------------------------------- profiles in the yarn
    fig, ax = plt.subplots(1, 3, figsize=(13, 2.9))
    for k, g in groups.items():
        ti = g["Ti"]
        for a, c, bins in ((ax[0], "t_rel", np.linspace(0, 1, 21)), (ax[1], "s_rel", np.linspace(0, 1, 21)),
                           (ax[2], "d_bnd_um", np.arange(0, 61, 3.0))):
            gg = ti.assign(fa=ti["fvc"] * ti["area_um2"]).groupby(pd.cut(ti[c], bins), observed=True)
            prof = gg["fa"].sum() / gg["area_um2"].sum()
            a.plot([iv.mid for iv in prof.index], prof.to_numpy(), "-", color=colors[k], label=k)
    ax[0].set_xlabel(r"through thickness $t$ (0 = upper edge)"); ax[1].set_xlabel(r"across width $s$")
    ax[2].set_xlabel("distance to yarn boundary [µm]")
    for a in ax:
        a.set_ylabel("area-weighted local FVC")
    if not one:
        ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"{prefix}fvc_profiles.png"); plt.close(fig)

    # ---------------------------------------------------------------- window size + gaps
    fig, ax = plt.subplots(1, 3, figsize=(13, 2.9))
    L = np.array(WINDOWS_UM, float)
    stds = []
    for k, g in groups.items():
        w = g["W"].groupby("L_um")["fvc"].agg(["mean", "std"]).reset_index()
        stds.append(w.set_index("L_um")["std"])
        ax[0].plot(w["L_um"], w["std"], "-o", ms=4, color=colors[k], label=k)
        ax[1].plot(w["L_um"], w["mean"], "-o", ms=4, color=colors[k])
        e = g["E"]["g_um"].to_numpy()
        xs = np.linspace(-1.5, 6, 300)
        ax[2].plot(xs, gaussian_kde(e[e < 10])(xs) * (e < 10).mean(), color=colors[k], label=k)
    ref = np.mean([s_.get(L[1], np.nan) for s_ in stds])
    ax[0].plot(L, ref * L[1] / L, color=INK, lw=1, ls=":", label=r"$\propto L^{-1}$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("window size $L$ [µm]"); ax[0].set_ylabel(r"std of $\phi^{(\mathrm{f})}_\mu(L)$"); ax[0].legend(fontsize=7)
    for a in ax[:2]:
        a.set_xscale("log"); a.set_xticks(L, [f"{x:g}" for x in L]); a.minorticks_off()
    ax[1].set_xlabel("window size $L$ [µm]"); ax[1].set_ylabel(r"mean of $\phi^{(\mathrm{f})}_\mu(L)$")
    ax[2].set_xlabel("gap between neighbours $g_{ij}$ [µm]"); ax[2].set_ylabel("density")
    if not one:
        ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"{prefix}window_gap.png"); plt.close(fig)


def fvc_map(sp, tri, path, zoom=(80.0, 40.0)):
    """Panorama with the triangles coloured by local FVC + zoom on the yarn centre."""
    import h5py
    from matplotlib.collections import PolyCollection
    plt = _plt()
    with h5py.File(sp["h5"], "r") as f:
        prev = f["image_preview"][()]
        fac = int(f["image_preview"].attrs.get("preview_factor", 2))
        px = float(f.attrs["pixel_size_x"])
    simp_xy = _triangles_xy(sp)
    clim = (0.4, HEX_PACKING)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 1.6]}, layout="constrained")
    ext = (0, prev.shape[1] * fac * px, prev.shape[0] * fac * px, 0)
    ax[0].imshow(prev[::4, ::4], cmap="gray", extent=ext)
    pc = PolyCollection(simp_xy, array=tri["fvc"].to_numpy(), cmap=CMAP, edgecolors="face", linewidths=0.05)
    pc.set_clim(*clim); ax[0].add_collection(pc)
    ax[0].set_title(f"{sp['specimen']}: local FVC of the Delaunay triangles", fontsize=9)
    ax[0].set_xlabel("x [µm]"); ax[0].set_ylabel("y [µm]"); ax[0].grid(False)
    cx, cy = tri["x_um"].median(), tri["y_um"].median()
    zw, zh = zoom
    r0, r1 = int((cy - zh / 2) / px / fac), int((cy + zh / 2) / px / fac)
    c0, c1 = int((cx - zw / 2) / px / fac), int((cx + zw / 2) / px / fac)
    ax[1].imshow(prev[r0:r1, c0:c1], cmap="gray", extent=(c0 * fac * px, c1 * fac * px, r1 * fac * px, r0 * fac * px))
    inw = ((np.abs(tri["x_um"] - cx) < zw / 2) & (np.abs(tri["y_um"] - cy) < zh / 2)).to_numpy()
    pz = PolyCollection(simp_xy[inw], array=tri["fvc"].to_numpy()[inw], cmap=CMAP, alpha=0.55,
                        edgecolors="#0b0b0b", linewidths=0.4)
    pz.set_clim(*clim); ax[1].add_collection(pz)
    ax[1].set_xlim(c0 * fac * px, c1 * fac * px); ax[1].set_ylim(r1 * fac * px, r0 * fac * px)
    ax[1].set_xlabel("x [µm]"); ax[1].set_ylabel("y [µm]"); ax[1].grid(False); ax[1].set_title("zoom, yarn centre", fontsize=9)
    ax[0].add_patch(plt.Rectangle((cx - zw / 2, cy - zh / 2), zw, zh, fill=False, ec="#21918c", lw=1))
    # concave hull of the fibre centres (region of the statistics) and the hand-drawn yarn mask for comparison
    from matplotlib.collections import LineCollection
    fib = pd.read_csv(stats_dir(sp) / f"{sp['specimen']}.fibres.csv").set_index("id")
    seg = hull_outline(tri, lambda ids: fib.loc[ids, ["x_um", "y_um"]].to_numpy())
    ax[0].add_collection(LineCollection(seg, colors="#0b0b0b", linewidths=0.6))
    with h5py.File(sp["h5"], "r") as f:
        m = f["mask_preview"][::4, ::4]
    ax[0].contour(np.linspace(0, ext[1], m.shape[1]), np.linspace(0, ext[2], m.shape[0]), m, [0.5],
                  colors="#21918c", linewidths=0.8, linestyles="--")
    ax[0].plot([], [], color="#0b0b0b", lw=0.8, label="concave hull of the fibre centres")
    ax[0].plot([], [], color="#21918c", lw=0.8, ls="--", label="yarn mask")
    ax[0].legend(fontsize=7, loc="lower right", frameon=True)
    fig.colorbar(pc, ax=ax, label=r"$\phi^{(\mathrm{f})}_\mu(\mathcal{T}_k)$", shrink=0.6)
    fig.savefig(path); plt.close(fig)


def plot_one(sp):
    """All figures of one panorama into its stats folder ({specimen}.fig_*.png)."""
    o, name = stats_dir(sp), sp["specimen"]
    s = json.loads((o / f"{name}.summary.json").read_text())
    fib = pd.read_csv(o / f"{name}.fibres.csv")
    tri = pd.read_csv(o / f"{name}.triangles.csv")
    g = {"V": fib[fib["valid"]], "Ti": tri[~tri["boundary"]], "E": pd.read_csv(o / f"{name}.edges.csv"),
         "W": pd.read_csv(o / f"{name}.windows.csv")}
    make_figures({name: g}, {name: COL.get(s["type_name"], COL["I"])}, o, prefix=f"{name}.fig_")
    fvc_map(sp, tri, o / f"{name}.fig_fvc_map.png")
    print(f"  plots → {o}/{name}.fig_*.png", flush=True)


def xdmf_one(sp, step=1, labels=False):
    """
    ParaView export of one panorama → stats/{specimen}.fvc.h5 + .fvc.xdmf with three grids:
    image (point data: image, fibre, yarn [, label]), triangles (cell data: every triangle column; point data:
    fibre id and shape) and edges (cell data: centre spacing and gap).
    """
    import h5py
    o, name = stats_dir(sp), sp["specimen"]
    fib = pd.read_csv(o / f"{name}.fibres.csv")
    tri = pd.read_csv(o / f"{name}.triangles.csv")
    edg = pd.read_csv(o / f"{name}.edges.csv")
    with h5py.File(sp["h5"], "r") as f:
        px = float(f.attrs["pixel_size_x"])
        lab = f["labeling/labels"][()]
        img = {"image": f["image"][()], "fibre": lab > 0, "yarn": f["mask"][()] > 0}
    if labels:
        img["label"] = lab
    del lab
    idx = pd.Series(np.arange(len(fib)), index=fib["id"])
    tcols = [c for c in tri.columns if c not in ("v0", "v1", "v2", "x_um", "y_um")]
    ncols = [c for c in ("id", "d_eq_um", "area_um2", "aspect", "psi1_deg", "s1_um", "s2_um", "iou1", "iou3",
                         "iou5", "n_min", "solidity", "d_bnd_um", "t_rel", "valid") if c in fib.columns]
    xp = export_fvc_xdmf(
        o / f"{name}.fvc", img, px, step=step,
        nodes=fib[["x_um", "y_um"]].to_numpy(),
        triangles=np.stack([idx[tri[v]].to_numpy() for v in ("v0", "v1", "v2")], 1),
        tri_fields={c: tri[c].to_numpy() for c in tcols},
        node_fields={("fibre_id" if c == "id" else c): fib[c].to_numpy() for c in ncols},
        edges=np.stack([idx[edg["i"]].to_numpy(), idx[edg["j"]].to_numpy()], 1),
        edge_fields={c: edg[c].to_numpy() for c in ("l_um", "g_um", "yarn")})
    mb = sum(p.stat().st_size for p in (xp, xp.parent / f"{name}.fvc.h5")) / 1e6
    print(f"  xdmf → {xp} ({mb:.0f} MB)", flush=True)


# -------------------------------------------------------------------------------------------- aggregation

def load_all(root, only=None):
    S, F, T, E, W = [], [], [], [], []
    for sp in find_specimens(root):
        name = sp["specimen"]
        if only and name not in only:
            continue
        o = stats_dir(sp)
        if not (o / f"{name}.summary.json").exists():
            continue
        s = json.loads((o / f"{name}.summary.json").read_text())
        S.append(s)
        for lst, kind in ((F, "fibres"), (T, "triangles"), (E, "edges"), (W, "windows")):
            df = pd.read_csv(o / f"{name}.{kind}.csv")
            df["specimen"], df["type"] = name, s["type_name"]
            lst.append(df)
    cat = lambda L: pd.concat(L, ignore_index=True)  # noqa: E731
    return S, cat(F), cat(T), cat(E), cat(W)


def aggregate(root, out, only=None, map_specimen=None):
    S, F, T, E, W = load_all(root, only)
    if not S:
        sys.exit("no summaries found, run `yarn_stats.py run` first")
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    types = [t for t in ("I", "II", "III") if t in set(F["type"])]
    V = F[F["valid"]]
    Ti = T[~T["boundary"]]

    # ---------------------------------------------------------------- tables
    rows = []
    for s in S:
        rows.append({"specimen": s["specimen"], "type": s["type_name"], "rovings": round(s["n_rovings_est"], 2),
                     "fibres_valid": s["n_valid"], "yarn_mm2": s["yarn_area_mm2"],
                     "hull_mm2": s["hull_area_mm2"], "phi_hull": s["fvc_triangulated"],
                     "phi_mask": s["fvc_yarn_mask"], "phi_tri_area": s["tri_area"]["mean"],
                     "phi_tri_number": s["tri_number"]["mean"], "phi_tri_median_number": s["tri_number"]["median"],
                     "phi_tri_std_area": s["tri_area"]["std"], "d_eq_um": s["shape"]["d_eq_um"]["mean"],
                     "aspect": s["shape"]["aspect"]["mean"], "gap_median_um": s["gap_um"]["median"]})
    per_pan = pd.DataFrame(rows)
    per_pan.to_csv(out / "per_panorama.csv", index=False, float_format="%.4f")

    trows = []
    for t in types:
        v, ti, e = V[V["type"] == t], Ti[Ti["type"] == t], E[E["type"] == t]
        r = {"type": t, "panoramas": int((per_pan["type"] == t).sum()), "fibres": len(v)}
        for c in ("d_eq_um", "aspect", "psi1_deg", "solidity", "r2", "r3"):
            r[f"{c}_mean"], r[f"{c}_std"] = v[c].mean(), v[c].std()
        for h in range(1, N_HARM + 1):
            r[f"iou{h}"] = v[f"iou{h}"].mean()
        r["n_min_le3"] = (v["n_min"] <= 3).mean()
        a = weighted_summary(ti["fvc"], ti["area_um2"])
        n = weighted_summary(ti["fvc"])
        r.update({f"phi_area_{k}": x for k, x in a.items()})
        r.update({f"phi_number_{k}": x for k, x in n.items()})
        r["gap_median_um"] = e["g_um"].median()
        r["spacing_median_um"] = e["l_um"].median()
        trows.append(r)
    per_type = pd.DataFrame(trows)
    per_type.to_csv(out / "per_type.csv", index=False, float_format="%.4f")
    wv = (W.groupby(["type", "L_um"])["fvc"].agg(["mean", "std", "count"]).reset_index())
    wv.to_csv(out / "window_fvc.csv", index=False, float_format="%.5f")

    groups = {f"Type {t}": {"V": V[V["type"] == t], "Ti": Ti[Ti["type"] == t], "E": E[E["type"] == t],
                            "W": W[W["type"] == t]} for t in types}
    make_figures(groups, {f"Type {t}": COL[t] for t in types}, out)
    ms = map_specimen or S[0]["specimen"]
    sp = next(p for p in find_specimens(root) if p["specimen"] == ms)
    fvc_map(sp, T[T["specimen"] == ms], out / "fig_fvc_map.png")

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(per_pan.round(3).to_string(index=False))
        print(per_type.round(4).T.to_string())
        print(wv.round(4).to_string(index=False))
    print(f"→ {out}")


# descriptor columns of the fibre table compared across batches and types: (column, label)
SHAPE_COLS = (("d_eq_um", r"$d_\mathrm{eq}$"), ("aspect", r"$\varsigma_1/\varsigma_2$"),
              ("r2", r"$\varsigma_{2,1}/\varsigma_{1,1}$"), ("r3", r"$\varsigma_{3,1}/\varsigma_{1,1}$"),
              ("r4", r"$\varsigma_{4,1}/\varsigma_{1,1}$"), ("r5", r"$\varsigma_{5,1}/\varsigma_{1,1}$"),
              ("e2", r"$\varsigma_{2,2}/\varsigma_{1,1}$"), ("e3", r"$\varsigma_{3,2}/\varsigma_{1,1}$"),
              ("solidity", "solidity"), ("iou1", r"$\mathrm{IoU}_1$"), ("iou3", r"$\mathrm{IoU}_3$"))
SHAPE_UNIT = {"d_eq_um": "µm"}   # all other shape descriptors are dimensionless
EQ_MARGIN = 0.25         # TOST margin ε = EQ_MARGIN × fibre-to-fibre std of the descriptor


def load_fibres(root, batches=("batch_01", "batch_02")):
    """Valid fibres of all panoramas of *batches* with columns specimen, type, batch."""
    F = []
    for sp in find_specimens(root):
        name, o = sp["specimen"], stats_dir(sp)
        b = f"batch_{name.split('_')[1]}"
        if b not in batches or not (o / f"{name}.fibres.csv").exists():
            continue
        df = pd.read_csv(o / f"{name}.fibres.csv")
        df = df[df["valid"]].copy()
        df["specimen"], df["type"], df["batch"] = name, TYPE_NAME[name.split("_")[0]], b
        F.append(df)
    return pd.concat(F, ignore_index=True)


def compare(root, out):
    """
    Fibre shape (size, first-harmonic ellipse, higher Fourier harmonics, reconstruction quality) compared
    across batches and types. Panoramas are the unit of replication.

    Writes into *out*: cmp_per_panorama.csv, cmp_per_cell.csv, cmp_variance.csv, cmp_pairs.csv (effect size,
    W1, TOST), cmp_classifier.csv and fig_cmp_{spectrum,iou,kde,panorama_means,variance}.png.
    """
    from scipy.stats import gaussian_kde
    from sam2.addons.shape_compare import (nested_variance, effect_size, wasserstein_scaled, tost_welch,
                                           classifier_test)
    plt = _plt()
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    V = load_fibres(root)
    cols = [c for c, _ in SHAPE_COLS]
    lab = dict(SHAPE_COLS)
    types = [t for t in ("I", "II", "III") if t in set(V["type"])]
    batches = sorted(set(V["batch"]))
    sd = V[cols].std()
    print(f"{len(V)} valid fibres, {V['specimen'].nunique()} panoramas", flush=True)

    # ---------------------------------------------------------------- tables
    pan = V.groupby(["type", "batch", "specimen"])[cols].mean().reset_index()
    pan["fibres"] = V.groupby(["type", "batch", "specimen"]).size().to_numpy()
    pan["n_min_le3"] = V.groupby(["type", "batch", "specimen"])["n_min"].apply(lambda s: (s <= 3).mean()).to_numpy()
    pan.to_csv(out / "cmp_per_panorama.csv", index=False, float_format="%.5f")
    cell = V.groupby(["type", "batch"])[cols].agg(["mean", "std"])
    cell.columns = [f"{c}_{s}" for c, s in cell.columns]
    cell.insert(0, "panoramas", V.groupby(["type", "batch"])["specimen"].nunique())
    cell.insert(1, "fibres", V.groupby(["type", "batch"]).size())
    cell.reset_index().to_csv(out / "cmp_per_cell.csv", index=False, float_format="%.5f")

    var = []
    for c in cols:
        a = nested_variance(V, c, ["type", "batch", "specimen"])
        b = nested_variance(V, c, ["batch", "type", "specimen"])
        var.append({"descriptor": c, "type": a["type"], "batch_in_type": a["batch"], "batch": b["batch"],
                    "type_in_batch": b["type"], "panorama": a["specimen"], "fibre": a["within"]})
    var = pd.DataFrame(var)
    var.to_csv(out / "cmp_variance.csv", index=False, float_format="%.6f")

    pairs = []
    def _pair(kind, g, sel_a, sel_b, pa, pb):
        for c in cols:
            eps = EQ_MARGIN * sd[c]
            t = tost_welch(pa[c], pb[c], eps)
            pairs.append({"comparison": kind, "group": g, "descriptor": c,
                          "mean_a": V.loc[sel_a, c].mean(), "mean_b": V.loc[sel_b, c].mean(),
                          "d": effect_size(V.loc[sel_a, c], V.loc[sel_b, c], sd[c]),
                          "W1_scaled": wasserstein_scaled(V.loc[sel_a, c], V.loc[sel_b, c], sd[c]),
                          "eps": eps, **{f"tost_{k}": v for k, v in t.items()}})
    for t in types:                                   # batch_01 vs batch_02 within a type
        if len(batches) < 2:
            break
        sa, sb = (V["type"] == t) & (V["batch"] == batches[0]), (V["type"] == t) & (V["batch"] == batches[1])
        pa = pan[(pan["type"] == t) & (pan["batch"] == batches[0])]
        pb = pan[(pan["type"] == t) & (pan["batch"] == batches[1])]
        _pair("batch", f"{t}: {batches[0]} − {batches[1]}", sa, sb, pa, pb)
    for i, ta in enumerate(types):                    # types, both batches pooled
        for tb in types[i + 1:]:
            _pair("type", f"{ta} − {tb}", V["type"] == ta, V["type"] == tb, pan[pan["type"] == ta],
                  pan[pan["type"] == tb])
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / "cmp_pairs.csv", index=False, float_format="%.5f")

    # W1 reference: panorama pairs of the same cell vs other batch / other type
    w1 = []
    names = pan[["specimen", "type", "batch"]].to_numpy()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (s1, t1, b1), (s2, t2, b2) = names[i], names[j]
            rel = "same type+batch" if (t1 == t2 and b1 == b2) else "same type, other batch" if t1 == t2 else \
                "other type"
            x1, x2 = V[V["specimen"] == s1], V[V["specimen"] == s2]
            w1.append({"relation": rel, **{c: wasserstein_scaled(x1[c], x2[c], sd[c]) for c in cols}})
    w1 = pd.DataFrame(w1).groupby("relation")[cols].median()
    w1.to_csv(out / "cmp_w1_reference.csv", float_format="%.4f")

    X = V[cols].to_numpy()
    clf = []
    ba, ch = classifier_test(X, V["type"].to_numpy(), V["specimen"].to_numpy())
    clf.append({"task": "type from shape (both batches)", "split": "leave one panorama out", "ba": ba, "chance": ch})
    for t in types:
        s = (V["type"] == t).to_numpy()
        if V.loc[s, "batch"].nunique() > 1:
            ba, ch = classifier_test(X[s], V.loc[s, "batch"].to_numpy(), V.loc[s, "specimen"].to_numpy())
            clf.append({"task": f"batch from shape (type {t})", "split": "leave one panorama out", "ba": ba,
                        "chance": ch})
    ba, ch = classifier_test(X, V["batch"].to_numpy(), V["specimen"].to_numpy())
    clf.append({"task": "batch from shape (all types)", "split": "leave one panorama out", "ba": ba, "chance": ch})
    clf = pd.DataFrame(clf)
    clf.to_csv(out / "cmp_classifier.csv", index=False, float_format="%.3f")

    # ---------------------------------------------------------------- figures
    ls = {b: ("-" if k == 0 else "--") for k, b in enumerate(batches)}
    mk = {b: ("o" if k == 0 else "s") for k, b in enumerate(batches)}
    groups = [(t, b) for t in types for b in batches if ((V["type"] == t) & (V["batch"] == b)).any()]
    gl = lambda t, b: f"{t} / {b.replace('batch_', 'b')}"   # noqa: E731

    from matplotlib.lines import Line2D
    def legend(a, **kw):
        h = [Line2D([], [], color=COL[t], lw=2, label=f"type {t}") for t in types]
        h += [Line2D([], [], color=INK, ls=ls[b], marker=mk[b], ms=4, mfc=INK if k == 0 else "white",
                     label=b.replace("_", " ")) for k, b in enumerate(batches)]
        a.legend(handles=h, fontsize=7, **kw)

    def style_line(a, t, b):
        return dict(ls=ls[b], marker=mk[b], ms=4, color=COL[t], mfc=COL[t] if b == batches[0] else "white")

    # Fourier spectrum: mean semi-axes of harmonics n relative to the first
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ns = np.arange(2, N_HARM + 1)
    for t, b in groups:
        g = V[(V["type"] == t) & (V["batch"] == b)]
        for a, p in zip(ax, ("r", "e")):
            m = np.array([g[f"{p}{n}"].mean() for n in ns])
            a.plot(ns, m, **style_line(a, t, b))
    for a, p, yt in zip(ax, (r"$\varsigma_{n,1}/\varsigma_{1,1}$", r"$\varsigma_{n,2}/\varsigma_{1,1}$"),
                        ((0.01, 0.02, 0.03, 0.04, 0.05), (0.006, 0.01, 0.014, 0.018, 0.022))):
        unit_ticks(a, "x", ns); unit_ticks(a, "y", yt, fmt="{:.3g}")
        a.set_xlabel("harmonic $n$" + ARROW); a.set_ylabel(p + ARROW)
    ax[0].set_title("major semi-axis (mean)", fontsize=9); ax[1].set_title("minor semi-axis (mean)", fontsize=9)
    legend(ax[0], ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig_cmp_spectrum.png"); plt.close(fig)

    # reconstruction quality over H
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    Hs = np.arange(1, N_HARM + 1)
    for t, b in groups:
        g = V[(V["type"] == t) & (V["batch"] == b)]
        ax[0].plot(Hs, [g[f"iou{h}"].median() for h in Hs], **style_line(ax[0], t, b))
        ax[1].plot(Hs, [100 * (g["n_min"] <= h).mean() for h in Hs], **style_line(ax[1], t, b))
    ax[0].axhline(IOU_TARGET, color=PLASMA10[0], lw=1, ls=":")
    ax[0].set_ylabel(r"$\mathrm{IoU}_H$ (median)" + ARROW)
    ax[1].set_ylabel(rf"fibers with $\mathrm{{IoU}}_H\geq{IOU_TARGET}$" + ARROW)
    unit_ticks(ax[0], "y", (0.93, 0.94, 0.95, 0.96, 0.97, 0.98), fmt="{:.2f}")
    unit_ticks(ax[1], "y", (0, 20, 40, 60, 80, 100), unit="%")
    for a in ax:
        unit_ticks(a, "x", Hs); a.set_xlabel("harmonics $H$" + ARROW)
    legend(ax[0], ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig_cmp_iou.png"); plt.close(fig)

    # distributions (density axis without ticks, as the marginal KDEs in the thesis)
    kc = ("d_eq_um", "aspect", "r2", "r3", "solidity", "iou3")
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.4))
    for a, c in zip(ax.ravel(), kc):
        lo, hi = V[c].quantile([0.005, 0.995])
        xs = np.linspace(lo, hi, 300)
        for t, b in groups:
            x = V.loc[(V["type"] == t) & (V["batch"] == b), c].to_numpy()
            a.plot(xs, gaussian_kde(x)(xs), ls=ls[b], lw=1.5, color=COL[t])
        unit_ticks(a, "x", nice_ticks(lo, hi, 5), unit=SHAPE_UNIT.get(c, "−"), fmt="{:.4g}")
        a.set_xlim(lo, hi); a.set_yticks([]); a.set_ylim(bottom=0)
        a.set_xlabel(lab[c] + ARROW); a.set_ylabel("density" + ARROW)
    legend(ax[0, 0])
    fig.tight_layout(); fig.savefig(out / "fig_cmp_kde.png"); plt.close(fig)

    # panorama means
    pc = ("d_eq_um", "aspect", "r2", "r3", "solidity", "iou3")
    fig, ax = plt.subplots(1, len(pc), figsize=(13, 2.9))
    for a, c in zip(ax, pc):
        for k, (t, b) in enumerate(groups):
            p = pan[(pan["type"] == t) & (pan["batch"] == b)]
            xk = k + 0.4 * (k // len(batches))
            a.plot(np.full(len(p), xk), p[c], mk[b], ms=5, color=COL[t], mfc=COL[t] if b == batches[0] else "white")
            a.plot([xk - 0.25, xk + 0.25], [p[c].mean()] * 2, color=COL[t], lw=1.5)
        m = V[c].mean()
        a.axhspan(m - EQ_MARGIN * sd[c], m + EQ_MARGIN * sd[c], color=PLASMA10[8], alpha=0.15, lw=0)
        lo, hi = a.get_ylim()
        unit_ticks(a, "y", nice_ticks(lo, hi, 6), unit=SHAPE_UNIT.get(c, "−"), fmt="{:.4g}"); a.set_ylim(lo, hi)
        a.set_xticks([k + 0.4 * (k // len(batches)) for k in range(len(groups))])
        a.set_xticklabels([gl(t, b) for t, b in groups], rotation=60, fontsize=7); a.set_title(lab[c], fontsize=9)
    fig.suptitle(rf"panorama means (band: overall mean $\pm\,{EQ_MARGIN}\,s_X$)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_cmp_panorama_means.png"); plt.close(fig)

    # variance shares (between-group parts only, in % of total)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    xk = np.arange(len(var))
    parts = (("type", "type", PLASMA10[1]), ("batch_in_type", "batch in type", PLASMA10[4]),
             ("panorama", "panorama in batch", PLASMA10[7]))
    bottom = np.zeros(len(var))
    for c, l, colr in parts:
        ax.bar(xk, 100 * var[c], bottom=bottom, color=colr, width=0.6, label=l)
        bottom += 100 * var[c].to_numpy()
    ax.set_xticks(xk); ax.set_xticklabels([lab[c] for c in var["descriptor"]], rotation=30, fontsize=8)
    unit_ticks(ax, "y", nice_ticks(0, bottom.max() * 1.05, 6), unit="%")
    ax.set_ylabel("share of total variance" + ARROW); ax.legend(fontsize=8)
    ax.set_title("between-group variance (rest = fiber-to-fiber scatter)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_cmp_variance.png"); plt.close(fig)

    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(pan.round(4).to_string(index=False))
        print((100 * var.set_index("descriptor")).round(3).to_string())
        print(pairs[["comparison", "group", "descriptor", "mean_a", "mean_b", "d", "W1_scaled", "eps", "tost_delta",
                     "tost_ci_low", "tost_ci_high", "tost_p"]].round(4).to_string(index=False))
        print(w1.round(3).to_string())
        print(clf.round(3).to_string(index=False))
    print(f"→ {out}")


FVC_MARGIN = 0.02        # TOST margin for panorama FVC means (2 percentage points)


def load_fvc(root, batches=("batch_01", "batch_02")):
    """Summaries, interior triangles, Delaunay edges and windows of all panoramas of *batches*."""
    S, T, E, W = [], [], [], []
    for sp in find_specimens(root):
        name, o = sp["specimen"], stats_dir(sp)
        b = f"batch_{name.split('_')[1]}"
        if b not in batches or not (o / f"{name}.summary.json").exists():
            continue
        key = {"specimen": name, "type": TYPE_NAME[name.split("_")[0]], "batch": b}
        s = json.loads((o / f"{name}.summary.json").read_text())
        S.append({**key, "summary": s})
        t = pd.read_csv(o / f"{name}.triangles.csv")
        T.append(t[~t["boundary"]].assign(**key))
        E.append(pd.read_csv(o / f"{name}.edges.csv").assign(**key))
        W.append(pd.read_csv(o / f"{name}.windows.csv").assign(**key))
    cat = lambda L: pd.concat(L, ignore_index=True)  # noqa: E731
    return S, cat(T), cat(E), cat(W)


def compare_fvc(root, out):
    """
    Local FVC from the Delaunay triangulation (concave hull) compared across batches and types.
    Every triangle T_k carries phi(T_k) = |fibre ∩ T_k| / |T_k|, statistics are area-weighted (each µm² of
    yarn counts once) unless stated. Panoramas are the unit of replication.

    Writes fvc_per_panorama.csv, fvc_per_cell.csv, fvc_variance.csv, fvc_pairs.csv, fvc_windows.csv and
    fig_fvc_{kde,panorama,profiles,windows,gap,variance}.png into *out*.
    """
    from scipy.stats import gaussian_kde
    from sam2.addons.yarn_stats import weighted_quantile
    from sam2.addons.shape_compare import nested_variance, wasserstein_scaled, tost_welch
    plt = _plt()
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    S, T, E, W = load_fvc(root)
    types = [t for t in ("I", "II", "III") if t in set(T["type"])]
    batches = sorted(set(T["batch"]))
    groups = [(t, b) for t in types for b in batches if ((T["type"] == t) & (T["batch"] == b)).any()]
    cell = lambda df, t, b: df[(df["type"] == t) & (df["batch"] == b)]  # noqa: E731
    print(f"{len(T)} interior triangles, {len(S)} panoramas", flush=True)

    # ---------------------------------------------------------------- per panorama
    rows = []
    for s in S:
        sm = s["summary"]
        t = T[T["specimen"] == s["specimen"]]
        e = E[E["specimen"] == s["specimen"]]
        w = t["area_um2"].to_numpy()
        q05, q50, q95 = weighted_quantile(t["fvc"].to_numpy(), w, [0.05, 0.5, 0.95])
        rows.append({"specimen": s["specimen"], "type": s["type"], "batch": s["batch"],
                     "rovings": round(sm["n_rovings_est"], 2), "fibres": sm["n_valid"],
                     "hull_mm2": sm["hull_area_mm2"], "phi_mask": sm["fvc_yarn_mask"],
                     "phi_hull": sm["fvc_triangulated"], "phi_A": np.average(t["fvc"], weights=w),
                     "phi_N": t["fvc"].mean(), "median_A": q50, "q05_A": q05, "q95_A": q95,
                     "std_A": np.sqrt(np.average((t["fvc"] - np.average(t["fvc"], weights=w)) ** 2, weights=w)),
                     "corr_area_fvc": np.corrcoef(t["area_um2"], t["fvc"])[0, 1],
                     "frac_above_hex": np.average(t["fvc"] > HEX_PACKING, weights=w),
                     "tri_area_median_um2": t["area_um2"].median(), "gap_median_um": e["g_um"].median(),
                     "spacing_median_um": e["l_um"].median()})
    pan = pd.DataFrame(rows)
    pan.to_csv(out / "fvc_per_panorama.csv", index=False, float_format="%.4f")
    num = [c for c in pan.columns if c not in ("specimen", "type", "batch")]
    pc = pan.groupby(["type", "batch"])[num].agg(["mean", "min", "max"])
    pc.columns = [f"{c}_{k}" for c, k in pc.columns]
    pc.reset_index().to_csv(out / "fvc_per_cell.csv", index=False, float_format="%.4f")

    # ---------------------------------------------------------------- variance split, pairs, W1
    W20 = W[W["L_um"] == 20]
    var = []
    for name, df, val, wcol in ((r"$\phi(\mathcal{T}_k)$, area-w.", T, "fvc", "area_um2"),
                                (r"$\phi(\mathcal{T}_k)$, number-w.", T, "fvc", None),
                                (r"$\phi(L{=}20$ µm$)$", W20, "fvc", None),
                                (r"gap $g_{ij}$", E[E["g_um"] < 10], "g_um", None)):
        a = nested_variance(df, val, ["type", "batch", "specimen"], wcol)
        var.append({"quantity": name, "scope": "both batches", "type": a["type"], "batch_in_type": a["batch"],
                    "panorama": a["specimen"], "within": a["within"]})
    for b in batches:                                  # type effect within one batch
        for name, df, wcol in ((r"$\phi(\mathcal{T}_k)$, area-w.", T, "area_um2"), (r"$\phi(L{=}20$ µm$)$", W20, None)):
            d = df[df["batch"] == b]
            a = nested_variance(d, "fvc", ["type", "specimen"], wcol)
            var.append({"quantity": name, "scope": b.replace("_", " "), "type": a["type"], "batch_in_type": 0.0,
                        "panorama": a["specimen"], "within": a["within"]})
    var = pd.DataFrame(var)
    var.to_csv(out / "fvc_variance.csv", index=False, float_format="%.5f")

    sdA = np.sqrt(np.average((T["fvc"] - np.average(T["fvc"], weights=T["area_um2"])) ** 2, weights=T["area_um2"]))
    pairs = []
    def _pair(kind, g, pa, pb, ta, tb):
        r = tost_welch(pa["phi_A"], pb["phi_A"], FVC_MARGIN)
        pairs.append({"comparison": kind, "group": g, "phi_A_a": pa["phi_A"].mean(), "phi_A_b": pb["phi_A"].mean(),
                      "W1_A_scaled": wasserstein_scaled(ta["fvc"], tb["fvc"], sdA, ta["area_um2"], tb["area_um2"]),
                      "eps": FVC_MARGIN, **{f"tost_{k}": v for k, v in r.items()}})
    for t in types:
        if len(batches) > 1:
            _pair("batch", f"{t}: {batches[0]} − {batches[1]}", cell(pan, t, batches[0]), cell(pan, t, batches[1]),
                  cell(T, t, batches[0]), cell(T, t, batches[1]))
    for b in batches:
        for i, ta in enumerate(types):
            for tb in types[i + 1:]:
                _pair(f"type ({b})", f"{ta} − {tb}", cell(pan, ta, b), cell(pan, tb, b), cell(T, ta, b), cell(T, tb, b))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / "fvc_pairs.csv", index=False, float_format="%.4f")

    # window FVC: std(L) and integral range A = Var(L) L² (stationary limit), slope of log std vs log L
    wv = W.groupby(["type", "batch", "L_um"])["fvc"].agg(["mean", "std", "count"]).reset_index()
    wv["A_um2"] = wv["std"] ** 2 * wv["L_um"] ** 2
    slope = (wv.groupby(["type", "batch"])
             .apply(lambda g: np.polyfit(np.log(g["L_um"]), np.log(g["std"]), 1)[0], include_groups=False)
             .rename("slope_log_std").reset_index())
    wv = wv.merge(slope, on=["type", "batch"])
    wv.to_csv(out / "fvc_windows.csv", index=False, float_format="%.5f")

    # ---------------------------------------------------------------- figures (thesis style)
    from matplotlib.lines import Line2D
    ls = {b: ("-" if k == 0 else "--") for k, b in enumerate(batches)}
    mk = {b: ("o" if k == 0 else "s") for k, b in enumerate(batches)}
    gl = lambda t, b: f"{t} / {b.replace('batch_', 'b')}"   # noqa: E731
    def legend(a, **kw):
        h = [Line2D([], [], color=COL[t], lw=2, label=f"type {t}") for t in types]
        h += [Line2D([], [], color=INK, ls=ls[b], marker=mk[b], ms=4, mfc=INK if k == 0 else "white",
                     label=b.replace("_", " ")) for k, b in enumerate(batches)]
        a.legend(handles=h, fontsize=7, **kw)
    def sty(t, b):
        return dict(ls=ls[b], marker=mk[b], ms=4, color=COL[t], mfc=COL[t] if b == batches[0] else "white")
    phi = r"$\phi^{(\mathrm{f})}_\mu$"

    # 1) area-weighted density and CDF of phi(T)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    xs = np.linspace(0.2, 0.95, 300)
    for t, b in groups:
        c = cell(T, t, b)
        ax[0].plot(xs, gaussian_kde(c["fvc"], weights=c["area_um2"], bw_method=0.08)(xs), ls=ls[b], lw=1.6, color=COL[t])
        o = np.argsort(c["fvc"].to_numpy())
        cw = np.cumsum(c["area_um2"].to_numpy()[o]); cw /= cw[-1]
        ax[1].plot(c["fvc"].to_numpy()[o], 100 * cw, ls=ls[b], lw=1.6, color=COL[t])
    for a in ax:
        a.axvline(HEX_PACKING, color=PLASMA10[0], lw=1, ls=":")
        unit_ticks(a, "x", (0.2, 0.4, 0.6, 0.8, 1.0), fmt="{:.1f}"); a.set_xlim(0.2, 1.0)
        a.set_xlabel(phi + r"$(\mathcal{T}_k)$" + ARROW)
    ax[0].set_yticks([]); ax[0].set_ylabel(r"$p_A$" + ARROW); ax[0].set_ylim(bottom=0)
    unit_ticks(ax[1], "y", (0, 25, 50, 75, 100), unit="%"); ax[1].set_ylabel("yarn area below" + ARROW)
    ax[0].set_title("area-weighted density", fontsize=9); ax[1].set_title("area-weighted CDF", fontsize=9)
    legend(ax[0], loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig_fvc_kde.png"); plt.close(fig)

    # 2) panorama values: phi_A, phi_N, std_A, gap median
    cols = (("phi_A", r"$\bar\phi_A$", "−"), ("phi_N", r"$\bar\phi_N$", "−"), ("std_A", r"std$_A$", "−"),
            ("gap_median_um", r"median $g$", "µm"))
    fig, ax = plt.subplots(1, len(cols), figsize=(11, 2.9))
    for a, (c, l, u) in zip(ax, cols):
        for k, (t, b) in enumerate(groups):
            p = cell(pan, t, b)
            xk = k + 0.4 * (k // len(batches))
            a.plot(np.full(len(p), xk), p[c], mk[b], ms=5, color=COL[t], mfc=COL[t] if b == batches[0] else "white")
            a.plot([xk - 0.25, xk + 0.25], [p[c].mean()] * 2, color=COL[t], lw=1.5)
        lo, hi = a.get_ylim()
        unit_ticks(a, "y", nice_ticks(lo, hi, 6), unit=u, fmt="{:.3g}"); a.set_ylim(lo, hi)
        a.set_xticks([k + 0.4 * (k // len(batches)) for k in range(len(groups))])
        a.set_xticklabels([gl(t, b) for t, b in groups], rotation=60, fontsize=7); a.set_title(l, fontsize=9)
    fig.suptitle("panorama values (interior triangles of the concave hull)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_fvc_panorama.png"); plt.close(fig)

    # 3) profiles: area-weighted mean phi in bins of t, s, d_bnd
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    specs = (("t_rel", np.linspace(0, 1, 11), "through thickness $t$", (0, 0.2, 0.4, 0.6, 0.8, 1.0), "−"),
             ("s_rel", np.linspace(0, 1, 11), "width position $s$", (0, 0.2, 0.4, 0.6, 0.8, 1.0), "−"),
             ("d_bnd_um", np.arange(0, 49, 4.0), r"distance to hull $d_\partial$", (0, 10, 20, 30, 40, 50), "µm"))
    for a, (c, bins, xl, xt, u) in zip(ax, specs):
        mid = 0.5 * (bins[1:] + bins[:-1])
        for t, b in groups:
            g = cell(T, t, b)
            k = np.digitize(g[c], bins) - 1
            ok = (k >= 0) & (k < len(mid))
            num_ = np.bincount(k[ok], weights=(g["fvc"] * g["area_um2"])[ok], minlength=len(mid))
            den = np.bincount(k[ok], weights=g["area_um2"][ok], minlength=len(mid))
            a.plot(mid, num_ / np.where(den > 0, den, np.nan), **sty(t, b))
        unit_ticks(a, "x", xt, unit=u, fmt="{:g}"); a.set_xlabel(xl + ARROW)
        unit_ticks(a, "y", (0.5, 0.6, 0.7, 0.8), fmt="{:.1f}"); a.set_ylim(0.45, 0.85)
        a.set_ylabel(r"$\bar\phi_A$" + ARROW)
    legend(ax[0], ncol=2, loc="lower center")
    fig.tight_layout(); fig.savefig(out / "fig_fvc_profiles.png"); plt.close(fig)

    # 4) window FVC: mean and std over L (log-log with L^-1 reference)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    Ls = np.array(sorted(W["L_um"].unique()))
    for t, b in groups:
        g = wv[(wv["type"] == t) & (wv["batch"] == b)]
        ax[0].plot(g["L_um"], g["mean"], **sty(t, b))
        ax[1].plot(g["L_um"], g["std"], **sty(t, b))
    ref = wv["std"].max() * (Ls / Ls[0]) ** -1.0
    ax[1].plot(Ls, ref, color=PLASMA10[0], lw=1, ls=":", label=r"$\propto L^{-1}$")
    for a in ax:
        a.set_xscale("log"); unit_ticks(a, "x", Ls, unit="µm"); a.set_xlabel("window size $L$" + ARROW)
    ax[1].set_yscale("log"); unit_ticks(ax[1], "y", (0.02, 0.05, 0.1, 0.2), fmt="{:g}")
    unit_ticks(ax[0], "y", (0.55, 0.6, 0.65, 0.7, 0.75), fmt="{:.2f}"); ax[0].set_ylim(0.55, 0.75)
    ax[0].set_ylabel(phi + "$(L)$ mean" + ARROW); ax[1].set_ylabel(phi + "$(L)$ std" + ARROW)
    legend(ax[0], ncol=2, loc="center right")
    fig.tight_layout(); fig.savefig(out / "fig_fvc_windows.png"); plt.close(fig)

    # 5) gap and spacing between Delaunay neighbours
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    xg, xl_ = np.linspace(-1, 5, 300), np.linspace(5, 12, 300)
    for t, b in groups:
        e = cell(E, t, b)
        g = e["g_um"].to_numpy(); g = g[g < 10]
        ax[0].plot(xg, gaussian_kde(g, bw_method=0.05)(xg), ls=ls[b], lw=1.6, color=COL[t])
        ax[1].plot(xl_, gaussian_kde(e["l_um"], bw_method=0.05)(xl_), ls=ls[b], lw=1.6, color=COL[t])
    unit_ticks(ax[0], "x", (-1, 0, 1, 2, 3, 4, 5), unit="µm"); ax[0].set_xlabel("gap $g_{ij}$" + ARROW)
    unit_ticks(ax[1], "x", (5, 6, 7, 8, 9, 10, 11, 12), unit="µm"); ax[1].set_xlabel(r"spacing $\ell_{ij}$" + ARROW)
    for a in ax:
        a.set_yticks([]); a.set_ylim(bottom=0); a.set_ylabel("density" + ARROW)
    legend(ax[0])
    fig.tight_layout(); fig.savefig(out / "fig_fvc_gap.png"); plt.close(fig)

    # 6) variance shares
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xk = np.arange(len(var))
    bottom = np.zeros(len(var))
    for c, l, colr in (("type", "type", PLASMA10[1]), ("batch_in_type", "batch in type", PLASMA10[4]),
                       ("panorama", "panorama in batch", PLASMA10[7])):
        ax.bar(xk, 100 * var[c], bottom=bottom, color=colr, width=0.6, label=l)
        bottom += 100 * var[c].to_numpy()
    ax.set_xticks(xk)
    ax.set_xticklabels([f"{q}\n{sc}" for q, sc in zip(var["quantity"], var["scope"])], rotation=0, fontsize=7)
    unit_ticks(ax, "y", nice_ticks(0, bottom.max() * 1.1, 6), unit="%")
    ax.set_ylabel("share of total variance" + ARROW); ax.legend(fontsize=8)
    ax.set_title("between-group variance (rest = within panorama)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_fvc_variance.png"); plt.close(fig)

    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(pan.round(4).to_string(index=False))
        print((100 * var.set_index(["quantity", "scope"])).round(2).to_string())
        print(pairs.round(4).to_string(index=False))
        print(wv.round(4).to_string(index=False))
    print(f"→ {out}")


def _triangles_xy(sp):
    """Triangle vertices (T, 3, 2) in µm from the vertex fibre ids of the triangle table."""
    o, name = stats_dir(sp), sp["specimen"]
    fib = pd.read_csv(o / f"{name}.fibres.csv").set_index("id")
    tri = pd.read_csv(o / f"{name}.triangles.csv")
    xy = fib[["x_um", "y_um"]]
    return np.stack([xy.loc[tri[v]].to_numpy() for v in ("v0", "v1", "v2")], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "plot", "xdmf", "aggregate", "compare", "compare-fvc"))
    ap.add_argument("root", help="data root, e.g. data_phd")
    ap.add_argument("--only", nargs="*", help="specimen names, e.g. 160_01_00")
    ap.add_argument("--batch", default="batch_01", help="only specimens of this batch (default batch_01)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hull-edge", type=float, default=HULL_EDGE,
                    help="run: concave hull, peel boundary edges longer than this × median Delaunay edge")
    ap.add_argument("--step", type=int, default=1, help="xdmf: image subsampling (1 = full resolution)")
    ap.add_argument("--labels", action="store_true", help="xdmf: also export the instance label map (uint32)")
    ap.add_argument("--out", default="results/yarn_stats", help="aggregate: output folder for tables + figures")
    ap.add_argument("--map", help="aggregate: specimen for the FVC map figure")
    a = ap.parse_args()
    root = a.root if a.root.endswith("/") else a.root + "/"
    if a.cmd in ("run", "plot", "xdmf"):
        for sp in find_specimens(root):
            if a.only and sp["specimen"] not in a.only:
                continue
            if a.batch and f"/{a.batch}/" not in str(sp["dir"]) + "/":
                continue
            if not Path(sp["h5"]).exists():
                continue
            if a.cmd == "run":
                run_one(sp, a.force, a.hull_edge)
            elif (stats_dir(sp) / f"{sp['specimen']}.summary.json").exists():
                print(sp["specimen"], flush=True)
                if a.cmd == "plot":
                    plot_one(sp)
                else:
                    xdmf_one(sp, a.step, a.labels)
    elif a.cmd == "compare":
        compare(root, a.out)
    elif a.cmd == "compare-fvc":
        compare_fvc(root, a.out)
    else:
        aggregate(root, a.out, a.only, a.map)


if __name__ == "__main__":
    main()

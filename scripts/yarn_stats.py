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
COL = {"I": "#3e049c", "II": "#cc4778", "III": "#f89540"}
INK, GRID = "#52514e", "#e4e3df"


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


def _triangles_xy(sp):
    """Triangle vertices (T, 3, 2) in µm from the vertex fibre ids of the triangle table."""
    o, name = stats_dir(sp), sp["specimen"]
    fib = pd.read_csv(o / f"{name}.fibres.csv").set_index("id")
    tri = pd.read_csv(o / f"{name}.triangles.csv")
    xy = fib[["x_um", "y_um"]]
    return np.stack([xy.loc[tri[v]].to_numpy() for v in ("v0", "v1", "v2")], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "plot", "xdmf", "aggregate"))
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
    else:
        aggregate(root, a.out, a.only, a.map)


if __name__ == "__main__":
    main()

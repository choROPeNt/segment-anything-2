"""
Statistics of a yarn cross-section (micrograph panorama): fibre shape, Delaunay local fibre volume content
(FVC) and window FVC.

Input is one specimen h5 of the ``specimen-folder/1`` layout (see :mod:`sam2.addons.specimen_store`):
``/labeling/labels`` (instance label map, 0 = matrix) and ``/mask`` (yarn region).  Every connected
component of the mask is one yarn.

Fibre shape (:func:`fibre_table`)
    per instance: area, equivalent diameter, perimeter, solidity, centroid, position in the yarn, and the
    elliptic Fourier descriptors (Kuhl & Giardina 1982) up to *n_harm*: semi-axes ς_{n,1} ≥ ς_{n,2} of every
    harmonic ellipse (singular values of M_n = [[a_n, b_n], [c_n, d_n]]), orientation ψ_1 of the first
    harmonic and the reconstruction IoU_H for H = 1..n_harm.

Local FVC (:func:`delaunay_fvc`)
    Delaunay triangulation of the fibre centroids of one yarn.  Triangles whose centroid lies outside the
    yarn are dropped, the remaining ones partition the region spanned by the outer fibre centres.  Each
    triangle T_k gets φ_k = |fibre ∩ T_k| / |T_k| (pixel-exact, every pixel belongs to one triangle), so the
    area-weighted mean Σ|T_k| φ_k / Σ|T_k| is exactly the pixel FVC of the triangulated region.
    :func:`delaunay_edges` gives the centre spacing and the gap between neighbouring fibres.

Window FVC (:func:`window_fvc`)
    φ of square windows of side L on a regular grid, windows completely inside the yarn only.  All windows
    of one L have the same area, so their plain mean needs no weighting; the variance over L is the
    finite-window variance that sets the RVE size.

Weighted statistics: :func:`weighted_quantile`, :func:`weighted_summary`.
"""

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import Delaunay

from .efd import efd_fit_quality


# ------------------------------------------------------------------------------------------------ input

def load_panorama(h5_path, labels_key="labeling/labels", mask_key="mask"):
    """Label map, yarn mask and pixel size (µm/px) of one specimen h5."""
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        labels = f[labels_key][()]
        mask = f[mask_key][()] > 0
    px = float(attrs.get("pixel_size_x", 1.0))
    return {"labels": labels, "mask": mask, "px": px, "attrs": attrs,
            "specimen": str(attrs.get("specimen", "")), "type": str(attrs.get("type", ""))}


def yarn_regions(mask, min_area_px=10_000):
    """Connected components of the yarn mask larger than *min_area_px* → (label map, number of yarns)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return lab.astype(np.int32), 0
    size = np.bincount(lab.ravel())
    keep = np.flatnonzero(size >= min_area_px)
    keep = keep[keep > 0]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lab], len(keep)


def through_thickness(yarn_lab, n_yarn):
    """
    Per yarn and image column the upper / lower yarn edge (rows) → dict yarn → (top, bottom) arrays of
    length W (NaN where the column misses the yarn).  Used for the normalised through-thickness position.
    """
    out = {}
    H = yarn_lab.shape[0]
    for y in range(1, n_yarn + 1):
        m = yarn_lab == y
        has = m.any(0)
        top = np.where(has, np.argmax(m, 0), np.nan).astype(float)
        bot = np.where(has, H - 1 - np.argmax(m[::-1], 0), np.nan).astype(float)
        out[y] = (top, bot)
    return out


def boundary_distance(yarn_lab, down=4):
    """Distance (px) of every pixel to the yarn boundary, on a *down*-times coarser grid (0 outside)."""
    d = ndi.distance_transform_edt(yarn_lab[::down, ::down] > 0) * down
    return d, down


# -------------------------------------------------------------------------------------------- fibre shape

def fibre_table(labels, px, yarn_lab=None, n_harm=5, iou_target=0.95, n_pts=256, border=2, progress=None):
    """
    One row per instance of *labels*.

    Columns: ``id``, ``area_um2``, ``d_eq_um`` (2 sqrt(A/π)), ``perim_um``, ``solidity``, ``circularity``
    (4πA/P²), ``x_um``, ``y_um`` (centroid, image coordinates), ``touches_border``, ``yarn`` (component at
    the centroid, 0 = outside), ``frac_in_yarn``, ``s1_um``, ``s2_um`` (semi-axes ς_{1,1}, ς_{1,2} of the
    first harmonic), ``aspect`` (ς_{1,1}/ς_{1,2}), ``psi1_deg`` (major axis of the first harmonic against
    the image x axis, counter-clockwise, in (−90°, 90°]), ``r{n}`` = ς_{n,1}/ς_{1,1} and ``e{n}`` =
    ς_{n,2}/ς_{1,1} for n = 2..n_harm, ``iou{H}`` for H = 1..n_harm and ``n_min`` (smallest H with
    IoU_H ≥ *iou_target*, n_harm + 1 if none).
    """
    objs = ndi.find_objects(labels)
    Hh, Ww = labels.shape
    rows = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        lid = i + 1
        crop = labels[sl] == lid
        area = int(crop.sum())
        if area < 10:
            continue
        ys, xs = np.nonzero(crop)
        cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
        r = {"id": lid, "area_um2": area * px ** 2, "d_eq_um": 2 * np.sqrt(area / np.pi) * px,
             "x_um": cx * px, "y_um": cy * px,
             "touches_border": bool(sl[0].start < border or sl[1].start < border
                                    or sl[0].stop > Hh - border or sl[1].stop > Ww - border)}
        if yarn_lab is not None:
            r["yarn"] = int(yarn_lab[int(round(cy)), int(round(cx))])
            r["frac_in_yarn"] = float((yarn_lab[sl][crop] > 0).mean())
        cs, _ = cv2.findContours(np.pad(crop, 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        perim = cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        r["perim_um"] = perim * px
        r["solidity"] = cv2.contourArea(c) / hull if hull > 0 else np.nan     # both polygon areas
        r["circularity"] = 4 * np.pi * area / max(perim, 1.0) ** 2
        q = efd_fit_quality(crop, n_max=n_harm, n_pts=n_pts)
        if q is not None:
            M = q["coeffs"].reshape(-1, 2, 2)                 # [[a, b], [c, d]] per harmonic
            U, S, _ = np.linalg.svd(M)
            u = U[0][:, 0]                                    # major axis of the first harmonic, image coords
            psi = np.degrees(np.arctan2(-u[1], u[0]))         # y up
            psi = (psi + 90) % 180 - 90
            r.update(s1_um=S[0, 0] * px, s2_um=S[0, 1] * px, aspect=S[0, 0] / max(S[0, 1], 1e-9),
                     psi1_deg=psi if psi != -90 else 90.0)
            for n in range(2, n_harm + 1):
                r[f"r{n}"] = S[n - 1, 0] / S[0, 0]
                r[f"e{n}"] = S[n - 1, 1] / S[0, 0]
            for h in range(1, n_harm + 1):
                r[f"iou{h}"] = q["iou"][h - 1]
            ok = np.flatnonzero(q["iou"] >= iou_target)
            r["n_min"] = int(ok[0] + 1) if len(ok) else n_harm + 1
        rows.append(r)
        if progress is not None and len(rows) % 2000 == 0:
            progress(len(rows))
    return pd.DataFrame(rows)


def add_yarn_position(df, yarn_lab, px, n_yarn):
    """
    Add ``d_bnd_um`` (distance of the centroid to the yarn boundary), ``t_rel`` (through-thickness position
    0 = upper yarn edge … 1 = lower edge, in the centroid's column) and ``s_rel`` (position across the yarn
    width, 0 … 1) to a table with ``x_um``, ``y_um``, ``yarn`` columns (in place).
    """
    dist, down = boundary_distance(yarn_lab)
    edges = through_thickness(yarn_lab, n_yarn)
    xs = np.clip(np.round(df["x_um"].to_numpy() / px).astype(int), 0, yarn_lab.shape[1] - 1)
    ys = np.clip(np.round(df["y_um"].to_numpy() / px).astype(int), 0, yarn_lab.shape[0] - 1)
    df["d_bnd_um"] = dist[np.minimum(ys // down, dist.shape[0] - 1), np.minimum(xs // down, dist.shape[1] - 1)] * px
    t = np.full(len(df), np.nan)
    s = np.full(len(df), np.nan)
    yarn = df["yarn"].to_numpy()
    for y, (top, bot) in edges.items():
        sel = yarn == y
        cols = np.flatnonzero(~np.isnan(top))
        x0, x1 = cols.min(), cols.max()
        tt, bb = top[xs[sel]], bot[xs[sel]]
        t[sel] = (ys[sel] - tt) / np.maximum(bb - tt, 1)
        s[sel] = (xs[sel] - x0) / max(x1 - x0, 1)
    df["t_rel"], df["s_rel"] = t, s
    return df


def valid_fibres(df, area_range=(0.4, 2.0), aspect_max=2.0, min_in_yarn=0.5):
    """
    Bool mask of complete fibres inside a yarn: not cut by the image border, ≥ *min_in_yarn* of the area
    inside the yarn, area within *area_range* × median, first-harmonic aspect ≤ *aspect_max*.
    """
    med = df["area_um2"].median()
    ok = ~df["touches_border"] & (df["yarn"] > 0) & (df["frac_in_yarn"] >= min_in_yarn)
    ok &= df["area_um2"].between(area_range[0] * med, area_range[1] * med)
    ok &= df["aspect"].fillna(np.inf) <= aspect_max
    return ok


# ------------------------------------------------------------------------------------------ Delaunay FVC

def _fill_triangles(P, shape, ids, out=None):
    """Rasterise triangles P (T, 3, 2) [x, y] px with value ids into an int32 map (−1 = none)."""
    out = np.full(shape, -1, np.int32) if out is None else out
    pts = np.round(P * 16).astype(np.int32)
    for p, v in zip(pts, ids):
        cv2.fillPoly(out, [p], int(v), lineType=cv2.LINE_8, shift=4)
    return out


def concave_hull(tri, xy, max_edge):
    """
    Tightest boundary of a point set by peeling its Delaunay triangulation from the outside (χ-shape).

    Starting at the convex hull, a triangle is removed while one of its *open* edges (on the current outer
    boundary) is longer than *max_edge*; its neighbours become boundary triangles and are tested in turn.
    Only triangles reachable from the outside are removed, so enclosed resin pockets stay part of the region.

    Returns bool (T,) kept triangles.
    """
    from collections import deque
    S, nb = tri.simplices, tri.neighbors
    P = xy[S]
    # length of the edge opposite vertex k (= the edge shared with neighbours[:, k])
    L = np.stack([np.linalg.norm(P[:, (k + 1) % 3] - P[:, (k + 2) % 3], axis=1) for k in range(3)], 1)
    keep = np.ones(len(S), bool)
    q = deque(np.flatnonzero((nb < 0).any(1)))
    while q:
        t = q.popleft()
        if not keep[t]:
            continue
        n = nb[t]
        open_e = (n < 0) | ~keep[np.maximum(n, 0)]
        if (L[t][open_e] > max_edge).any():
            keep[t] = False
            q.extend(int(m) for m in n if m >= 0 and keep[m])
    return keep


def delaunay_fvc(xy, fibre, px, yarn_mask=None, hull_edge=2.0, index_map=None, id_offset=0):
    """
    Local FVC of the Delaunay triangles of the fibre centroids *xy* (N, 2) [x, y] px of one yarn.

    The region is the tightest boundary of the fibre centres (:func:`concave_hull`): triangles are peeled
    from the outside while their boundary edge is longer than *hull_edge* × the median Delaunay edge.
    With *yarn_mask* given instead (*hull_edge* = None), triangles with the centroid outside the mask are dropped.

    Parameters
    ----------
    fibre : (H, W) bool fibre pixels (``labels > 0``)
    index_map : optional int32 (H, W) map shared by several yarns (filled in place, −1 = none)
    id_offset : first triangle id written into *index_map*

    Returns
    -------
    (triangles DataFrame, simplices (T, 3) into *xy* of the rows of the DataFrame, index_map, info dict with
    ``max_edge_px`` and ``n_peeled``).  Columns: ``tri`` (id in the map), ``x_um``, ``y_um`` (triangle
    centroid), ``area_um2`` (pixel count), ``fibre_um2``, ``fvc``, ``max_edge_um``, ``min_angle_deg``,
    ``boundary`` (has an edge on the region boundary).
    """
    tri = Delaunay(xy)
    P = xy[tri.simplices]
    cen = P.mean(1)
    H, W = fibre.shape
    info = {}
    if hull_edge is not None:
        E = np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [2, 0]]])
        E = np.unique(np.sort(E, 1), axis=0)
        med = float(np.median(np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)))
        keep = concave_hull(tri, xy, hull_edge * med)
        info = {"median_edge_px": med, "max_edge_px": hull_edge * med, "n_peeled": int((~keep).sum())}
    else:
        ci = np.clip(np.round(cen).astype(int), 0, [W - 1, H - 1])
        keep = yarn_mask[ci[:, 1], ci[:, 0]]
        info = {"n_peeled": int((~keep).sum())}
    k_idx = np.flatnonzero(keep)
    ids = id_offset + np.arange(len(k_idx))
    index_map = _fill_triangles(P[k_idx], (H, W), ids, index_map)

    sel = (index_map >= id_offset) & (index_map < id_offset + len(k_idx))
    loc = index_map[sel] - id_offset
    n_px = np.bincount(loc, minlength=len(k_idx))
    n_fib = np.bincount(loc, weights=fibre[sel], minlength=len(k_idx))

    Pk = P[k_idx]
    e = np.linalg.norm(Pk - np.roll(Pk, -1, axis=1), axis=2)                    # edge k: vertex k → k+1
    a, b, c = e[:, 1], e[:, 2], e[:, 0]                                          # opposite vertex 0, 1, 2
    cosA = np.clip((b ** 2 + c ** 2 - a ** 2) / np.maximum(2 * b * c, 1e-9), -1, 1)
    cosB = np.clip((a ** 2 + c ** 2 - b ** 2) / np.maximum(2 * a * c, 1e-9), -1, 1)
    angles = np.degrees(np.stack([np.arccos(cosA), np.arccos(cosB)], 1))
    min_ang = np.minimum(angles.min(1), 180 - angles.sum(1))

    nb = tri.neighbors[k_idx]
    boundary = np.any((nb < 0) | ~keep[np.maximum(nb, 0)], axis=1)
    df = pd.DataFrame({
        "tri": ids, "x_um": cen[k_idx, 0] * px, "y_um": cen[k_idx, 1] * px,
        "area_um2": n_px * px ** 2, "fibre_um2": n_fib * px ** 2, "fvc": n_fib / np.maximum(n_px, 1),
        "max_edge_um": e.max(1) * px, "min_angle_deg": min_ang, "boundary": boundary,
    })
    ok = n_px > 0
    return df[ok].reset_index(drop=True), tri.simplices[k_idx][ok], index_map, info


def hull_outline(tri_df, xy_of_id):
    """Boundary segments (S, 2, 2) of the triangulated region: edges used by exactly one triangle."""
    V = tri_df[["v0", "v1", "v2"]].to_numpy()
    E = np.sort(np.concatenate([V[:, [0, 1]], V[:, [1, 2]], V[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    b = u[c == 1]
    return np.stack([xy_of_id(b[:, 0]), xy_of_id(b[:, 1])], 1)


def delaunay_edges(simplices, xy, d_eq, px):
    """
    Unique edges of the kept triangles: centre spacing ``l_um`` and gap ``g_um`` = l − (d_i + d_j)/2
    between the equivalent circles of the two fibres (negative = touching / non-circular overlap).
    """
    e = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    l = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1) * px
    return pd.DataFrame({"i": e[:, 0], "j": e[:, 1], "l_um": l, "g_um": l - (d_eq[e[:, 0]] + d_eq[e[:, 1]]) / 2})


# ------------------------------------------------------------------------------------------- window FVC

def window_fvc(fibre, region, px, sizes_um=(5, 10, 20, 40, 80)):
    """
    φ of non-overlapping square windows of side L (µm) completely inside *region*.

    Returns DataFrame with ``L_um``, ``x_um``, ``y_um`` (window centre), ``fvc``.
    """
    H, W = fibre.shape
    out = []
    for L in sizes_um:
        w = max(int(round(L / px)), 1)
        h_, w_ = H // w, W // w
        if h_ == 0 or w_ == 0:
            continue
        f = fibre[:h_ * w, :w_ * w].reshape(h_, w, w_, w).sum((1, 3), dtype=np.int64)
        r = region[:h_ * w, :w_ * w].reshape(h_, w, w_, w).sum((1, 3), dtype=np.int64)
        iy, ix = np.nonzero(r == w * w)
        out.append(pd.DataFrame({"L_um": L, "x_um": (ix + 0.5) * w * px, "y_um": (iy + 0.5) * w * px,
                                 "fvc": f[iy, ix] / (w * w)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["L_um", "x_um", "y_um", "fvc"])


# ------------------------------------------------------------------------------------- weighted statistics

def weighted_quantile(v, w, q):
    """Quantiles *q* of *v* with weights *w* (inverse of the weighted empirical CDF, midpoint rule)."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cw, v)


def weighted_summary(v, w=None, prefix=""):
    """Mean, std, median, 5 / 95 % quantiles of *v* (area-weighted if *w* is given)."""
    v = np.asarray(v, float)
    w = np.ones_like(v) if w is None else np.asarray(w, float)
    m = np.average(v, weights=w)
    s = np.sqrt(np.average((v - m) ** 2, weights=w))
    q5, q50, q95 = weighted_quantile(v, w, [0.05, 0.5, 0.95])
    return {f"{prefix}mean": m, f"{prefix}std": s, f"{prefix}median": q50, f"{prefix}q05": q5, f"{prefix}q95": q95}


# ------------------------------------------------------------------------------------------- XDMF export

def _xdmf_item(h5name, path, arr):
    nt = {"u": "UInt", "i": "Int", "f": "Float", "b": "UChar"}[arr.dtype.kind]
    prec = 1 if arr.dtype.kind == "b" else arr.dtype.itemsize
    dims = " ".join(str(d) for d in arr.shape)
    return (f'<DataItem Dimensions="{dims}" NumberType="{nt}" Precision="{prec}" Format="HDF">'
            f'{h5name}:{path}</DataItem>')


def export_fvc_xdmf(path, image_fields, px, step=1, nodes=None, triangles=None, tri_fields=None, node_fields=None,
                    edges=None, edge_fields=None):
    """
    One HDF5 file + ``.xdmf`` for ParaView (open the .xdmf with the *XDMF Reader*), coordinates in µm with
    y pointing up (image row 0 at the top, so all grids overlap with the micrograph upright).

    Parameters
    ----------
    path : output path without suffix → ``{path}.h5`` + ``{path}.xdmf``
    image_fields : {name: (H, W) array} full-resolution image fields (subsampled by *step*)
    px : pixel size (µm)
    nodes : (N, 2) node positions [x, y] in image coordinates (µm, y down)
    triangles : (T, 3) node indices, *tri_fields* {name: (T,)} cell data
    node_fields : {name: (N,)} point data of the triangle / edge meshes
    edges : (E, 2) node indices, *edge_fields* {name: (E,)} cell data of the edge grid (polylines)
    """
    import h5py
    from pathlib import Path

    path = Path(path)
    h5p, xp = path.parent / f"{path.name}.h5", path.parent / f"{path.name}.xdmf"
    H, W = next(iter(image_fields.values())).shape[:2]
    rows = np.arange(0, H, step)[::-1]                       # stored bottom row first → y increases upwards
    ny, nx = len(rows), len(range(0, W, step))
    grids = []
    with h5py.File(h5p, "w") as f:
        attrs = []
        for k, v in image_fields.items():
            a = np.ascontiguousarray(np.asarray(v)[rows][:, ::step])[None]      # (1, ny, nx), z y x
            if a.dtype == bool:
                a = a.astype(np.uint8)
            f.create_dataset(f"image/{k}", data=a, compression="gzip", chunks=True)
            attrs.append(f'<Attribute Name="{k}" AttributeType="Scalar" Center="Node">'
                         f'{_xdmf_item(h5p.name, f"image/{k}", a)}</Attribute>')
        y0 = -rows[0] * px
        grids.append(f'''<Grid Name="image" GridType="Uniform">
  <Topology TopologyType="3DCoRectMesh" Dimensions="1 {ny} {nx}"/>
  <Geometry GeometryType="ORIGIN_DXDYDZ">
    <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">0.0 {y0} 0.0</DataItem>
    <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">1.0 {step * px} {step * px}</DataItem>
  </Geometry>
  {"".join(attrs)}
</Grid>''')
        if nodes is not None:
            xyz = np.column_stack([nodes[:, 0], -nodes[:, 1], np.zeros(len(nodes))]).astype(np.float64)
            f.create_dataset("mesh/nodes", data=xyz)
            geo = f'<Geometry GeometryType="XYZ">{_xdmf_item(h5p.name, "mesh/nodes", xyz)}</Geometry>'
            nattrs = ""
            for k, v in (node_fields or {}).items():
                a = np.asarray(v)
                a = a.astype(np.uint8) if a.dtype == bool else a
                f.create_dataset(f"mesh/node/{k}", data=a)
                nattrs += (f'<Attribute Name="{k}" AttributeType="Scalar" Center="Node">'
                           f'{_xdmf_item(h5p.name, f"mesh/node/{k}", a)}</Attribute>')
            for name, cells, fields, topo in (("triangles", triangles, tri_fields, 'TopologyType="Triangle"'),
                                              ("edges", edges, edge_fields, 'TopologyType="Polyline" NodesPerElement="2"')):
                if cells is None:
                    continue
                c = np.asarray(cells, dtype=np.int64)
                f.create_dataset(f"mesh/{name}/cells", data=c)
                cattrs = ""
                for k, v in (fields or {}).items():
                    a = np.asarray(v)
                    a = a.astype(np.uint8) if a.dtype == bool else a
                    f.create_dataset(f"mesh/{name}/{k}", data=a)
                    cattrs += (f'<Attribute Name="{k}" AttributeType="Scalar" Center="Cell">'
                               f'{_xdmf_item(h5p.name, f"mesh/{name}/{k}", a)}</Attribute>')
                grids.append(f'''<Grid Name="{name}" GridType="Uniform">
  <Topology {topo} NumberOfElements="{len(c)}">{_xdmf_item(h5p.name, f"mesh/{name}/cells", c)}</Topology>
  {geo}
  {cattrs}{nattrs}
</Grid>''')
    xp.write_text('<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n<Xdmf Version="3.0">\n<Domain>\n'
                  + "\n".join(grids) + "\n</Domain>\n</Xdmf>\n")
    return xp

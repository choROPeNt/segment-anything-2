"""
Specimen h5 files that mirror the panorama folder structure, with sync between folder and h5.

Folder (one panorama ``{s} = {type}_{batch}_{panorama}``)::

    {type}/batch_{BB}/{PP}/
        {s}.tif(f)  {s}.mask.tif(f)
        labeling/{s}.tiff.labeling  {s}.labels.tiff
        overview/{s}_overview.tiff  {s}_overview.vsi.labeling
        panorama/{s}-{row}.vsi / .tiff

h5 ``{type}/batch_{BB}/{s}.h5`` — one entry of :data:`LAYOUT` per folder file::

    /image (greyscale of the colour panorama), /mask (+ *_preview)
    /labeling/labels, labels_raw, instances/, labels_preview
    /overview/image_preview, /overview/labels
    /panorama/row_file, row_shape [h, w], rows/<k>  (VSI rows: list, sizes and pixel data of the
                                                     full-resolution scene; pixel size → root attrs)
    /legacy/             (superseded datasets, kept by :func:`migrate`)

Every imported dataset stores its source file (``source``, ``source_size``, ``source_mtime``,
``source_sha256``); exported files are recorded in the dataset attr ``exports``.  :func:`status` compares
these signatures with the folder (``deep=True``: compares the content), :func:`sync_import` /
:func:`sync_export` copy in either direction.  New kinds of files: add an :class:`Entry` to :data:`LAYOUT`.
"""

import contextlib
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import find_objects

from .labelkit_io import export_labels_to_labelkit, load_labkit_json

Image.MAX_IMAGE_PIXELS = None
LAYOUT_VERSION = "specimen-folder/1"


@dataclass(frozen=True)
class Entry:
    dataset: str          # path in the h5
    pattern: str          # file in the panorama folder ({s} = specimen name, {ext} = tif|tiff)
    kind: str             # image | mask | labkit | label_tiff | preview_image | files
    direction: str = "both"   # both | import | export


LAYOUT = (
    Entry("image", "{s}.{ext}", "image", "import"),        # stored as greyscale (to_gray)
    Entry("mask", "{s}.mask.{ext}", "mask"),
    Entry("labeling/labels", "labeling/{s}.tiff.labeling", "labkit"),
    Entry("labeling/labels", "labeling/{s}.labels.tiff", "label_tiff", "export"),
    Entry("overview/image_preview", "overview/{s}_overview.tiff", "preview_image", "import"),
    Entry("overview/labels", "overview/{s}_overview.vsi.labeling", "labkit"),
    Entry("panorama", "panorama/{s}-*.vsi", "vsi_rows", "import"),   # row list [h, w] + pixel size from the VSI
)


# ------------------------------------------------------------------------------------------- discovery

def find_specimens(root):
    """All panorama folders ``{type}/batch_{BB}/{PP}/`` that contain ``{type}_{BB}_{PP}.tif(f)``."""
    out = []
    for d in sorted(Path(root).glob("*/batch_*/*")):
        if not d.is_dir():
            continue
        t, bb, pp = d.parent.parent.name, d.parent.name.split("_")[-1], d.name
        s = f"{t}_{bb}_{pp}"
        ext = next((e for e in ("tiff", "tif") if (d / f"{s}.{e}").exists()), None)
        if ext:
            out.append({"specimen": s, "type": t, "batch": bb, "panorama": pp, "dir": d, "ext": ext,
                        "h5": d.parent / f"{s}.h5"})
    return out


def _files(sp, e):
    pat = e.pattern.format(s=sp["specimen"], ext=sp["ext"])
    if e.kind in ("files", "vsi_rows"):
        files = [p for p in sp["dir"].glob(pat) if not p.name.startswith(".")]
        return sorted(files, key=lambda p: (len(p.stem), p.stem))          # -1, -2, … -10 in row order
    if "{ext}" in e.pattern:          # masks may use the other tif extension
        for ext in (sp["ext"], "tiff", "tif"):
            p = sp["dir"] / e.pattern.format(s=sp["specimen"], ext=ext)
            if p.exists():
                return [p]
    p = sp["dir"] / pat
    return [p] if p.exists() else []


def _sig(path, sha=False):
    st = path.stat()
    s = {"size": st.st_size, "mtime": int(st.st_mtime)}
    if sha:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        s["sha256"] = h.hexdigest()
    return s


# --------------------------------------------------------------------------------------------- reading

def to_gray(img):
    """Greyscale (ITU-R 601 luma 0.299 R + 0.587 G + 0.114 B, rounded) of an RGB image; 2-D input unchanged."""
    import cv2
    img = np.asarray(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _read_file(path, kind):
    if kind == "image":
        return to_gray(np.asarray(Image.open(path)))
    if kind in ("preview_image", "label_tiff"):
        return np.asarray(Image.open(path))
    if kind == "mask":
        return (np.asarray(Image.open(path)) > 0).astype(np.uint8)
    if kind == "labkit":
        return read_labkit_labels(path)
    raise ValueError(kind)


def read_labkit_labels(path, verbose=False):
    """
    Dense uint32 label map of a Labkit ``.labeling``. Numeric label names keep their id. Labels drawn in
    Labkit under its default names ("Label 1", ...) get new ids after the largest numeric id, in natural
    name order, so new fibres are not dropped. Empty labels are skipped.
    """
    import re
    with contextlib.redirect_stdout(io.StringIO()):
        lab, data = load_labkit_json(path)
    lab = lab.astype(np.uint32)
    W = data["interval"]["max"][0] - data["interval"]["min"][0] + 1
    lin = [np.unique(np.asarray(c, np.int64) @ np.array([1, W])) for c in data["labels"].values() if c]
    if lin:
        _, cnt = np.unique(np.concatenate(lin), return_counts=True)
        if (cnt > 1).any():
            print(f"  warning: {path}: {int((cnt > 1).sum())} px belong to more than one label "
                  "(the label written last wins, named labels after numeric ones)")
    named = [k for k, c in data["labels"].items() if c and not k.strip().isdigit()]
    if named:
        named.sort(key=lambda k: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", k)])
        xmin, ymin = data["interval"]["min"][:2]
        nxt = int(max([int(k) for k, c in data["labels"].items() if c and k.strip().isdigit()] or [0])) + 1
        for k in named:
            xy = np.asarray(data["labels"][k], dtype=np.int64)
            lab[xy[:, 1] - ymin, xy[:, 0] - xmin] = nxt
            if verbose:
                print(f"  {k!r} -> id {nxt}")
            nxt += 1
    return lab


def _same_content(path, ds, kind):
    a, b = _read_file(path, kind), ds[()]
    if kind == "mask":
        return a.shape == b.shape and np.array_equal(a > 0, b > 0)
    return a.shape == b.shape and np.array_equal(a.astype(np.int64), b.astype(np.int64))


# -------------------------------------------------------------------------------------------- status

def status(sp, deep=False):
    """
    State of every :data:`LAYOUT` entry of one specimen.

    Returns a list of dicts (entry, file(s), state) with state one of
    ``ok`` | ``folder changed`` | ``untracked`` (both exist, no signature — run adopt) | ``differs`` |
    ``new in folder`` | ``missing in folder`` | ``missing in h5`` | ``-`` (neither exists).
    """
    rows = []
    h5 = sp["h5"]
    hf = h5py.File(h5, "r") if h5.exists() else None
    try:
        for e in LAYOUT:
            files = _files(sp, e)
            obj = hf.get(e.dataset) if hf is not None else None
            if e.kind in ("files", "vsi_rows"):
                recorded = json.loads(obj.attrs.get("sources", "[]")) if obj is not None else []
                cur = [{"file": p.name, **_sig(p)} for p in files]
                st = "-" if not files and obj is None else "new in folder" if obj is None else \
                     "ok" if {json.dumps(r, sort_keys=True) for r in recorded} == {json.dumps(c, sort_keys=True) for c in cur} \
                     else "folder changed"
                rows.append({"entry": e, "files": f"{len(files)} files", "state": st})
                continue
            f = files[0] if files else None
            if f is None and obj is None:
                st = "-"
            elif f is None:
                st = "missing in folder" if e.direction != "import" else "-"
            elif obj is None:
                st = "new in folder" if e.direction != "export" else "missing in h5"
            else:
                if e.direction == "export":
                    rec = json.loads(obj.attrs.get("exports", "{}")).get(e.pattern.format(s=sp["specimen"], ext=sp["ext"]))
                else:
                    rec = {"size": obj.attrs["source_size"], "mtime": obj.attrs["source_mtime"]} if "source_size" in obj.attrs else None
                cur = _sig(f)
                if rec is None:
                    st = "untracked"
                    if deep and e.kind != "preview_image":
                        st = "untracked, same content" if _same_content(f, obj, e.kind) else "differs"
                elif int(rec["size"]) == cur["size"] and int(rec["mtime"]) == cur["mtime"]:
                    st = "ok"
                else:
                    st = "folder changed"
                    if deep and e.kind != "preview_image" and _same_content(f, obj, e.kind):
                        st = "ok (file touched, same content)"
            rows.append({"entry": e, "files": f.relative_to(sp["dir"]).as_posix() if f else "", "state": st})
    finally:
        if hf is not None:
            hf.close()
    return rows


# ------------------------------------------------------------------------------------------- writing

def _put(hf, name, data, attrs=None):
    if name in hf:
        del hf[name]
    ds = hf.create_dataset(name, data=data, compression="gzip", compression_opts=4)
    for k, v in (attrs or {}).items():
        ds.attrs[k] = v
    return ds


def write_instances(hf, label_map, group="labeling/instances", pixel_size=None):
    """Rebuild the instance crops (segmentation_crop, bbox, area[, area_um2]) from a label map."""
    if group in hf:
        del hf[group]
    g = hf.create_group(group)
    n = 0
    for lid, sl in enumerate(find_objects(label_map), 1):
        if sl is None:
            continue
        seg = label_map[sl] == lid
        y, x = sl[0].start, sl[1].start
        h, w = seg.shape
        ig = g.create_group(f"{lid:05d}")
        ig.create_dataset("segmentation_crop", data=seg, compression="gzip")
        ig.create_dataset("bbox", data=np.array([x, y, w, h], dtype=np.int32))
        ig.attrs.update({"area": int(seg.sum()), "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h})
        if pixel_size:
            ig.attrs["area_um2"] = float(seg.sum()) * pixel_size ** 2
        n += 1
    return n


def _record_source(ds, path, root):
    s = _sig(path, sha=True)
    ds.attrs.update({"source": path.relative_to(root).as_posix(), "source_size": s["size"],
                     "source_mtime": s["mtime"], "source_sha256": s["sha256"]})


def _import_entry(hf, sp, e, root, preview=2):
    files = _files(sp, e)
    if e.kind == "files":
        g = hf.require_group(e.dataset)
        g.attrs["sources"] = json.dumps([{"file": p.name, **_sig(p)} for p in files])
        return f"{len(files)} files listed"
    if e.kind == "vsi_rows":
        return _import_vsi_rows(hf, e.dataset, files)
    f = files[0]
    data = _read_file(f, e.kind)
    msg = ""
    if e.kind == "preview_image":
        import cv2
        h, w = data.shape[:2]
        data = cv2.resize(data, (w // preview, h // preview), interpolation=cv2.INTER_AREA)
        ds = _put(hf, e.dataset, data, {"preview_factor": preview, "CLASS": np.bytes_("IMAGE"),
                                        "IMAGE_VERSION": np.bytes_("1.2")})
        if data.ndim == 3:
            ds.attrs.update({"IMAGE_SUBCLASS": np.bytes_("IMAGE_TRUECOLOR"), "INTERLACE_MODE": np.bytes_("INTERLACE_PIXEL")})
    else:
        ds = _put(hf, e.dataset, data, {"axes": "YXC" if data.ndim == 3 else "YX"})
        if e.kind == "image":
            ds.attrs["color"] = "greyscale (ITU-R 601 luma) of the RGB source"
    _record_source(ds, f, root)
    if e.dataset in ("image", "mask", "labeling/labels") and preview:
        _previews(hf, [e.dataset], preview)
    if e.dataset == "labeling/labels":
        n = write_instances(hf, data, pixel_size=hf.attrs.get("pixel_size_x"))
        hf.attrs["n_instances"] = n
        msg = f"{n} instances rebuilt"
        if "mask" in hf:
            out = int(((data > 0) & (hf["mask"][()] == 0)).sum())
            if out:
                msg += f", WARNING {out} fibre px outside /mask"
    if e.dataset == "image":
        hf.attrs["image_shape"] = data.shape
    return msg


def vsi_info(path, load=False):
    """Size [h, w] and pixel size (µm) of the full-resolution scene of a VSI (``load``: also its pixels)."""
    from aicsimageio import AICSImage
    from aicsimageio.readers.bioformats_reader import BioformatsReader
    img = AICSImage(path, reader=BioformatsReader)
    img.set_scene(img.scenes[0])
    pps = img.physical_pixel_sizes
    info = {"shape": [int(img.dims.Y), int(img.dims.X)], "pixel_size_x": pps.X, "pixel_size_y": pps.Y,
            "scene": img.scenes[0]}
    if load:
        info["data"] = np.asarray(img.get_image_data("YX", T=0, C=0, Z=0))
    return info


def _import_vsi_rows(hf, name, files, store_data=True):
    infos = [vsi_info(p) for p in files]
    if name in hf:
        del hf[name]
    g = hf.create_group(name)
    if store_data:
        rg = g.create_group("rows")
        for p, info in zip(files, infos):
            k = p.stem.rsplit("-", 1)[-1]
            data = vsi_info(p, load=True)["data"]
            assert list(data.shape) == info["shape"], (p, data.shape, info["shape"])
            ds = rg.create_dataset(k, data=data, compression="gzip", compression_opts=4, chunks=True)
            ds.attrs.update({"source": p.name, "scene": info["scene"], "axes": "YX",
                             "pixel_size_x": info["pixel_size_x"] or 0.0, "pixel_size_y": info["pixel_size_y"] or 0.0,
                             "pixel_size_unit": "µm"})
    g.create_dataset("row_file", data=np.array([p.name for p in files], dtype=h5py.string_dtype()))
    g.create_dataset("row_shape", data=np.array([i["shape"] for i in infos], dtype=np.int32).reshape(-1, 2))
    g["row_shape"].attrs["columns"] = "h, w (px) of the full-resolution scene"
    g.attrs["sources"] = json.dumps([{"file": p.name, **_sig(p)} for p in files])
    px = {(i["pixel_size_x"], i["pixel_size_y"]) for i in infos}
    msg = f"{len(files)} VSI rows"
    if len(px) == 1 and None not in next(iter(px)):
        pxx, pxy = next(iter(px))
        old = hf.attrs.get("pixel_size_x")
        hf.attrs.update({"pixel_size_x": pxx, "pixel_size_y": pxy, "pixel_size_unit": "µm",
                         "pixel_size_source": f"VSI {files[0].name} (scene {infos[0]['scene']})"})
        g.attrs.update({"pixel_size_x": pxx, "pixel_size_y": pxy, "pixel_size_unit": "µm"})
        msg += f", pixel size {pxx:.6f} µm" + (f" (was {old:.6f})" if old is not None and abs(old - pxx) > 1e-9 else "")
    elif files:
        msg += f", WARNING inconsistent pixel sizes {sorted(px)}"
    return msg


def _previews(hf, keys, factor):
    import cv2
    for key in keys:
        src = hf[key]
        h, w = src.shape[:2]
        if key == "image":
            prev = cv2.resize(src[()], (w // factor, h // factor), interpolation=cv2.INTER_AREA)
        else:
            prev = src[: (h // factor) * factor: factor, : (w // factor) * factor: factor]
        ds = _put(hf, f"{key}_preview", prev, {"preview_of": key, "preview_factor": factor})
        if key == "image":
            ds.attrs.update({"CLASS": np.bytes_("IMAGE"), "IMAGE_VERSION": np.bytes_("1.2")})
            ds.attrs.update({"IMAGE_SUBCLASS": np.bytes_("IMAGE_TRUECOLOR"), "INTERLACE_MODE": np.bytes_("INTERLACE_PIXEL")}
                            if prev.ndim == 3 else {"IMAGE_SUBCLASS": np.bytes_("IMAGE_GRAYSCALE")})


def _set_root_attrs(hf, sp):
    hf.attrs.update({"specimen": sp["specimen"], "type": sp["type"], "batch": sp["batch"],
                     "panorama": sp["panorama"], "layout": LAYOUT_VERSION})


def sync_import(sp, root, states=("new in folder", "folder changed"), create=False, pixel_size=None, preview=2,
                entries=None):
    """Folder → h5 for all entries whose state is in *states* (``entries``: only these datasets). Creates the h5 if *create*."""
    if not sp["h5"].exists() and not create:
        return ["h5 missing (use create)"]
    todo = [r for r in status(sp) if r["state"] in states and r["entry"].direction != "export"
            and (entries is None or r["entry"].dataset in entries)]
    log = []
    with h5py.File(sp["h5"], "a") as hf:
        _set_root_attrs(hf, sp)
        if pixel_size and "pixel_size_x" not in hf.attrs:
            hf.attrs.update({"pixel_size_x": pixel_size, "pixel_size_y": pixel_size, "pixel_size_unit": "µm"})
        for r in sorted(todo, key=lambda r: [x.dataset for x in LAYOUT].index(r["entry"].dataset)):
            log.append(f"import {r['files']} -> /{r['entry'].dataset}  {_import_entry(hf, sp, r['entry'], root, preview)}")
    return log


def sync_export(sp, root, states=("missing in folder",), force=False, entries=None):
    """h5 → folder for exportable entries whose state is in *states* (``force``: all exportable; ``entries``: only these patterns)."""
    log = []
    rows = status(sp)
    with h5py.File(sp["h5"], "a") as hf:
        for r in rows:
            e = r["entry"]
            if e.direction == "import" or e.kind == "files" or e.dataset not in hf:
                continue
            if not (force or r["state"] in states):
                continue
            if entries is not None and e.pattern not in entries:
                continue
            rel = e.pattern.format(s=sp["specimen"], ext=sp["ext"])
            path = sp["dir"] / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            data = hf[e.dataset][()]
            if e.kind == "labkit":
                H, W = data.shape
                meta = {"interval": {"min": [0, 0], "max": [W - 1, H - 1], "n": 2},
                        "pixelSizes": [{"size": 1.0}, {"size": 1.0}]}
                with contextlib.redirect_stdout(io.StringIO()):
                    export_labels_to_labelkit(data, range(1, int(data.max()) + 1), path, meta)
            elif e.kind == "label_tiff":
                Image.fromarray(data.astype(np.int32)).save(path)
            elif e.kind == "mask":
                Image.fromarray((data > 0).astype(np.uint8) * 255).save(path)
            elif e.kind == "image":
                Image.fromarray(data).save(path)
            ds = hf[e.dataset]
            if e.direction == "export":
                ex = json.loads(ds.attrs.get("exports", "{}"))
                ex[rel] = _sig(path)
                ds.attrs["exports"] = json.dumps(ex)
            else:
                _record_source(ds, path, root)
            log.append(f"export /{e.dataset} -> {rel}")
    return log


def adopt(sp, root):
    """Record signatures for entries that exist in both places with identical content (no data is changed)."""
    log = []
    rows = status(sp, deep=True)
    with h5py.File(sp["h5"], "a") as hf:
        for r in rows:
            e = r["entry"]
            if r["state"] != "untracked, same content":
                if r["state"] in ("differs",):
                    log.append(f"NOT adopted (content differs): {r['files']} vs /{e.dataset}")
                continue
            path = sp["dir"] / r["files"]
            if e.direction == "export":
                ex = json.loads(hf[e.dataset].attrs.get("exports", "{}"))
                ex[r["files"]] = _sig(path)
                hf[e.dataset].attrs["exports"] = json.dumps(ex)
            else:
                _record_source(hf[e.dataset], path, root)
            log.append(f"adopted {r['files']} = /{e.dataset}")
    return log


# ------------------------------------------------------------------------------------------ migration

MIGRATION = (("labels", "labeling/labels"), ("labels_preview", "labeling/labels_preview"),
             ("labels_raw", "labeling/labels_raw"), ("instances", "labeling/instances"),
             ("mask", "legacy/mask_roi_instances"), ("mask_preview", None), ("mask_roi", "legacy/mask_roi"))


def migrate(sp):
    """Flat merged h5 (image, labels, labels_raw, mask, mask_roi, instances) → folder layout. Moves only."""
    log = []
    with h5py.File(sp["h5"], "a") as hf:
        if hf.attrs.get("layout") == LAYOUT_VERSION:
            return ["already migrated"]
        for old, new in MIGRATION:
            if old not in hf:
                continue
            if new is None:
                del hf[old]; log.append(f"deleted /{old} (rebuilt on import)")
                continue
            hf.require_group(new.rsplit("/", 1)[0])
            hf.move(old, new); log.append(f"/{old} -> /{new}")
        for k in ("legacy/mask_roi_instances",):
            if k in hf:
                hf[k].attrs["description"] = "previous /mask: ROI (former mask_refined) extended by all instances"
        _set_root_attrs(hf, sp)
    return log


def convert_image_to_gray(sp, preview=2):
    """
    Replace an RGB /image (+ /image_preview) by greyscale and rewrite the file so the space is freed
    (HDF5 does not shrink on delete).  All other objects and attributes are copied unchanged; the new file
    replaces the old one only after /image was checked against the greyscale of the old RGB image.
    """
    import os
    h5 = sp["h5"]
    tmp = h5.with_suffix(".gray_tmp.h5")
    with h5py.File(h5, "r") as src:
        if src["image"].ndim == 2:
            return "already greyscale"
        rgb = src["image"][()]
        gray = to_gray(rgb)
        with h5py.File(tmp, "w") as dst:
            dst.attrs.update(dict(src.attrs))
            for k in src:
                if k not in ("image", "image_preview"):
                    src.copy(src[k], dst, k)
            ds = _put(dst, "image", gray, {k: v for k, v in src["image"].attrs.items() if k != "axes"})
            ds.attrs.update({"axes": "YX", "color": "greyscale (ITU-R 601 luma) of the RGB source"})
            _previews(dst, ["image"], preview)
    with h5py.File(tmp, "r") as t, h5py.File(h5, "r") as o:
        assert np.array_equal(t["image"][()], gray) and t["image"].shape == rgb.shape[:2]
        assert set(t.keys()) == set(o.keys())
        assert np.array_equal(t["labeling/labels"][()], o["labeling/labels"][()])
    before = h5.stat().st_size
    os.replace(tmp, h5)
    return f"{before / 1e6:.0f} MB -> {h5.stat().st_size / 1e6:.0f} MB"


def repack(h5, drop=()):
    """Rewrite *h5* without the objects in *drop* (frees their space); all else copied, verified, then swapped in."""
    import os
    h5 = Path(h5)
    tmp = h5.with_suffix(".repack_tmp.h5")
    with h5py.File(h5, "r") as src, h5py.File(tmp, "w") as dst:
        dst.attrs.update(dict(src.attrs))
        for k in src:
            if k in drop:
                continue
            if isinstance(src[k], h5py.Group):
                g = dst.create_group(k); g.attrs.update(dict(src[k].attrs))
                for kk in src[k]:
                    if f"{k}/{kk}" not in drop:
                        src.copy(src[k][kk], g, kk)
            else:
                src.copy(src[k], dst, k)
    expect = []
    with h5py.File(h5, "r") as o:
        o.visit(lambda n: expect.append(n) if not any(n == d or n.startswith(d + "/") for d in drop) else None)
    got = []
    with h5py.File(tmp, "r") as t:
        t.visit(got.append)
    assert sorted(got) == sorted(expect), "repack lost objects"
    before = h5.stat().st_size
    os.replace(tmp, h5)
    return f"{before / 1e6:.0f} MB -> {h5.stat().st_size / 1e6:.0f} MB"


def merge_review(sp, root, name="bad", delete=True):
    """
    Merge a reviewed Labkit file ``labeling/{s}.{name}.labeling`` (exported with ``{s}.{name}.ids.json``) back:

    * all exported ids are removed from /labeling/labels, every non-empty label of the reviewed file is written
      back — under its old id if it was exported, under a new id otherwise (labels drawn new in Labkit); labels
      left empty / deleted in Labkit stay deleted; reviewed pixels never overwrite other fibres (counted)
    * previous labels → /legacy/labels_pre_review; instances, preview and n_instances rebuilt; file repacked
    * ``{s}.tiff.labeling`` and ``{s}.labels.tiff`` re-exported from the h5 (sync signatures updated)
    * the review files are deleted (``delete=False`` keeps them)
    """
    s = sp["specimen"]
    lab_p = sp["dir"] / "labeling" / f"{s}.{name}.labeling"
    ids_p = sp["dir"] / "labeling" / f"{s}.{name}.ids.json"
    with open(ids_p, encoding="utf-8") as f:
        ids = [int(x["id"]) for x in json.load(f)["instances"]]
    with open(lab_p, encoding="utf-8") as f:
        rev = json.load(f)
    with h5py.File(sp["h5"], "a") as hf:
        old = hf["labeling/labels"][()]
        H, W = old.shape
        if rev["interval"]["max"] != [W - 1, H - 1]:
            raise ValueError(f"{lab_p.name}: interval {rev['interval']} does not match labels {old.shape}")
        new = old.astype(np.uint32).copy()
        new[np.isin(new, ids)] = 0
        next_id, kept, added, conflicts = int(old.max()) + 1, [], [], 0
        for k, pts in rev["labels"].items():
            if not pts:
                continue
            a = np.asarray(pts, dtype=np.int64)
            x, y = a[:, 0], a[:, 1]
            if k.isdigit() and int(k) in ids:
                lid = int(k); kept.append(lid)
            else:
                lid = next_id; next_id += 1; added.append((k, lid))
            free = new[y, x] == 0
            conflicts += int(np.count_nonzero(~free))
            new[y[free], x[free]] = lid
        deleted = [i for i in ids if i not in kept]
        changed = int(np.count_nonzero(old != new))
        _put(hf, "legacy/labels_pre_review", old, {"axes": "YX", "description": f"/labeling/labels before merging {lab_p.name}"})
        _put(hf, "labeling/labels", new, {"axes": "YX"})
        _previews(hf, ["labeling/labels"], 2)
        n = write_instances(hf, new, pixel_size=hf.attrs.get("pixel_size_x"))
        hf.attrs["n_instances"] = n
        outside = int(((new > 0) & (hf["mask"][()] == 0)).sum()) if "mask" in hf else 0
    repack(sp["h5"])
    sync_export(sp, root, force=True, entries=("labeling/{s}.tiff.labeling", "labeling/{s}.labels.tiff"))
    if delete:
        lab_p.unlink(); ids_p.unlink()
    return {"replaced": len(kept), "deleted": deleted, "added": added, "changed_px": changed,
            "conflict_px": conflicts, "n_instances": n, "fibre_px_outside_mask": outside,
            "review_files_deleted": delete}

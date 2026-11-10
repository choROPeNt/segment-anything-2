"""
Elliptic Fourier descriptors (EFD) of fibre cross-sections and shape refinement with a low-harmonic prior.

Descriptors (Kuhl & Giardina 1982)
    :func:`mask_contour`, :func:`resample_contour`, :func:`efd`, :func:`efd_normalize`,
    :func:`efd_canonical_sign`, :func:`efd_reconstruct`, :func:`rasterize`, :func:`shape_features`

Refinement of a segmented instance to an N-harmonic curve
    :func:`robust_efd_fit` — contour only: IRLS least-squares Fourier fit with Tukey weights, bridges
                             notches / leaks of the mask (the curve cannot leave the mask's information).
                             Use N >= 6 for correction: kidney-shaped fibres need more than 3 harmonics
                             (reviewed labels of 160-02_01: low-H3 fibres reach IoU 0.953 with H3, 0.987
                             with H6), N = 3 is a compact *descriptor*, not a correction target.
    :func:`shape_prior`, :func:`mahalanobis` — Gaussian prior on the normalised shape features

Curve parameters ``theta`` = [x0, y0, a_1..a_N, b_1..b_N, c_1..c_N, d_1..d_N]:

    x(t) = x0 + Σ a_n cos(2πnt) + b_n sin(2πnt),   y(t) = y0 + Σ c_n cos(2πnt) + d_n sin(2πnt)
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist


# --------------------------------------------------------------------------------------------- contours

def mask_contour(mask, pad=2):
    """Largest external contour of a bool mask as (K, 2) float [x, y], one fixed orientation."""
    m = np.pad(np.asarray(mask, dtype=np.uint8), pad)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) - pad
    if len(c) < 8:
        return None
    x, y = c[:, 0], c[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        c = c[::-1]
    return c


def resample_contour(c, n=256):
    """Resample a closed contour to *n* points equally spaced in arc length."""
    closed = np.vstack([c, c[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1)


def rasterize(pts, shape, offset=0.0):
    """Fill polygon *pts* (pixel coords [x, y]) into a bool canvas of *shape*, sub-pixel accurate."""
    canvas = np.zeros(shape, np.uint8)
    p = np.round((np.asarray(pts) + offset) * 16).astype(np.int32)
    cv2.fillPoly(canvas, [p], 1, lineType=cv2.LINE_8, shift=4)
    return canvas.astype(bool)


# ------------------------------------------------------------------------------------------ descriptors

def efd(contour, order):
    """
    Elliptic Fourier coefficients of a closed polygon.

    Returns (order, 4) [a_n, b_n, c_n, d_n] and the DC term (2,) [A0, C0].
    """
    closed = np.vstack([contour, contour[:1]])
    d = np.diff(closed, axis=0)
    dt = np.linalg.norm(d, axis=1)
    keep = dt > 0
    d, dt, mid = d[keep], dt[keep], (closed[:-1] + closed[1:])[keep] / 2
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]
    phi = 2 * np.pi * t / T
    n = np.arange(1, order + 1)[:, None]
    const = T / (2 * n[:, 0] ** 2 * np.pi ** 2)
    dcos = np.cos(n * phi[1:]) - np.cos(n * phi[:-1])
    dsin = np.sin(n * phi[1:]) - np.sin(n * phi[:-1])
    dxy = d / dt[:, None]
    coeffs = np.stack([const * (dcos @ dxy[:, 0]), const * (dsin @ dxy[:, 0]),
                       const * (dcos @ dxy[:, 1]), const * (dsin @ dxy[:, 1])], axis=1)
    return coeffs, (mid * dt[:, None]).sum(0) / T


def efd_normalize(coeffs):
    """Start-point, rotation and scale normalisation → (normalised coeffs with a1 = 1, b1 = c1 = 0, ψ, E)."""
    a1, b1, c1, d1 = coeffs[0]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    n = np.arange(1, len(coeffs) + 1)
    cs, sn = np.cos(n * theta), np.sin(n * theta)
    a, b, c, d = coeffs.T
    a, b, c, d = a * cs + b * sn, -a * sn + b * cs, c * cs + d * sn, -c * sn + d * cs
    psi = np.arctan2(c[0], a[0])
    E = np.hypot(a[0], c[0])
    cp, sp = np.cos(psi), np.sin(psi)
    norm = np.stack([cp * a + sp * c, cp * b + sp * d, -sp * a + cp * c, -sp * b + cp * d], axis=1) / E
    return norm, psi, E


def efd_canonical_sign(norm):
    """Resolve the 180° ambiguity of the normalisation (even harmonics flip): choose c2 ≥ 0."""
    if norm.shape[0] > 1 and norm[1, 2] < 0:
        norm = norm * np.where(np.arange(1, len(norm) + 1) % 2 == 0, -1, 1)[:, None]
    return norm


def efd_reconstruct(coeffs, n_harm, n_pts=256):
    """Contour (n_pts, 2) from the first *n_harm* harmonics (no DC term)."""
    t = np.linspace(0, 1, n_pts, endpoint=False)
    n = np.arange(1, n_harm + 1)[:, None]
    cs, sn = np.cos(2 * np.pi * n * t), np.sin(2 * np.pi * n * t)
    a, b, c, d = coeffs[:n_harm].T
    return np.stack([a @ cs + b @ sn, c @ cs + d @ sn], axis=1)


def shape_features(norm):
    """Rotation / sign invariant shape features: minor/major axis ratio |d1| and |harmonic n|, n = 2..N."""
    return np.concatenate([[abs(norm[0, 3])], np.linalg.norm(norm[1:], axis=1)])


def shape_prior(norms, areas=None):
    """
    Gaussian shape prior from the normalised EFD coefficients of well-fitting instances.

    Returns dict with mean ``mu`` and inverse covariance ``icov`` of :func:`shape_features`
    (and the 1–99 % ``area_range`` if *areas* is given).
    """
    F = np.stack([shape_features(efd_canonical_sign(n)) for n in norms])
    prior = {"mu": F.mean(0), "icov": np.linalg.pinv(np.cov(F.T))}
    if areas is not None:
        prior["area_range"] = tuple(np.percentile(areas, [1, 99]))
    return prior


def mahalanobis(norm, prior):
    f = shape_features(efd_canonical_sign(norm)) - prior["mu"]
    return float(np.sqrt(f @ prior["icov"] @ f))


# ------------------------------------------------------------------------------------- curve parameters

def _basis(t, n_harm):
    ang = 2 * np.pi * np.outer(t, np.arange(1, n_harm + 1))
    return np.hstack([np.ones((len(t), 1)), np.cos(ang), np.sin(ang)])


def theta_to_coeffs(theta, n_harm):
    """Curve parameters → (EFD coefficients (N, 4), centre (2,))."""
    th = np.asarray(theta, dtype=float)
    a, b, c, d = (th[2 + k * n_harm: 2 + (k + 1) * n_harm] for k in range(4))
    return np.stack([a, b, c, d], axis=1), th[:2]


def coeffs_to_theta(coeffs, dc):
    return np.concatenate([np.asarray(dc, dtype=float), coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]])


def curve_from_theta(theta, n_harm, n_pts=128):
    """Points (n_pts, 2) and tangents (n_pts, 2) of the curve."""
    coeffs, dc = theta_to_coeffs(theta, n_harm)
    t = np.arange(n_pts) / n_pts
    n = np.arange(1, n_harm + 1)[:, None]
    w = 2 * np.pi * n
    cs, sn = np.cos(w * t), np.sin(w * t)
    a, b, c, d = coeffs.T
    pts = np.stack([dc[0] + a @ cs + b @ sn, dc[1] + c @ cs + d @ sn], axis=1)
    tan = np.stack([(-a[:, None] * w * sn + b[:, None] * w * cs).sum(0),
                    (-c[:, None] * w * sn + d[:, None] * w * cs).sum(0)], axis=1)
    return pts, tan


# ------------------------------------------------------------------------------------------ refinement

def robust_efd_fit(contour, n_harm=3, n_iter=15, c=3.0, min_scale=1.0, n_dense=1024, n_pts=256):
    """
    Robust least-squares Fourier fit of a closed contour (IRLS, Tukey weights): points far from the
    current fit (leaks, notches) get weight 0 and every point is re-parametrised to its closest curve
    position, so the low-order curve bridges defects instead of following them.

    Returns theta (curve parameters), the final point weights (n_pts,) and residuals (n_pts,) in px.
    """
    pts = resample_contour(contour, n_pts)
    t = np.arange(len(pts)) / len(pts)
    w = np.ones(len(pts))
    td = np.arange(n_dense) / n_dense
    Bd = _basis(td, n_harm)
    for _ in range(n_iter):
        sw = np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(_basis(t, n_harm) * sw, pts * sw, rcond=None)[0]
        curve = Bd @ coef
        d2 = cdist(pts, curve, "sqeuclidean")
        k = d2.argmin(1)
        t, r = td[k], np.sqrt(d2[np.arange(len(pts)), k])
        scale = max(1.4826 * np.median(r), min_scale)
        u = r / (c * scale)
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
    N = n_harm
    coeffs = np.stack([coef[1:N + 1, 0], coef[N + 1:, 0], coef[1:N + 1, 1], coef[N + 1:, 1]], axis=1)
    return coeffs_to_theta(coeffs, coef[0]), w, r


# --------------------------------------------------------------------------------------- quality / QA

def efd_fit_quality(mask, n_max=5, n_pts=256):
    """
    EFD of one instance mask and how well N = 1..n_max harmonics reproduce it.

    Returns dict with ``iou`` (n_max,) reconstruction IoU per N, ``coeffs`` (n_max, 4), ``dc`` (2,),
    normalised coefficients ``norm`` (canonical sign), orientation ``psi`` and size ``E`` (semi-major axis of the
    first harmonic, px) — or None if the mask has no usable contour.
    """
    c = mask_contour(mask)
    if c is None:
        return None
    coeffs, dc = efd(resample_contour(c, n_pts), n_max)
    h, w = mask.shape
    P = max(h, w) // 2 + 4
    target = np.pad(np.asarray(mask, dtype=bool), P)
    canvas = target.shape
    iou = np.zeros(n_max)
    for N in range(1, n_max + 1):
        rec = rasterize(efd_reconstruct(coeffs, N, n_pts) + dc, canvas, P)
        iou[N - 1] = (rec & target).sum() / max((rec | target).sum(), 1)
    norm, psi, E = efd_normalize(coeffs)
    return {"iou": iou, "coeffs": coeffs, "dc": dc, "norm": efd_canonical_sign(norm), "psi": psi, "E": E}


def contour_defect(mask, n_harm=6, thr=2.0, n_pts=256):
    """
    Local contour defect of an instance: robust N-harmonic fit (:func:`robust_efd_fit`), then the largest
    deviation and the longest contiguous stretch of contour (px) further than *thr* px from the fit.
    Leaks into gaps and notches show up as long stretches; a smooth unusual shape does not.
    """
    c = mask_contour(mask)
    if c is None:
        return {"max_dev": np.nan, "defect_len": np.nan}
    _, _, r = robust_efd_fit(c, n_harm, n_pts=n_pts)
    b = r > thr
    run = 0
    if b.all():
        run = len(b)
    elif b.any():
        b = np.roll(b, -int(np.argmin(b)))
        d = np.diff(np.concatenate([[0], b.astype(int), [0]]))
        run = int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())
    perim = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum()
    return {"max_dev": float(r.max()), "defect_len": run * perim / n_pts}


def fourier_fit_quality(mask, n_max=5, n_iter=8, n_dense=512, n_pts=256):
    """
    Least-squares Fourier fit with optimised parametrisation (every contour point re-assigned to its closest
    curve position, :func:`robust_efd_fit` without outlier rejection) for N = 1..n_max.

    More compact than the arc-length EFD of :func:`efd`: an ellipse needs N = 1 here, whereas its arc-length
    EFD has a clear 3rd harmonic.  Returns dict with ``iou`` (n_max,), ``theta`` (list of curve parameters per
    N) and the normalised coefficients ``norm`` (list per N, canonical sign) — or None without contour.
    """
    c = mask_contour(mask)
    if c is None:
        return None
    h, w = mask.shape
    P = max(h, w) // 2 + 4
    target = np.pad(np.asarray(mask, dtype=bool), P)
    iou, thetas, norms = np.zeros(n_max), [], []
    for N in range(1, n_max + 1):
        th, _, _ = robust_efd_fit(c, N, n_iter=n_iter, c=1e9, n_dense=n_dense, n_pts=n_pts)
        rec = rasterize(curve_from_theta(th, N, n_pts)[0], target.shape, P)
        iou[N - 1] = (rec & target).sum() / max((rec | target).sum(), 1)
        coeffs, _ = theta_to_coeffs(th, N)
        thetas.append(th); norms.append(efd_canonical_sign(efd_normalize(coeffs)[0]))
    return {"iou": iou, "theta": thetas, "norm": norms}


def contour_notches(mask, min_depth=1.5):
    """
    Concavities of an instance from the convexity defects of its contour, deepest-relative first.

    Returns dict with ``notch_depth`` (px) and ``notch_ratio`` (depth / width of the opening) of the concavity
    with the largest depth/width.  Narrow deep notches (segmentation running into a dark spot / crack at the
    fibre edge: depth ≳ 4 px, ratio ≳ 0.25 at 50x) separate from the wide shallow concavity of kidney-shaped
    fibres (≲ 3 px, ≲ 0.2).
    """
    import cv2
    c = mask_contour(mask)
    if c is None:
        return {"notch_depth": 0.0, "notch_ratio": 0.0}
    c = c.astype(np.int32).reshape(-1, 1, 2)
    hull = np.sort(cv2.convexHull(c, returnPoints=False), axis=0)
    best = (0.0, 0.0)
    try:
        dd = cv2.convexityDefects(c, hull)
    except cv2.error:                       # self-intersecting / degenerate contour
        dd = None
    if dd is not None:
        for s, e, _, d in dd[:, 0]:
            depth = d / 256.0
            if depth < min_depth:
                continue
            ratio = depth / max(float(np.linalg.norm(c[s, 0] - c[e, 0])), 1.0)
            if ratio > best[1]:
                best = (depth, ratio)
    return {"notch_depth": best[0], "notch_ratio": best[1]}

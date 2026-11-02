"""
Two-point statistics of a binary phase (e.g. fibres in matrix) inside an arbitrary observation window.

* :func:`s2_window`            — two-point probability S2(h) with window (translation) correction
* :func:`radial_profile`       — radial (optionally directional) average of a centred 2D map
* :func:`integral_range`       — integral range A(R) = ∫_{|h|≤R} C2(h) dh / (φ(1 − φ)), cumulative, unclamped
* :func:`box_fraction_variance` — measured variance of the local phase fraction in square boxes
* :func:`box_variance_from_c2` — the same, predicted exactly from C2 (and asymptotically from A)
* :func:`disk_covariance`      — set covariance of a disk (mean over a radius distribution)
* :func:`c2_from_pcf`          — C2 of non-overlapping disks from the point pair correlation g(r)
* :func:`render_disks`         — binary image of disks (e.g. simulated fibre centres + radii)

Conventions: ``C2(h) = S2(h) − φ²``, ``C2(0) = φ(1 − φ)``, ``C2 → 0`` for |h| → ∞.  Maps are centred:
index ``[R + dy, R + dx]`` is the displacement (dx, dy) in pixels.  Unlike
:func:`sam2.addons.descriptors.s2_descriptor` nothing is periodic and negative C2 (anti-correlation of a
regular packing) is kept.
"""

import cv2
import numpy as np


def _autocorr(a, R):
    H, W = a.shape
    s = (H + R + 1, W + R + 1)                       # zero padding ≥ R → no wrap-around for |h| ≤ R
    F = np.fft.rfft2(a, s=s)
    ac = np.fft.irfft2(F * np.conj(F), s=s)
    return np.roll(ac, (R, R), axis=(0, 1))[:2 * R + 1, :2 * R + 1]


def s2_window(phase, window, rmax_px):
    """
    Two-point probability of *phase* inside *window* (both (H, W); *phase* may be a fraction field in
    [0, 1], e.g. an area-downsampled binary image):

        S2(h) = Σ_x (P·W)(x) (P·W)(x + h) / Σ_x W(x) W(x + h)

    Returns
    -------
    S2 : (2R + 1, 2R + 1) centred map, φ : phase fraction in the window, gamma : (2R + 1, 2R + 1) number of
    pixel pairs per displacement (window set covariance, for weighting / masking)
    """
    W = np.asarray(window, dtype=np.float64)
    P = np.asarray(phase, dtype=np.float64) * W
    R = int(rmax_px)
    gam = np.clip(np.round(_autocorr(W, R)), 0, None)
    num = _autocorr(P, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(gam > 0, num / gam, np.nan)
    return s2, float(P.sum() / W.sum()), gam


def _grid(R):
    d = np.arange(-R, R + 1)
    return d[None, :], d[:, None]                      # dx, dy


def radial_profile(cmap, pixel_size, r, angle_range=None, weights=None):
    """
    Radial average of a centred map at distances *r* (physical units; bins of one pixel around each r).

    *angle_range* (deg0, deg1): only displacements with axial orientation in that sector (0° = x axis).
    *weights*: optional per-displacement weights (e.g. the pair counts from :func:`s2_window`).
    """
    R = cmap.shape[0] // 2
    dx, dy = _grid(R)
    dist = np.hypot(dx, dy) * pixel_size
    w = np.ones_like(cmap) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(cmap) & (w > 0)
    if angle_range is not None:
        a0, a1 = np.mod(np.deg2rad(angle_range), np.pi)
        ang = np.mod(np.arctan2(dy, dx), np.pi) * np.ones_like(cmap)
        ok &= (ang >= a0) & (ang < a1) if a0 < a1 else (ang >= a0) | (ang < a1)
    out = np.full(len(r), np.nan)
    for k, rk in enumerate(np.asarray(r, dtype=float)):
        sel = ok & (np.abs(dist - rk) <= 0.5 * pixel_size)
        if sel.any():
            out[k] = np.sum(cmap[sel] * w[sel]) / np.sum(w[sel])
    return out


def integral_range(c2map, phi, pixel_size, r):
    """
    Cumulative integral range A(R) = ∫_{|h| ≤ R} C2(h) dh / (φ (1 − φ)) for every R in *r* (unclamped:
    negative C2 of a regular packing reduces A).  For large boxes Var[φ_B] → φ (1 − φ) A / |B|; |W| / A is the
    number of independent "units" of the microstructure in the window.
    """
    R = c2map.shape[0] // 2
    dx, dy = _grid(R)
    dist = (np.hypot(dx, dy) * pixel_size).ravel()
    c = np.nan_to_num(c2map).ravel()
    order = np.argsort(dist)
    cum = np.cumsum(c[order]) * pixel_size ** 2 / (phi * (1 - phi))
    idx = np.searchsorted(dist[order], np.asarray(r, dtype=float), side="right") - 1
    return cum[np.clip(idx, 0, None)]


def box_fraction_variance(phase, window, sizes_px, min_cover=0.99):
    """
    Variance of the phase fraction in square boxes of side a (pixels, each in *sizes_px*), boxes on a grid
    fully (≥ *min_cover*) inside the window.  Returns (len,) variance, (len,) number of boxes, (len,) mean.
    """
    P = np.asarray(phase, dtype=np.float64)
    Wm = np.asarray(window, dtype=bool)
    H, W = P.shape
    var, nb, mean = (np.full(len(sizes_px), np.nan) for _ in range(3))
    for k, a in enumerate(sizes_px):
        a = int(a)
        h, w = H // a, W // a
        if h == 0 or w == 0:
            continue
        cov = Wm[:h * a, :w * a].reshape(h, a, w, a).mean(axis=(1, 3))
        frac = P[:h * a, :w * a].reshape(h, a, w, a).mean(axis=(1, 3))
        sel = frac[cov >= min_cover]
        nb[k] = len(sel)
        if len(sel) > 1:
            var[k], mean[k] = sel.var(ddof=1), sel.mean()
    return var, nb, mean


def box_variance_from_c2(c2map, sizes_px):
    """
    Variance of the phase fraction in a square box of side a predicted exactly from C2:

        Var[φ_B] = (1 / a⁴) Σ_h C2(h) (a − |h_x|)(a − |h_y|)        (|h_x|, |h_y| < a)

    Only boxes with a ≤ R (map half-width) are evaluated (NaN otherwise).
    """
    R = c2map.shape[0] // 2
    dx, dy = _grid(R)
    c = np.nan_to_num(c2map)
    out = np.full(len(sizes_px), np.nan)
    for k, a in enumerate(sizes_px):
        a = float(a)
        if a > R + 1:
            continue
        wgt = np.clip(a - np.abs(dx), 0, None) * np.clip(a - np.abs(dy), 0, None)
        out[k] = np.sum(c * wgt) / a ** 4
    return out


def disk_covariance(h, radii):
    """Set covariance γ_D(h) = |D ∩ (D + h)| of a disk, averaged over *radii* → same shape as *h*."""
    h = np.asarray(h, dtype=float)
    out = np.zeros_like(h)
    for rho in np.atleast_1d(radii):
        x = np.clip(h / (2 * rho), 0, 1)
        out += 2 * rho ** 2 * (np.arccos(x) - x * np.sqrt(1 - x ** 2))
    return out / len(np.atleast_1d(radii))


def c2_from_pcf(r_g, g, intensity, radii, pixel_size, rmax_px):
    """
    C2 of a system of non-overlapping disks (germ-grain model) from the centre pair correlation:

        C2(h) = λ γ̄_D(h) + λ² [(g − 1) ∗ γ̄_D](h)

    with γ̄_D the mean disk set covariance.  Links point-pattern descriptors (λ, g; e.g. the LGCP g = exp(k))
    to the phase two-point function.  Radii are subsampled to ≤ 200 values for γ̄_D.

    Returns
    -------
    (2R + 1, 2R + 1) centred C2 map on the pixel grid
    """
    R = int(rmax_px)
    radii = np.atleast_1d(radii)
    if len(radii) > 200:
        radii = np.quantile(radii, np.linspace(0.0025, 0.9975, 200))
    dx, dy = _grid(R)
    dist = np.hypot(dx, dy) * pixel_size
    gam = disk_covariance(dist, radii)
    gm1 = np.interp(dist, np.concatenate([[0.0], r_g]), np.concatenate([[np.nan_to_num(g)[0]], np.nan_to_num(g)]),
                    right=1.0) - 1.0
    s = (4 * R + 1, 4 * R + 1)
    conv = np.fft.irfft2(np.fft.rfft2(gm1, s=s) * np.fft.rfft2(gam, s=s), s=s)[R:3 * R + 1, R:3 * R + 1]
    return intensity * gam + intensity ** 2 * conv * pixel_size ** 2


def render_disks(xy, radii, shape, pixel_size, down=1):
    """
    Fraction image (area coverage per pixel) of disks with centres *xy* (physical units) and *radii*, on a
    grid of *shape* with *pixel_size*; rendered at ``down``-fold resolution and area-averaged.
    """
    H, W = shape
    img = np.zeros((H * down, W * down), dtype=np.uint8)
    scale = 16 * down / pixel_size                     # cv2 sub-pixel shift = 4 bits
    for (x, y), rho in zip(np.asarray(xy, dtype=float), np.atleast_1d(radii)):
        cv2.circle(img, (int(round(x * scale)), int(round(y * scale))), int(round(rho * scale)), 1, -1,
                   lineType=cv2.LINE_8, shift=4)
    if down > 1:
        return cv2.resize(img.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)

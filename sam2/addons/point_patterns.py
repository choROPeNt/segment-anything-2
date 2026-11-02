"""
Second-order statistics of planar point patterns (e.g. fibre centroids) in an arbitrary observation window
given as a binary mask, plus a log-Gaussian Cox process (LGCP) model.

* :func:`window_mask_from_image` — specimen region from a grey image (when no ROI mask exists)
* :class:`MaskWindow`      — window from a binary mask: area, containment, set covariance γ_W(h), boundary
                             distance, uniform sampling
* :func:`k_function`       — Ripley's K (translation edge correction, optional direction sector)
* :func:`l_function`       — Besag's L = sqrt(K / π)
* :func:`pair_correlation` — pair correlation function g(r) (Epanechnikov kernel, translation correction)
* :func:`nn_function`      — nearest-neighbour distance distribution G(r) (border correction)
* :func:`envelope`         — pointwise simulation envelopes and global MAD test
* :func:`quadrat_dispersion` — index of dispersion Var/mean of quadrat counts vs. quadrat size
* :func:`lgcp_pcf`, :func:`lgcp_k`, :func:`fit_lgcp_mincontrast`, :func:`simulate_lgcp` — LGCP
* :func:`simulate_hardcore_cox` — hard-core (Gibbs) process, optionally modulated by a field
* :func:`excursion_log_field` — two-phase field (bundles / resin pockets) from a Gaussian field
* :func:`boundary_profile` — relative intensity vs. distance to the window boundary
* :func:`fit_by_simulation` — simulation-based minimum contrast (grid search)
* :func:`kernel_intensity` — edge-corrected Gaussian kernel estimate of the intensity λ(x)

All coordinates are ``[x, y]`` in physical units (``pixel_size`` × pixel index of the mask).
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class MaskWindow:
    """
    Observation window defined by a binary mask.

    Parameters
    ----------
    mask       : (H, W) bool
    pixel_size : physical size of one mask pixel
    """

    def __init__(self, mask, pixel_size=1.0):
        self.mask = np.asarray(mask, dtype=bool)
        self.pixel_size = float(pixel_size)
        self.area = float(self.mask.sum()) * self.pixel_size ** 2
        self._cov = None          # cropped set covariance (pixel² counts), centred at [R, R]
        self._cov_r = 0
        self._bdist = None

    @classmethod
    def from_mask(cls, mask, pixel_size=1.0, down=1):
        """Window from a full-resolution mask, block-averaged by *down* (pixel kept if ≥ 50 % covered)."""
        mask = np.asarray(mask, dtype=bool)
        if down > 1:
            h, w = mask.shape[0] // down, mask.shape[1] // down
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5
        return cls(mask, pixel_size * down)

    def eroded(self, dist):
        """
        Window shrunk by *dist* (physical units).  Use the fibre radius when the mask hugs the object
        outlines: centroids cannot lie in that outer band, which otherwise biases λ low and g(r) high.
        """
        if dist <= 0:
            return MaskWindow(self.mask, self.pixel_size)
        d = np.asarray(distance_transform_edt(np.pad(self.mask, 1)))[1:-1, 1:-1] * self.pixel_size
        return MaskWindow(d > dist, self.pixel_size)

    @property
    def shape(self):
        return self.mask.shape

    def _index(self, xy):
        ij = np.floor(np.asarray(xy, dtype=float) / self.pixel_size).astype(int)
        return ij[:, 0], ij[:, 1]           # ix, iy

    def contains(self, xy):
        """(N,) bool: point inside the window."""
        ix, iy = self._index(xy)
        H, W = self.mask.shape
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        out = np.zeros(len(ix), dtype=bool)
        out[ok] = self.mask[iy[ok], ix[ok]]
        return out

    def set_covariance(self, rmax):
        """
        Compute (and cache) the set covariance γ_W(h) = |W ∩ (W + h)| for |h| ≤ *rmax* via FFT
        autocorrelation of the mask.
        """
        R = int(np.ceil(rmax / self.pixel_size)) + 1
        if self._cov is not None and self._cov_r >= R:
            return
        H, W = self.mask.shape
        m = self.mask.astype(np.float64)
        F = np.fft.rfft2(m, s=(H + R + 1, W + R + 1))       # zero padding ≥ R → no wrap-around for |h| ≤ R
        ac = np.fft.irfft2(F * np.conj(F), s=(H + R + 1, W + R + 1))
        ac = np.roll(ac, (R, R), axis=(0, 1))[:2 * R + 1, :2 * R + 1]
        self._cov = np.clip(np.round(ac), 0, None)
        self._cov_r = R

    def gamma(self, dxy):
        """Set covariance γ_W at displacement vectors *dxy* (N, 2) → (N,) area (physical units²)."""
        dxy = np.asarray(dxy, dtype=float)
        rmax = np.abs(dxy).max() * np.sqrt(2) if len(dxy) else 0.0
        self.set_covariance(rmax)
        R = self._cov_r
        ix = np.clip(np.round(dxy[:, 0] / self.pixel_size).astype(int), -R, R) + R
        iy = np.clip(np.round(dxy[:, 1] / self.pixel_size).astype(int), -R, R) + R
        return self._cov[iy, ix] * self.pixel_size ** 2

    def max_reliable_r(self, rmax, frac=0.2, n_angles=36):
        """
        Largest r ≤ *rmax* for which γ_W(h) ≥ *frac* · |W| in every direction, i.e. at least that share of
        the window still contributes pairs at distance r.  Beyond it the translation correction divides by
        tiny overlaps (e.g. pairs across a thin tape) and K / g get very noisy.
        """
        self.set_covariance(rmax)
        R = self._cov_r
        rs = np.arange(1, int(rmax / self.pixel_size) + 1)
        th = np.linspace(0, np.pi, n_angles, endpoint=False)
        ix = np.clip(np.round(rs[:, None] * np.cos(th)).astype(int), -R, R) + R
        iy = np.clip(np.round(rs[:, None] * np.sin(th)).astype(int), -R, R) + R
        ok = (self._cov[iy, ix] * self.pixel_size ** 2 / self.area).min(1) >= frac
        if ok.all():
            return float(rmax)
        return float(max(np.argmin(ok), 1) * self.pixel_size)

    def boundary_distance(self, xy):
        """(N,) distance of each point to the window boundary (0 outside)."""
        if self._bdist is None:
            self._bdist = np.asarray(distance_transform_edt(np.pad(self.mask, 1)))[1:-1, 1:-1] * self.pixel_size
        ix, iy = self._index(xy)
        H, W = self.mask.shape
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        out = np.zeros(len(ix))
        out[ok] = self._bdist[iy[ok], ix[ok]]
        return out

    def sample_uniform(self, n, rng=None):
        """*n* independent uniform points in the window (binomial process) → (n, 2)."""
        rng = np.random.default_rng(rng)
        iy, ix = np.nonzero(self.mask)
        k = rng.integers(0, len(ix), n)
        return np.column_stack([ix[k] + rng.random(n), iy[k] + rng.random(n)]) * self.pixel_size


def window_mask_from_image(image, threshold=None, close=15, fill_holes=True):
    """
    Specimen region (e.g. a tape / roving in embedding resin on a dark background) from a grey image:
    threshold (Otsu if None) → morphological closing (*close* px) → largest connected component →
    holes filled (pores / resin pockets inside stay part of the window).
    """
    from scipy.ndimage import binary_fill_holes

    im = np.asarray(image)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2GRAY)
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if threshold is None:
        threshold, _ = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (im > threshold).astype(np.uint8)
    if close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = m.astype(bool)
    return binary_fill_holes(m) if fill_holes else m


# --------------------------------------------------------------------------------------------------
# summary functions
# --------------------------------------------------------------------------------------------------

def _pairs(xy, rmax, angle_range=None):
    """Unordered pairs with distance ≤ rmax → displacement (P, 2), distance (P,)."""
    tree = cKDTree(xy)
    ij = tree.query_pairs(rmax, output_type="ndarray")
    if len(ij) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    dxy = xy[ij[:, 1]] - xy[ij[:, 0]]
    if angle_range is not None:
        a0, a1 = np.deg2rad(angle_range)
        ang = np.mod(np.arctan2(dxy[:, 1], dxy[:, 0]), np.pi)     # axial orientation in [0, π)
        a0, a1 = np.mod(a0, np.pi), np.mod(a1, np.pi)
        sel = (ang >= a0) & (ang < a1) if a0 < a1 else (ang >= a0) | (ang < a1)
        dxy = dxy[sel]
    return dxy, np.hypot(dxy[:, 0], dxy[:, 1])


def _sector_fraction(angle_range):
    if angle_range is None:
        return 1.0
    a0, a1 = np.mod(np.deg2rad(angle_range), np.pi)
    width = a1 - a0 if a0 < a1 else np.pi - a0 + a1
    return width / np.pi


def k_function(xy, window, r, angle_range=None):
    """
    Ripley's K with translation edge correction:

        K(r) = |W|² / (n (n − 1)) · Σ_{i≠j} 1{d_ij ≤ r} / γ_W(x_j − x_i)

    Parameters
    ----------
    xy          : (n, 2) points inside *window*
    window      : :class:`MaskWindow`
    r           : (R,) distances
    angle_range : optional (deg0, deg1) axial sector of pair orientations (0° = x axis); the result is
                  rescaled by π / sector width, so it equals the isotropic K for an isotropic pattern

    Returns
    -------
    (R,) K(r);  CSR → π r²
    """
    xy = np.asarray(xy, dtype=float)
    r = np.asarray(r, dtype=float)
    n = len(xy)
    dxy, d = _pairs(xy, r.max(), angle_range)
    w = 1.0 / np.maximum(window.gamma(dxy), 1e-12)
    order = np.argsort(d)
    cw = np.concatenate([[0.0], np.cumsum(w[order])])
    K = 2.0 * cw[np.searchsorted(d[order], r, side="right")]
    return K * window.area ** 2 / (n * (n - 1)) / _sector_fraction(angle_range)


def l_function(K):
    """Besag's L(r) = sqrt(K(r) / π);  CSR → r."""
    return np.sqrt(np.maximum(K, 0) / np.pi)


def pair_correlation(xy, window, r, bandwidth=None):
    """
    Pair correlation function with Epanechnikov kernel and translation edge correction:

        g(r) = |W|² / (n (n − 1)) · Σ_{i≠j} k_h(r − d_ij) / (2π r γ_W(x_j − x_i))

    *bandwidth* h (kernel half-width) defaults to Stoyan's rule 0.15 / sqrt(λ).  CSR → 1.
    """
    xy = np.asarray(xy, dtype=float)
    r = np.asarray(r, dtype=float)
    n = len(xy)
    if bandwidth is None:
        bandwidth = 0.15 / np.sqrt(n / window.area)
    h = float(bandwidth)
    dxy, d = _pairs(xy, r.max() + h)
    w = 1.0 / np.maximum(window.gamma(dxy), 1e-12)
    order = np.argsort(d)
    d, w = d[order], w[order]
    g = np.zeros(len(r))
    for k, rk in enumerate(r):
        lo, hi = np.searchsorted(d, [rk - h, rk + h])
        t = (rk - d[lo:hi]) / h
        g[k] = np.sum(0.75 / h * (1 - t ** 2) * w[lo:hi])
    g *= 2.0 * window.area ** 2 / (n * (n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(r > 0, g / (2 * np.pi * r), np.nan)
    return g


def nn_distances(xy):
    """(n,) distance of every point to its nearest neighbour."""
    d, _ = cKDTree(xy).query(xy, k=2)
    return d[:, 1]


def nn_function(xy, window, r):
    """
    Nearest-neighbour distance distribution G(r), border (reduced-sample) correction: only points farther
    than r from the window boundary contribute at r.  CSR → 1 − exp(−λ π r²).
    """
    xy = np.asarray(xy, dtype=float)
    dnn = nn_distances(xy)
    b = window.boundary_distance(xy)
    G = np.full(len(r), np.nan)
    for k, rk in enumerate(r):
        sel = b >= rk
        if sel.any():
            G[k] = np.mean(dnn[sel] <= rk)
    return G


def envelope(obs, sims, r=None, r_range=None, alpha=0.05):
    """
    Pointwise envelopes and the global maximum-absolute-deviation (MAD) test.

    Parameters
    ----------
    obs     : (R,) observed summary function
    sims    : (S, R) summary functions of S simulations of the null model
    r       : (R,) distances (only needed with *r_range*)
    r_range : (rmin, rmax) range used for the MAD test statistic
    alpha   : pointwise level of the envelope (quantiles alpha/2, 1 − alpha/2)

    Returns
    -------
    dict(lo, hi, mean, p_mad, t_obs): p_mad = rank-based Monte-Carlo p-value of
    T = max_r |S(r) − mean(r)| (small p → null model rejected)
    """
    sims = np.asarray(sims, dtype=float)
    obs = np.asarray(obs, dtype=float)
    lo, hi = np.nanquantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
    mean = np.nanmean(sims, axis=0)
    sel = np.ones(len(obs), dtype=bool)
    if r_range is not None:
        r = np.asarray(r)
        sel = (r >= r_range[0]) & (r <= r_range[1])
    sel &= np.isfinite(obs) & np.all(np.isfinite(sims), axis=0)
    t_obs = np.max(np.abs(obs[sel] - mean[sel]))
    t_sim = np.max(np.abs(sims[:, sel] - mean[sel]), axis=1)
    p = (1 + np.sum(t_sim >= t_obs)) / (1 + len(sims))
    return {"lo": lo, "hi": hi, "mean": mean, "p_mad": float(p), "t_obs": float(t_obs)}


def quadrat_dispersion(xy, window, sizes, min_cover=0.99):
    """
    Index of dispersion D = Var(N) / E(N) of point counts in square quadrats of side *a* (each size in
    *sizes*), using only grid cells covered ≥ *min_cover* by the window.

    CSR → 1, regular (hard core) → < 1, clustered → > 1 and growing with a.

    Returns
    -------
    (len(sizes),) D, (len(sizes),) number of quadrats used
    """
    xy = np.asarray(xy, dtype=float)
    H, W = window.shape
    D, nq = np.full(len(sizes), np.nan), np.zeros(len(sizes), dtype=int)
    for k, a in enumerate(sizes):
        c = max(1, int(round(a / window.pixel_size)))           # quadrat side in window pixels
        h, w = H // c, W // c
        cover = window.mask[:h * c, :w * c].reshape(h, c, w, c).mean(axis=(1, 3))
        ix = np.floor(xy[:, 0] / (c * window.pixel_size)).astype(int)
        iy = np.floor(xy[:, 1] / (c * window.pixel_size)).astype(int)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        counts = np.zeros((h, w))
        np.add.at(counts, (iy[ok], ix[ok]), 1.0)
        sel = counts[cover >= min_cover]
        nq[k] = len(sel)
        if len(sel) > 1 and sel.mean() > 0:
            D[k] = sel.var(ddof=1) / sel.mean()
    return D, nq


# --------------------------------------------------------------------------------------------------
# log-Gaussian Cox process
# --------------------------------------------------------------------------------------------------

def _corr(r, scale, model):
    r = np.asarray(r, dtype=float) / scale
    if model == "exponential":
        return np.exp(-r)
    if model == "gaussian":
        return np.exp(-r ** 2)
    raise ValueError(f"unknown covariance model {model!r} (exponential | gaussian)")


def lgcp_pcf(r, sigma2, scale, model="exponential"):
    """Pair correlation of an LGCP: g(r) = exp(σ² ρ(r / ℓ))."""
    return np.exp(sigma2 * _corr(r, scale, model))


def lgcp_k(r, sigma2, scale, model="exponential", n_grid=2000):
    """K function of an LGCP: K(r) = 2π ∫₀ʳ s g(s) ds (numerical)."""
    r = np.asarray(r, dtype=float)
    s = np.linspace(0, r.max(), n_grid)
    f = 2 * np.pi * s * lgcp_pcf(s, sigma2, scale, model)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(s))])
    return np.interp(r, s, cum)


def fit_lgcp_mincontrast(r, obs, stat="g", r_range=None, q=None, model="exponential", x0=None, bounds=None,
                         g_ref=None):
    """
    Minimum-contrast fit of an LGCP (Diggle 2003) to an observed g or K function:

        min_{σ², ℓ}  Σ_r ( T_obs(r)^q − T_model(r)^q )²

    Parameters
    ----------
    r, obs  : distances and observed summary function
    stat    : "g" (pair correlation) or "K"
    r_range : (rmin, rmax) fitting range — exclude the short-range hard core of touching fibres
    q       : power (default 1 for g, 1/4 for K)
    model   : covariance model of the Gaussian field, "exponential" or "gaussian"
    x0      : start (σ², ℓ); default (0.1, r_range mean)
    bounds  : ((σ²_min, ℓ_min), (σ²_max, ℓ_max)); default ℓ ≤ 5 · rmax — beyond the fitting range ℓ is
              not identifiable
    g_ref   : optional (s, g(s)) reference pair correlation, e.g. of a homogeneous hard-core process at the
              same intensity (:func:`simulate_hardcore_cox` with no field).  The model is then the product
              g(r) = g_ref(r) · exp(σ² ρ(r / ℓ)) (hard-core Cox approximation), which describes the packing
              shells as well and can be fitted from about one fibre diameter on.

    Returns
    -------
    dict(sigma2, scale, model, stat, cost)
    """
    r = np.asarray(r, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if r_range is None:
        r_range = (r[r > 0].min(), r.max())
    sel = (r >= r_range[0]) & (r <= r_range[1]) & np.isfinite(obs)
    rr, oo = r[sel], obs[sel]
    if q is None:
        q = 1.0 if stat == "g" else 0.25
    if x0 is None:
        x0 = (0.1, 0.5 * (r_range[0] + r_range[1]))

    if g_ref is not None:
        s_ref, g_ref_v = (np.asarray(v, dtype=float) for v in g_ref)
        s_ref, g_ref_v = np.concatenate([[0.0], s_ref]), np.concatenate([[0.0], np.nan_to_num(g_ref_v)])
        s_grid = np.linspace(0, rr.max(), 2000)
        gr_grid = np.interp(s_grid, s_ref, g_ref_v, right=1.0)

    def model_fn(p):
        s2, ell = np.exp(p)
        if g_ref is None:
            return lgcp_pcf(rr, s2, ell, model) if stat == "g" else lgcp_k(rr, s2, ell, model)
        g = gr_grid * lgcp_pcf(s_grid, s2, ell, model)
        if stat == "g":
            return np.interp(rr, s_grid, g)
        f = 2 * np.pi * s_grid * g
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(s_grid))])
        return np.interp(rr, s_grid, cum)

    def resid(p):
        return np.maximum(oo, 0) ** q - np.maximum(model_fn(p), 0) ** q

    if bounds is None:
        bounds = ((1e-5, 0.05 * r_range[0]), (10.0, 5 * r_range[1]))
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res = least_squares(resid, np.clip(np.log(np.maximum(x0, 1e-6)), lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
    s2, ell = np.exp(res.x)
    at_bound = bool(np.any(np.isclose(res.x, lo, atol=1e-3) | np.isclose(res.x, hi, atol=1e-3)))
    return {"sigma2": float(s2), "scale": float(ell), "model": model, "stat": stat, "cost": float(res.cost),
            "at_bound": at_bound}


def gaussian_random_field(shape, pixel_size, sigma2, scale, model="exponential", rng=None):
    """Stationary zero-mean Gaussian field on a grid by circulant embedding (FFT) → (H, W)."""
    rng = np.random.default_rng(rng)
    H, W = shape
    pad = int(np.ceil(4 * scale / pixel_size))
    Hp, Wp = H + pad, W + pad
    dy = np.minimum(np.arange(Hp), Hp - np.arange(Hp)) * pixel_size
    dx = np.minimum(np.arange(Wp), Wp - np.arange(Wp)) * pixel_size
    c = sigma2 * _corr(np.hypot(dy[:, None], dx[None, :]), scale, model)
    eig = np.clip(np.fft.fft2(c).real, 0, None)
    z = np.fft.ifft2(np.sqrt(eig) * np.fft.fft2(rng.standard_normal((Hp, Wp)))).real
    return z[:H, :W]


def simulate_lgcp(window, intensity, sigma2, scale, model="exponential", rng=None, return_field=False):
    """
    One LGCP realisation in *window* with mean intensity *intensity*:

        Λ(x) = λ · exp(Z(x) − σ² / 2),   Z ~ GRF(0, σ² ρ(· / ℓ))

    Points are Poisson(Λ · pixel area) per window pixel, placed uniformly within the pixel.
    """
    rng = np.random.default_rng(rng)
    z = gaussian_random_field(window.shape, window.pixel_size, sigma2, scale, model, rng)
    lam = intensity * np.exp(z - sigma2 / 2) * window.mask
    counts = rng.poisson(lam * window.pixel_size ** 2)
    iy, ix = np.nonzero(counts)
    rep = counts[iy, ix]
    iy, ix = np.repeat(iy, rep), np.repeat(ix, rep)
    xy = np.column_stack([ix + rng.random(len(ix)), iy + rng.random(len(iy))]) * window.pixel_size
    return (xy, lam) if return_field else xy


def kernel_intensity(xy, window, sigma):
    """
    Gaussian kernel intensity estimate λ(x) on the window grid with edge correction
    (division by the kernel mass inside the window); NaN outside the window.
    """
    H, W = window.shape
    ix, iy = window._index(xy)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    counts = np.zeros((H, W))
    np.add.at(counts, (iy[ok], ix[ok]), 1.0)
    s = sigma / window.pixel_size
    num = gaussian_filter(counts, s, mode="constant")
    den = gaussian_filter(window.mask.astype(float), s, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = num / den / window.pixel_size ** 2
    lam[~window.mask] = np.nan
    return lam


def simulate_hardcore_cox(window, n, hard_core, log_field=None, field_pixel_size=None, n_sweeps=20, step=None,
                          rng=None, init=None, radii=None, max_grow_sweeps=200):
    """
    Hard-core Cox process with a fixed number of points: density ∝ Π_i exp(Z(x_i)) · 1{all |x_i − x_j| ≥ d_ij}.

    * monodisperse: d_ij = *hard_core*
    * polydisperse (*radii* given, e.g. the measured fibre radii): d_ij = *hard_core* · (r_i + r_j), i.e.
      *hard_core* is the contact factor (≈ 0.9–1 for touching fibres)

    With *log_field* = None it is a homogeneous hard-core process; with a field Z it is a modulated hard-core
    process (Z = LGCP field → smooth clustering; Z = 0 / −inf on an excursion set of a Gaussian field → fibre
    bundles and resin pockets, see :func:`excursion_log_field`).

    Sampling: random sequential adsorption (candidates ∝ exp(Z)) with all distances scaled by the largest
    factor s ≤ 1 that still places all n points, then Metropolis displacement moves (uniform in ±*step*,
    default d / 2, acceptance min(1, exp(Z(new) − Z(old)))).  Overlaps left from the scaled RSA may only
    shrink (a move must not bring an overlapping pair closer, nor create a new overlap), so they resolve
    one by one ("growing disks", for packings beyond RSA jamming); then *n_sweeps* further sweeps equilibrate.

    Parameters
    ----------
    window           : :class:`MaskWindow`
    n                : number of points
    hard_core        : minimum distance d, or contact factor with *radii*
    log_field        : optional (h, w) log-activity Z on a grid covering the window (−inf allowed)
    field_pixel_size : grid spacing of *log_field* (default: window pixel size)
    init             : optional (n, 2) start configuration instead of RSA
    radii            : optional (n,) radii (polydisperse hard core)
    max_grow_sweeps  : give up growing after that many sweeps (warning; the result has a smaller hard core)

    Returns
    -------
    (n, 2) points
    """
    import warnings

    rng = np.random.default_rng(rng)
    if radii is None:
        rad = np.full(n, 0.5 * float(hard_core))
    else:
        rad = float(hard_core) * np.asarray(radii, dtype=float)
        if len(rad) != n:
            raise ValueError("radii must have length n")
    rad_l = rad.tolist()
    d_max = 2 * rad.max()
    mask, ps = window.mask, window.pixel_size
    H, W = mask.shape
    fps = ps if field_pixel_size is None else float(field_pixel_size)
    z = None if log_field is None else np.asarray(log_field, dtype=float)
    if z is not None:
        fh, fw = z.shape

    def zval(x, y):
        if z is None:
            return 0.0
        return z[min(int(y / fps), fh - 1), min(int(x / fps), fw - 1)]

    def inside(x, y):
        if x < 0 or y < 0:
            return False
        ix, iy = int(x / ps), int(y / ps)
        return ix < W and iy < H and mask[iy, ix]

    cs = d_max                                    # cell size ≥ max. d_ij → neighbours in the 3 × 3 cells
    state = {"s": 1.0}
    cells, pts = {}, []

    def conflict(x, y, ri, skip=-1):
        # skip ≥ 0: moving point *skip*; an already overlapping pair may get closer only if it stays apart
        # by at least its old distance (overlaps shrink monotonically)
        cx, cy = int(x // cs), int(y // cs)
        s = state["s"]
        if skip >= 0:
            x0, y0 = pts[skip]
        for i in range(cx - 1, cx + 2):
            for j in range(cy - 1, cy + 2):
                for k in cells.get((i, j), ()):
                    if k != skip:
                        dx, dy = pts[k][0] - x, pts[k][1] - y
                        dd = s * (ri + rad_l[k])
                        d2 = dx * dx + dy * dy
                        if d2 < dd * dd:
                            if skip < 0:
                                return True
                            ox, oy = pts[k][0] - x0, pts[k][1] - y0
                            if d2 < ox * ox + oy * oy:
                                return True
        return False

    def min_ratio():
        P = np.array(pts)
        ij = cKDTree(P).query_pairs(d_max, output_type="ndarray")
        if len(ij) == 0:
            return 1.0
        dist = np.linalg.norm(P[ij[:, 0]] - P[ij[:, 1]], axis=1)
        return float(min(1.0, (dist / (rad[ij[:, 0]] + rad[ij[:, 1]])).min()))

    def rsa(s):
        cells.clear(); pts.clear()
        state["s"] = s
        iy, ix = np.nonzero(mask)
        w = np.ones(len(ix))
        if z is not None:
            w = np.exp([zval((a + 0.5) * ps, (b + 0.5) * ps) for a, b in zip(ix, iy)])
        w /= w.sum()
        tries = 0
        while len(pts) < n:
            m = 4 * (n - len(pts)) + 100
            k = rng.choice(len(ix), m, p=w)
            cand = np.column_stack([ix[k] + rng.random(m), iy[k] + rng.random(m)]) * ps
            for x, y in cand:
                if not conflict(x, y, rad_l[len(pts)]):
                    cells.setdefault((int(x // cs), int(y // cs)), []).append(len(pts))
                    pts.append([x, y])
                    if len(pts) == n:
                        return True
            tries += m
            if tries > 30 * n:
                return False
        return True

    if init is not None:
        for k, (x, y) in enumerate(np.asarray(init, dtype=float)):
            cells.setdefault((int(x // cs), int(y // cs)), []).append(k)
            pts.append([x, y])
    else:
        s0 = 1.0
        while not rsa(s0):
            s0 *= 0.9
            if s0 < 0.2:
                raise RuntimeError(f"cannot place {n} points in the window (support of the field too small)")
    state["s"] = 1.0                              # target for all moves from now on

    step = 0.5 * float(np.median(2 * rad)) if step is None else float(step)

    def sweep():
        who = rng.integers(0, n, n)
        dxy = (rng.random((n, 2)) * 2 - 1) * step
        u = np.log(rng.random(n))
        for t in range(n):
            k = who[t]
            x0, y0 = pts[k]
            x, y = x0 + dxy[t, 0], y0 + dxy[t, 1]
            if not inside(x, y):
                continue
            if z is not None and u[t] > zval(x, y) - zval(x0, y0):
                continue
            if conflict(x, y, rad_l[k], skip=k):
                continue
            c0, c1 = (int(x0 // cs), int(y0 // cs)), (int(x // cs), int(y // cs))
            if c0 != c1:
                cells[c0].remove(k)
                cells.setdefault(c1, []).append(k)
            pts[k] = [x, y]

    g, ratio = 0, min_ratio()
    while ratio < 1.0 and g < max_grow_sweeps:
        sweep()
        ratio = min_ratio() if g % 5 == 4 else ratio
        g += 1
    ratio = min_ratio()
    if ratio < 1.0:
        warnings.warn(f"hard core reached only {ratio:.3f} of its target in {max_grow_sweeps} sweeps "
                      f"(packing too dense)")
    for _ in range(n_sweeps):
        sweep()
    return np.array(pts)


def excursion_log_field(field, area_fraction, matrix_activity=0.0):
    """
    Two-phase log-activity from a Gaussian field: the dense phase (log-activity 0, e.g. fibre bundles) is the
    excursion set {Z > u}, the rest gets log(*matrix_activity*) (0 → −inf: no points, e.g. resin pockets).

    * scalar *area_fraction* p: u = (1 − p) quantile of *field* → exactly that share is dense
    * array *area_fraction* p(x) (same shape as *field*): inhomogeneous coverage, P(dense at x) = p(x), with
      u(x) = Φ⁻¹(1 − p(x)) on the standardised field (e.g. resin-rich surface layer of a tape)

    Returns
    -------
    log-activity (same shape as *field*), threshold u (scalar or array)
    """
    from scipy.stats import norm

    field = np.asarray(field, dtype=float)
    if np.ndim(area_fraction) == 0:
        u = np.quantile(field, 1 - area_fraction)
        dense = field > u
    else:
        p = np.clip(np.asarray(area_fraction, dtype=float), 1e-6, 1 - 1e-6)
        u = norm.ppf(1 - p)
        dense = (field - field.mean()) / field.std() > u
    low = np.log(matrix_activity) if matrix_activity > 0 else -np.inf
    return np.where(dense, 0.0, low), u


def boundary_profile(xy, window, edges):
    """
    Relative intensity λ(b) / λ vs. distance b to the window boundary, in the bins *edges*
    (e.g. resin-rich surface layer).  Returns bin centres, (len(edges) − 1,) ratios.
    """
    xy = np.asarray(xy, dtype=float)
    if window._bdist is None:
        window.boundary_distance(xy[:1])
    bd = window._bdist[window.mask]
    bp = window.boundary_distance(xy)
    lam = len(xy) / window.area
    edges = np.asarray(edges, dtype=float)
    area = np.histogram(bd, edges)[0] * window.pixel_size ** 2
    cnt = np.histogram(bp, edges)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cnt / area / lam
    finite = np.where(np.isfinite(edges[1:]), edges[1:], edges[:-1] + (edges[-2] - edges[-3] if len(edges) > 2 else 1))
    return 0.5 * (edges[:-1] + finite), rho


def fit_by_simulation(simulate, summary, obs, grid, n_rep=2, sel=None, weights=None, rng=None, verbose=False):
    """
    Simulation-based minimum contrast for models without a closed-form summary function:
    for every parameter set in *grid*, average *summary* over *n_rep* realisations of *simulate* and compute

        cost = Σ w · ( obs − mean_sim )² / Σ w     (over *sel*; w = *weights*, default 1)

    *summary* may concatenate several summary functions (e.g. L(r) − r and log dispersion); use *weights*
    (e.g. 1 / (block length · variance under a reference model)) to balance them.

    Parameters
    ----------
    simulate : callable(params: dict, rng) → (n, 2) points
    summary  : callable(points) → (R,) summary function (e.g. L(r) − r)
    obs      : (R,) observed summary function
    grid     : iterable of parameter dicts
    sel      : optional (R,) bool, r values used in the contrast

    Returns
    -------
    list of dict(params..., cost, summary=(R,) mean simulated summary), sorted by cost
    """
    rng = np.random.default_rng(rng)
    obs = np.asarray(obs, dtype=float)
    sel = np.isfinite(obs) if sel is None else (np.asarray(sel) & np.isfinite(obs))
    w = np.ones(len(obs)) if weights is None else np.asarray(weights, dtype=float)
    out = []
    for params in grid:
        m = np.nanmean([summary(simulate(params, rng)) for _ in range(n_rep)], axis=0)
        ok = sel & np.isfinite(m)
        cost = float(np.sum(w[ok] * (obs[ok] - m[ok]) ** 2) / np.sum(w[ok]))
        out.append({**params, "cost": cost, "summary": m})
        if verbose:
            print(params, f"cost {cost:.4g}")
    return sorted(out, key=lambda o: o["cost"])

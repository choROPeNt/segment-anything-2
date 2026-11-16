"""
Group comparison of per-instance descriptors (e.g. fibre shape) when instances are nested in images.

Instances of one image are not independent, so the image (panorama) is the unit of replication:

* :func:`nested_variance`   - exact split of the total sum of squares over a hierarchy of grouping levels
* :func:`effect_size`       - |mean difference| / pooled instance std (Cohen's d)
* :func:`wasserstein_scaled`- W1 distance between two samples, divided by a reference std
* :func:`tost_welch`        - two one-sided t-tests (Schuirmann) on group means, Welch standard error
* :func:`classifier_test`   - balanced accuracy of predicting a label from descriptors, leave-one-group-out
"""

import numpy as np
import pandas as pd
from scipy import stats


def nested_variance(df, value, levels, weights=None):
    """
    Split ``SS_tot`` of column *value* over nested *levels* (outermost first).

    ``SS_tot = sum_k SS_k + SS_within``, where ``SS_k`` is the scatter of the level-k cell means around the
    means of their parent cells, weighted with the cell sizes, and ``SS_within`` the scatter of the instances
    around the innermost cell means. With *weights* (column name, e.g. a triangle area) every instance counts
    with its weight, cell means are weighted means. Returns ``{level: share}`` plus ``"within"`` (sum 1).
    """
    x = df[value].to_numpy(float)
    w = np.ones_like(x) if weights is None else df[weights].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w)
    d = df.loc[ok, list(levels)].copy()
    d["_x"], d["_w"] = x[ok], w[ok]
    d["_xw"] = d["_x"] * d["_w"]
    wmean = lambda keys: d.groupby(keys)["_xw"].transform("sum") / d.groupby(keys)["_w"].transform("sum")  # noqa
    parent = (d["_xw"].sum() / d["_w"].sum())
    tot = (d["_w"] * (d["_x"] - parent) ** 2).sum()
    out, keys = {}, []
    for lv in levels:
        keys.append(lv)
        cell = wmean(keys)
        out[lv] = float((d["_w"] * (cell - parent) ** 2).sum() / tot)
        parent = cell
    out["within"] = float((d["_w"] * (d["_x"] - parent) ** 2).sum() / tot)
    return out


def effect_size(a, b, s=None):
    """|mean(a) - mean(b)| / s, with s the std of the pooled sample if not given."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    s = np.std(np.concatenate([a, b]), ddof=1) if s is None else s
    return float(abs(a.mean() - b.mean()) / s)


def wasserstein_scaled(a, b, s, wa=None, wb=None):
    """W1(a, b) / s (unit-free), optionally with sample weights *wa*, *wb*."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ka, kb = np.isfinite(a), np.isfinite(b)
    wa = None if wa is None else np.asarray(wa, float)[ka]
    wb = None if wb is None else np.asarray(wb, float)[kb]
    return float(stats.wasserstein_distance(a[ka], b[kb], wa, wb) / s)


def tost_welch(a, b, eps, alpha=0.05):
    """
    Equivalence of the means of two small samples *a*, *b* (e.g. panorama means) within ``±eps``.

    Welch SE and Welch-Satterthwaite degrees of freedom. Returns dict with ``delta`` (mean(a) - mean(b)),
    ``ci_low``/``ci_high`` (``1 - 2 alpha`` confidence interval), ``p`` (max of the two one-sided p values),
    ``df`` and ``equivalent`` (p < alpha).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    nu = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    dlt = a.mean() - b.mean()
    p = max(stats.t.sf((dlt + eps) / se, nu), stats.t.cdf((dlt - eps) / se, nu))
    h = stats.t.ppf(1 - alpha, nu) * se
    return {"delta": float(dlt), "ci_low": float(dlt - h), "ci_high": float(dlt + h), "p": float(p),
            "df": float(nu), "equivalent": bool(p < alpha)}


def classifier_test(X, y, groups=None, n_splits=5, max_per_group=3000, seed=0):
    """
    Balanced accuracy of a gradient-boosting classifier predicting *y* from *X*.

    With *groups* (e.g. panorama ids) the split is leave-one-group-out, so instances of a test image are
    never seen in training; otherwise stratified k-fold. At most *max_per_group* instances per group are
    drawn to keep the fit fast. Returns ``(balanced_accuracy, chance)`` with chance = 1 / number of classes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    X, y = np.asarray(X, float), np.asarray(y)
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        keep = np.concatenate([rng.permutation(np.flatnonzero(groups == g))[:max_per_group]
                               for g in np.unique(groups)])
        X, y, groups = X[keep], y[keep], groups[keep]
        split = LeaveOneGroupOut().split(X, y, groups)
    else:
        split = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    pred = np.empty_like(y)
    for tr, te in split:
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(balanced_accuracy_score(y, pred)), 1.0 / len(np.unique(y))


def group_table(df, value, by):
    """Mean, std and count of *value* per group (columns *by*)."""
    return df.groupby(by)[value].agg(["mean", "std", "count"]).reset_index()

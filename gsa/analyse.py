"""Indices of the sensitivity analysis from the files of gsa.collect.

    python3 -m gsa.analyse --share SHARE [--uniform SHARE_UNIFORM] --out FOLDER
                           [--B 500] [--sizes 128,256,512,1024]

Everything is computed from the aggregate outputs that gsa.collect writes, so
it runs anywhere. In FOLDER:

  indices.csv      first-order index S_i and OT index iota_i of every factor and
                   dummy, for every output and nested size: point estimate,
                   95% bootstrap interval, the 95th percentile of the matched
                   dummy and whether the index reads as non-zero (its lower end
                   above that percentile)
  convergence.csv  the statistics of Sarrazin et al. (2016) for each output,
                   family and size: ranking (95th percentile of the adjusted
                   rank difference), screening and indices (widest interval
                   among the factors read as zero and as non-zero), and the
                   largest change of an index from the previous size
  shapley.csv      nearest-neighbour Shapley effects (Broto et al. 2020). Step
                   1: all players and the continuous dummy, at every size.
                   Step 2: the players read as influential in step 1 (lower
                   end above the dummy's 95th percentile) alone, which lowers
                   the dimension of the neighbour search and so its bias. The
                   residual share is the variance left once the players are
                   fixed: the seeds X8 and X21, whose values carry no order a
                   neighbour can match, the factors left out, and the bias of
                   the estimator
  surrogate.csv    a gradient-boosting surrogate of each output on the
                   factors other than the seeds: cross-validated R2 and the
                   Shapley effects of the surrogate by the permutation
                   estimator of Song et al. (2016), as shares of Var Y
  uniform.csv      the sample with powers uniform in W against the same 256
                   points of the main sample: output summaries and indices
  summary.csv      each output: canonical value, median and 5th to 95th
                   percentile over the runs
  profile_bands.npz  5th, 50th and 95th percentiles over the runs of the
                   profile of every country and day type
  hourly_ot.csv    OT index of every factor for the hourly values of the
                   pooled profile, by day type, with the dummy thresholds

Players of the Shapley effects: the six appliance powers as one group, the
spread of the country multipliers X7, and X9 to X20 one by one, on the
coordinates of the sample, with every level of a discrete factor at the
centre of its interval so that runs with the same level coincide. The
bootstrap of every family resamples the runs (B = 500): it ignores the
stratification of the Sobol' design, so its intervals are conservative.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsa import design, estimators as est  # noqa: E402

OUTPUTS = ["Y1", "Y2", "Y2b", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10", "Y11"]
FACTOR_IDS = [f.id for f in design.FACTORS]
SEEDS = [f.id for f in design.FACTORS if f.kind == "seed"]
COLUMNS = FACTOR_IDS + design.DUMMIES
SIZES = (128, 256, 512, 1024)


def matched_dummy(fid):
    """The dummy with the same number of classes as factor fid."""
    f = next(f for f in design.FACTORS if f.id == fid)
    if f.kind == "levels":
        return f"D_{len(f.levels)}"
    if f.kind == "integer":
        return f"D_{int(f.high - f.low + 1)}"
    return "D_cont"


def w2sq_sorted(a, b):
    """est.w2sq_exact for samples already sorted."""
    na, nb = len(a), len(b)
    t = np.union1d(np.arange(1, na + 1, dtype=np.int64) * nb, np.arange(1, nb + 1, dtype=np.int64) * na)
    lengths = np.diff(np.r_[0, t]) / (na * nb)
    return float(np.sum(lengths * (a[(t - 1) // nb] - b[(t - 1) // na]) ** 2))


def column_classes(X):
    """Classes of every column of X (est.classes) as global ids: the classes
    of column j are off[j] to off[j] + K[j] - 1."""
    cls = np.column_stack([est.classes(X[:, j]) for j in range(X.shape[1])])
    K = cls.max(axis=0) + 1
    off = np.r_[0, np.cumsum(K)[:-1]]
    return cls, K, off


def indices_one_output(cls, K, off, W, y):
    """First-order index S_i and OT index iota_i of every column for one
    output y, from the classes of the columns (cls, n x J) with the runs of
    each class weighted by W[:, j] (weights adding up to the number of runs
    of the class; ones for the plain estimators). The squared W2 between a
    class and the whole sample is exact: int Q_c^2 + int Q^2 - 2 int Q_c Q,
    with int Q_c Q taken from the integral of the quantile function of the
    sample, which is linear between the points i/n."""
    n, J = cls.shape
    var = y.var()
    if var == 0:
        return np.zeros(J), np.zeros(J)
    y = y - y.mean()
    g = cls + off
    nK = int(K.sum())
    cnt = np.bincount(g.ravel(), minlength=nK)
    share = cnt / n
    col_of = np.repeat(np.arange(J), K)
    means = np.bincount(g.ravel(), weights=(W * y[:, None]).ravel(), minlength=nK) / np.maximum(cnt, 1)
    mcol = np.bincount(col_of, weights=share * means, minlength=J)
    S = np.bincount(col_of, weights=share * (means - mcol[col_of]) ** 2, minlength=J) / var
    grid = np.arange(n + 1) / n
    Gq = np.r_[0.0, np.cumsum(np.sort(y)) / n]
    M2 = float(np.mean(y ** 2))
    O = np.zeros(J)
    for j in range(J):
        order = np.lexsort((y, cls[:, j]))
        c = cls[order, j]
        ys = y[order]
        wn = W[order, j] / np.maximum(cnt[off[j] + c], 1)
        cs = np.cumsum(wn)
        first = np.r_[True, c[1:] != c[:-1]]
        last = np.r_[c[1:] != c[:-1], True]
        base = (cs - wn)[first][np.cumsum(first) - 1]
        hi = np.minimum(cs - base, 1.0)
        hi[last] = 1.0
        lo = np.clip(hi - wn, 0.0, 1.0)
        lo[first] = 0.0
        cross = np.bincount(c, weights=ys * (np.interp(hi, grid, Gq) - np.interp(lo, grid, Gq)),
                            minlength=K[j])
        m2 = np.bincount(c, weights=wn * ys ** 2, minlength=K[j])
        w2 = np.maximum(m2 + M2 - 2 * cross, 0.0)
        O[j] = np.sum(share[off[j]:off[j] + K[j]] * w2) / (2 * var)
    return S, O


def class_indices(X, Y):
    """First-order and OT indices (columns of X x columns of Y), the class-
    based estimators of gsa.estimators applied to every output at once."""
    cls, K, off = column_classes(X)
    W = np.ones(cls.shape)
    S, O = np.zeros((X.shape[1], Y.shape[1])), np.zeros((X.shape[1], Y.shape[1]))
    for o in range(Y.shape[1]):
        S[:, o], O[:, o] = indices_one_output(cls, K, off, W, Y[:, o])
    if np.any(S / 2 > O + 1e-9):
        raise AssertionError("S_i / 2 > iota_i: estimator error")
    return S, O


def boot_class_indices(X, Y, B, seed):
    rng = np.random.default_rng(seed)
    n = len(Y)
    point = class_indices(X, Y)
    reps = np.empty((2, B) + point[0].shape)
    for b in range(B):
        idx = rng.integers(0, n, n)
        reps[0, b], reps[1, b] = class_indices(X[idx], Y[idx])
    return point, reps


# ------------------------------------------- dependence between the columns
def design_dependence(X, names=COLUMNS):
    """Index of every column (as an output) on the classes of every other
    column of the sample: zero on average for independent columns but for
    the bias (K - 1)/(n - 1) of the estimator with K classes. A larger value
    means that the classes of one column carry an unbalanced distribution of
    the other, so that a given-data index of the first inherits part of the
    effect of the second."""
    n = len(X)
    S, _ = class_indices(X, X)
    K = np.array([len(np.unique(est.classes(X[:, i]))) for i in range(X.shape[1])])
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                rows.append({"classes_of": a, "column": b, "index": S[i, j],
                             "independent": (K[i] - 1) / (n - 1)})
    return pd.DataFrame(rows)


# ------------------------------- class-based indices adjusted for that dependence
N_DRIVERS = 3        # leading factors of an output whose shares the raking restores
N_COARSE = 4         # classes of a continuous driver in the raking


def w2sq_weighted(a, wa, b):
    """Squared W2 between a weighted sample (a sorted, weights wa adding up
    to one) and an unweighted one (b sorted), exact in one dimension."""
    ca = np.cumsum(wa)
    ca[-1] = 1.0
    cb = np.arange(1, len(b) + 1) / len(b)
    t = np.union1d(ca, cb)
    lengths = np.diff(np.r_[0.0, t])
    mid = t - lengths / 2
    ia = np.minimum(np.searchsorted(ca, mid), len(a) - 1)
    ib = np.minimum(np.searchsorted(cb, mid), len(b) - 1)
    return float(np.sum(lengths * (a[ia] - b[ib]) ** 2))


def raking_weights(classes, drivers, n_iter=50, tol=1e-6):
    """Weights of the runs, for each column of `classes` (n x J, the classes
    of J factors), that give every class the shares of the classes of each
    driver that the whole sample has: raking to the margins of the drivers
    within each class. drivers is a list of (classes of the driver, their
    number L, their shares in the whole sample). A driver class absent from a
    class cannot be restored there, and the shares of the other classes are
    matched among themselves. The weights add up to the number of runs of each
    class. Also returns, for each column, the imbalance left: the largest
    difference between a class's share of a driver class and the share in
    the whole sample, averaged over the classes."""
    classes = np.asarray(classes)
    if classes.ndim == 1:
        classes = classes[:, None]
    n, J = classes.shape
    K = classes.max(axis=0) + 1
    g = classes + np.r_[0, np.cumsum(K)[:-1]]
    nK = int(K.sum())
    W = np.ones((n, J))
    cells = [(g * L + d[:, None], L, target) for d, L, target in drivers]
    for _ in range(n_iter):
        worst = 0.0
        for cell, L, target in cells:
            tot = np.bincount(cell.ravel(), weights=W.ravel(), minlength=nK * L).reshape(nK, L)
            present = tot > 0
            want = np.where(present, target[None, :], 0.0)
            want = want / np.maximum(want.sum(axis=1, keepdims=True), 1e-300)
            have = tot / np.maximum(tot.sum(axis=1, keepdims=True), 1e-300)
            ratio = np.divide(want, have, out=np.ones_like(want), where=present)
            W *= ratio.ravel()[cell]
            worst = max(worst, float(np.max(np.abs(ratio - 1.0))))
        if worst < tol:
            break
    cnt = np.bincount(g.ravel(), minlength=nK)
    tot = np.bincount(g.ravel(), weights=W.ravel(), minlength=nK)
    W *= np.divide(cnt, tot, out=np.zeros(nK), where=tot > 0)[g]
    col_of = np.repeat(np.arange(J), K)
    gap = np.zeros(J)
    for cell, L, target in cells:
        tot = np.bincount(cell.ravel(), weights=W.ravel(), minlength=nK * L).reshape(nK, L)
        have = tot / np.maximum(tot.sum(axis=1, keepdims=True), 1e-300)
        dev = np.where(cnt > 0, np.abs(have - target[None, :]).max(axis=1), 0.0)
        gap = np.maximum(gap, np.bincount(col_of, weights=cnt / n * dev, minlength=J))
    return W, gap


def leading_drivers(S_raw, names=COLUMNS, factors=FACTOR_IDS):
    """For every output, the factors in decreasing order of their raw
    first-order index (column positions), enough to leave N_DRIVERS once the
    factor under study is set aside."""
    f_idx = np.array([names.index(c) for c in factors])
    return [list(f_idx[np.argsort(-S_raw[f_idx, o])][:N_DRIVERS + 1]) for o in range(S_raw.shape[1])]


def adjusted_indices(X, Y, drivers):
    """Class-based S_i and iota_i of every column of X for every output, the
    runs of each class reweighted by raking_weights on the leading drivers of
    the output (drivers[o], positions in X, the column under study set
    aside), so that each class carries the shares of the drivers that the
    whole sample has. Returns S, OT and the imbalance left (columns x outputs)."""
    n, m = Y.shape
    J = X.shape[1]
    cls, K, off = column_classes(X)
    spec = {}
    for d in {d for ds in drivers for d in ds}:
        c = est.classes(X[:, d], n_classes=N_COARSE)
        spec[d] = (c, int(c.max()) + 1, np.bincount(c) / n)
    S, O, G = np.zeros((J, m)), np.zeros((J, m)), np.zeros((J, m))
    cache = {}
    for o in range(m):
        groups = {}
        for j in range(J):
            groups.setdefault(tuple(d for d in drivers[o] if d != j)[:N_DRIVERS], []).append(j)
        W = np.ones((n, J))
        for drv, cols in groups.items():
            key = (drv, tuple(cols))
            if key not in cache:
                cache[key] = raking_weights(cls[:, cols], [spec[d] for d in drv])
            W[:, cols], G[cols, o] = cache[key]
        S[:, o], O[:, o] = indices_one_output(cls, K, off, W, Y[:, o])
    if np.any(S / 2 > O + 1e-9):
        raise AssertionError("S_i / 2 > iota_i: estimator error")
    return S, O, G


def boot_adjusted_indices(X, Y, B, seed, drivers=None):
    """Raw and adjusted indices with B bootstrap resamples of the adjusted
    ones (the drivers fixed at those of the full sample)."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    raw = class_indices(X, Y)
    if drivers is None:
        drivers = leading_drivers(raw[0])
    S, O, G = adjusted_indices(X, Y, drivers)
    reps = np.empty((2, B) + S.shape)
    for b in range(B):
        idx = rng.integers(0, n, n)
        reps[0, b], reps[1, b], _ = adjusted_indices(X[idx], Y[idx], drivers)
    return raw, (S, O), G, reps, drivers


def adjusted_rank_difference(s0, s1):
    r0, r1 = rankdata(-s0, method="average"), rankdata(-s1, method="average")
    w = np.maximum(s0, s1) ** 2
    return float(np.sum(np.abs(r1 - r0) * w) / np.sum(w)) if np.sum(w) > 0 else 0.0


def read_out(point, reps, names, factors, dummy_of):
    """Rows of the table of one family, output and size, and the three
    statistics of Sarrazin et al. over the factors.

    Resampling with replacement repeats runs, which raises the bias of a
    class-based estimator, so the bootstrap distribution sits above the
    point estimate (by up to a few hundredths for the OT index). The interval
    reported (lo, hi) is the percentile interval shifted by that bootstrap
    bias, clipped at zero; the raw percentiles are kept (lo_pct, hi_pct). An
    index reads as non-zero when the 2.5th percentile of its resamples lies
    above the 95th percentile of the resamples of its matched dummy, both
    raised alike by the resampling, and above zero (every family here is
    non-negative by definition, so a negative estimate never counts)."""
    lo_p, hi_p = np.percentile(reps, [2.5, 97.5], axis=0)
    bias = reps.mean(axis=0) - point
    lo, hi = np.maximum(lo_p - bias, 0.0), np.maximum(hi_p - bias, 0.0)
    pos = {c: i for i, c in enumerate(names)}
    thr = {d: float(np.percentile(reps[:, pos[d]], 95)) for d in set(dummy_of.values())}
    rows, nz = [], []
    for c in names:
        i = pos[c]
        t = thr[dummy_of[c]] if c in dummy_of else thr.get(c, np.nan)
        nonzero = bool(lo_p[i] > max(t, 0.0)) if c in dummy_of else False
        rows.append({"factor": c, "point": point[i], "lo": lo[i], "hi": hi[i], "lo_pct": lo_p[i],
                     "hi_pct": hi_p[i], "boot_bias": bias[i], "dummy_95": t, "nonzero": nonzero})
        if c in factors:
            nz.append(nonzero)
    f_idx = [pos[c] for c in factors]
    nz = np.array(nz)
    width = (hi_p - lo_p)[f_idx]
    rho = [adjusted_rank_difference(point[f_idx], r[f_idx]) for r in reps]
    stats = {"ranking": float(np.percentile(rho, 95)),
             "screening": float(width[~nz].max()) if (~nz).any() else 0.0,
             "indices": float(width[nz].max()) if nz.any() else 0.0,
             "n_nonzero": int(nz.sum())}
    return rows, stats


# ---------------------------------------------------------------- Shapley
PLAYERS = [("P_appliances", ["X1", "X2", "X3", "X4", "X5", "X6"]), ("X7", ["X7"])] + \
          [(f, [f]) for f in FACTOR_IDS if f not in ("X1", "X2", "X3", "X4", "X5", "X6", "X7")
           and f not in SEEDS]


def coordinates(table):
    """Coordinates of the factors and dummies for the neighbour search: the
    sample's coordinates, with every level of a discrete factor at the centre
    of its interval."""
    out = {}
    kinds = {f.id: f for f in design.FACTORS}
    for c in COLUMNS:
        u = table[f"u_{c}"].to_numpy(dtype=float)
        f = kinds.get(c)
        if f is not None and f.kind in ("levels", "integer"):
            L = len(f.levels) if f.kind == "levels" else int(f.high - f.low + 1)
            u = (np.minimum(np.floor(u * L), L - 1) + 0.5) / L
        elif f is None and c != "D_cont":
            L = int(c[2:])
            u = (np.minimum(np.floor(u * L), L - 1) + 0.5) / L
        out[c] = u
    return out


def shapley_table(U, Y, players, B, seed, k=3):
    """Shapley effects with intervals, and the residual share, for a list of
    (name, columns) players on coordinates U (dict) and outputs Y (n x m)."""
    n = len(Y)
    names = [p for p, _ in players]
    cols, mat = [], []
    for _, cs in players:
        cols.append(list(range(len(mat), len(mat) + len(cs))))
        mat += [U[c] for c in cs]
    Umat = np.column_stack(mat)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)])
    sh, res, shb, resb = est.shapley_knn_multi(Umat, Y, cols, k=k, counts=counts)
    return names, sh, res, shb, resb


# ---------------------------------------------------------------- surrogate
SURROGATE_FEATURES = [c for c in FACTOR_IDS if c not in SEEDS] + ["D_cont"]
SURROGATE_PLAYERS = PLAYERS + [("D_cont", ["D_cont"])]
KIND = {f.id: f.kind for f in design.FACTORS}


def make_surrogate(seed=0):
    """Gradient boosting on the factors other than the seeds and on the
    continuous dummy, whose effect on the fitted surrogate shows how much the
    fit draws on noise."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    cat = np.array([KIND.get(c) == "levels" for c in SURROGATE_FEATURES])
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, l2_regularization=1.0,
                                         categorical_features=cat, random_state=seed)


def independent_sample(r, rng):
    """r independent rows of the features of the surrogate, drawn like the
    factors of the design but without the dependence of the Sobol' columns."""
    pos = [COLUMNS.index(c) for c in SURROGATE_FEATURES]
    return design.analysis_matrix(rng.random((r, len(COLUMNS))))[:, pos]


def surrogate_indices(table, Y, B=0, n_perm=10000, n_perm_boot=1000, n_iid=2 ** 14, seed=0,
                      outputs=OUTPUTS):
    """For every output, a surrogate fitted to the runs of `table`: its
    cross-validated R2 and the share of Var Y it reproduces; its Shapley
    effects (Song et al. 2016) as shares of Var Y, with B bootstrap refits of
    the surrogate for their intervals; and its first-order and OT indices on
    n_iid independent points, the first-order ones as shares of Var Y and the
    OT ones on the surrogate's own scale."""
    from sklearn.model_selection import KFold, cross_val_predict
    X = table[SURROGATE_FEATURES].to_numpy(dtype=float)
    n = len(X)
    pos = {c: i for i, c in enumerate(SURROGATE_FEATURES)}
    players = [[pos[c] for c in cs] for _, cs in SURROGATE_PLAYERS]
    names = [p for p, _ in SURROGATE_PLAYERS]
    rng = np.random.default_rng(seed)
    Z = independent_sample(n_iid, rng)
    rows = []
    for o, y_name in enumerate(outputs):
        y = Y[:, o]
        var_y = y.var()
        pred = cross_val_predict(make_surrogate(seed), X, y, cv=KFold(5, shuffle=True, random_state=seed))
        r2 = 1 - np.mean((y - pred) ** 2) / var_y
        model = make_surrogate(seed).fit(X, y)
        sh, var_f = est.song_shapley(model.predict, independent_sample, players, n_perm=n_perm, seed=seed)
        sh = sh * var_f / var_y
        fz = model.predict(Z)
        S_f, O_f = class_indices(Z, fz[:, None])
        boot = np.full((B, len(names)), np.nan)
        for b in range(B):
            idx = rng.integers(0, n, n)
            mb = make_surrogate(seed).fit(X[idx], y[idx])
            sb, vb = est.song_shapley(mb.predict, independent_sample, players, n_perm=n_perm_boot,
                                      seed=seed + 1 + b)
            boot[b] = sb * vb / y[idx].var()
        if B:
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            thr = float(np.percentile(boot[:, names.index("D_cont")], 95))
        else:
            lo = hi = np.full(len(names), np.nan)
            thr = np.nan
        for i, p in enumerate(names):
            cols = [pos[c] for c in SURROGATE_PLAYERS[i][1]]
            rows.append({"output": y_name, "N": n, "player": p, "shapley": sh[i], "lo": lo[i],
                         "hi": hi[i], "dummy_95": thr,
                         "nonzero": bool(lo[i] > max(thr, 0.0)) and p != "D_cont" if B else np.nan,
                         "S": float(S_f[cols[0], 0] * var_f / var_y) if len(cols) == 1 else np.nan,
                         "OT": float(O_f[cols[0], 0]) if len(cols) == 1 else np.nan,
                         "cv_r2": float(r2), "explained": float(var_f / var_y)})
        # the appliance factors one by one (first-order and OT indices only)
        for c in ("X1", "X2", "X3", "X4", "X5", "X6"):
            rows.append({"output": y_name, "N": n, "player": c, "shapley": np.nan, "lo": np.nan,
                         "hi": np.nan, "dummy_95": np.nan, "nonzero": np.nan,
                         "S": float(S_f[pos[c], 0] * var_f / var_y), "OT": float(O_f[pos[c], 0]),
                         "cv_r2": float(r2), "explained": float(var_f / var_y)})
    return pd.DataFrame(rows)


def summary(table, baseline):
    """Each output over the runs, and the percentile of the canonical value
    among them (share of runs below it, ties counted half)."""
    rows = []
    for y in OUTPUTS:
        v = table[y].to_numpy(dtype=float)
        c = float(baseline.get(y, np.nan))
        rows.append({"output": y, "canonical": c,
                     "canonical_percentile": 100 * (np.mean(v < c) + 0.5 * np.mean(v == c))
                     if np.isfinite(c) else np.nan,
                     "median": np.median(v), "p05": np.percentile(v, 5), "p95": np.percentile(v, 95),
                     "mean": v.mean(), "sd": v.std(ddof=1)})
    return pd.DataFrame(rows)


def profile_bands(profiles):
    return np.percentile(profiles, [5, 50, 95], axis=0).astype(np.float32)


def shape_bands(profiles):
    """5th, 50th and 95th percentiles over the runs of the hourly shape
    (gsa.benchmarks.shape) of every country, all days: percentiles x
    countries x 24."""
    from gsa.benchmarks import hourly, shape
    shapes = np.array([[shape(hourly(profiles[r, k, 2])) for k in range(profiles.shape[1])]
                       for r in range(profiles.shape[0])])
    return np.percentile(shapes, [5, 50, 95], axis=0).astype(np.float32)


def hourly_ot(table, profiles, B, seed):
    """Adjusted OT and first-order indices of the factors and dummies for the
    hourly means of the pooled profile, by day type, with the 95th percentile
    of the matched dummies over B resamples."""
    from gsa.benchmarks import CLOCK_HOUR
    X = table[COLUMNS].to_numpy(dtype=float)
    dummy_of = {c: matched_dummy(c) for c in FACTOR_IDS}
    rows = []
    for d, day in enumerate(["working", "non_working", "all"]):
        prof = profiles[:, -1, d, :]                                      # ALL countries
        H = np.column_stack([prof[:, CLOCK_HOUR == h].mean(axis=1) for h in range(24)])
        raw, (S, O), G, reps, _ = boot_adjusted_indices(X, H, B, seed + d)
        for j, c in enumerate(COLUMNS):
            dj = COLUMNS.index(dummy_of.get(c, c))
            for h in range(24):
                rows.append({"day_type": day, "hour": h, "factor": c, "ot": O[j, h],
                             "dummy_95": float(np.percentile(reps[1, :, dj, h], 95)),
                             "S": S[j, h], "S_dummy_95": float(np.percentile(reps[0, :, dj, h], 95)),
                             "ot_raw": raw[1][j, h], "imbalance": G[j, h]})
    return pd.DataFrame(rows)


PARTS = ("summary", "dependence", "indices", "knn", "surrogate", "profiles", "uniform")


def load(share):
    table = pd.read_csv(Path(share) / "gsa_outputs.csv").sort_values("run").reset_index(drop=True)
    return table, table[COLUMNS].to_numpy(dtype=float), table[OUTPUTS].to_numpy(dtype=float)


def part_dependence(table, X, out, sizes):
    frames = []
    for N in sizes:
        d = design_dependence(X[:N])
        d.insert(0, "N", N)
        frames.append(d)
    pd.concat(frames).to_csv(out / "design_dependence.csv", index=False)


def part_indices(X, Y, out, sizes, B, log):
    """Adjusted first-order and OT indices on the nested designs, the raw ones
    beside them, and the statistics of Sarrazin et al. (2016)."""
    dummy_of = {c: matched_dummy(c) for c in FACTOR_IDS}
    f_idx = [COLUMNS.index(c) for c in FACTOR_IDS]
    rows, conv, prev = [], [], {}
    for N in sizes:
        t0 = time.time()
        raw, adj, G, reps, drivers = boot_adjusted_indices(X[:N], Y[:N], B, seed=N)
        for k, (fam, point, rep) in enumerate((("S", adj[0], reps[0]), ("OT", adj[1], reps[1]))):
            for o, y in enumerate(OUTPUTS):
                r, st = read_out(point[:, o], rep[:, :, o], COLUMNS, FACTOR_IDS, dummy_of)
                for i, row in enumerate(r):
                    rows.append({"output": y, "family": fam, "N": N, **row, "raw": raw[k][i, o],
                                 "imbalance": G[i, o],
                                 "drivers": " ".join(COLUMNS[d] for d in drivers[o][:N_DRIVERS + 1])})
                change = (float(np.max(np.abs(point[f_idx, o] - prev[(fam, o)])))
                          if (fam, o) in prev else np.nan)
                prev[(fam, o)] = point[f_idx, o]
                conv.append({"output": y, "family": fam, "N": N, **st, "max_change": change})
        log(f"  adjusted class-based indices, {N} runs: {time.time() - t0:.0f} s")
    pd.DataFrame(rows).to_csv(out / "indices.csv", index=False)
    pd.DataFrame(conv).to_csv(out / "convergence_indices.csv", index=False)


def part_knn(table, Y, out, sizes, B, log):
    """Nearest-neighbour Shapley effects, step 1 on every size and step 2 on
    the largest, and the statistics of Sarrazin et al. for step 1."""
    U = coordinates(table)
    step1 = PLAYERS + [("D_cont", ["D_cont"])]
    sh_rows, conv, prev = [], [], {}
    for N in sizes:
        t0 = time.time()
        UN = {c: v[:N] for c, v in U.items()}
        names, sh, res, shb, resb = shapley_table(UN, Y[:N], step1, B, seed=10 + N)
        thr = np.percentile(shb[:, names.index("D_cont"), :], 95, axis=0)
        lo, hi = np.percentile(shb, [2.5, 97.5], axis=0)
        for o, y in enumerate(OUTPUTS):
            nz = []
            for i, p in enumerate(names):
                nonzero = bool(lo[i, o] > max(thr[o], 0.0)) and p != "D_cont"
                nz.append(nonzero)
                sh_rows.append({"output": y, "step": 1, "N": N, "player": p, "point": sh[i, o],
                                "lo": lo[i, o], "hi": hi[i, o], "dummy_95": thr[o],
                                "nonzero": nonzero})
            rlo, rhi = np.percentile(resb[:, o], [2.5, 97.5])
            sh_rows.append({"output": y, "step": 1, "N": N, "player": "residual", "point": res[o],
                            "lo": rlo, "hi": rhi, "dummy_95": np.nan, "nonzero": False})
            p_idx = [i for i, p in enumerate(names) if p != "D_cont"]
            width = (hi - lo)[p_idx, o]
            nzv = np.array([nz[i] for i in p_idx])
            rho = [adjusted_rank_difference(sh[p_idx, o], shb[b, p_idx, o]) for b in range(B)]
            change = (float(np.max(np.abs(sh[p_idx, o] - prev[o]))) if o in prev else np.nan)
            prev[o] = sh[p_idx, o]
            conv.append({"output": y, "family": "Shapley_knn", "N": N,
                         "ranking": float(np.percentile(rho, 95)),
                         "screening": float(width[~nzv].max()) if (~nzv).any() else 0.0,
                         "indices": float(width[nzv].max()) if nzv.any() else 0.0,
                         "n_nonzero": int(nzv.sum()), "max_change": change})
        log(f"  nearest-neighbour Shapley effects, step 1, {N} runs: {time.time() - t0:.0f} s")
    N = sizes[-1]
    last = pd.DataFrame([r for r in sh_rows if r["N"] == N and r["step"] == 1])
    UN = {c: v[:N] for c, v in U.items()}
    for o, y in enumerate(OUTPUTS):
        keep = last[(last.output == y) & last.nonzero].player.tolist()
        if not keep:
            continue
        names, sh, res, shb, resb = shapley_table(UN, Y[:N, [o]], [p for p in PLAYERS if p[0] in keep],
                                                  B, seed=99)
        lo, hi = np.percentile(shb, [2.5, 97.5], axis=0)
        for i, p in enumerate(names):
            sh_rows.append({"output": y, "step": 2, "N": N, "player": p, "point": sh[i, 0],
                            "lo": lo[i, 0], "hi": hi[i, 0], "dummy_95": np.nan, "nonzero": True})
        rlo, rhi = np.percentile(resb[:, 0], [2.5, 97.5])
        sh_rows.append({"output": y, "step": 2, "N": N, "player": "residual", "point": res[0],
                        "lo": rlo, "hi": rhi, "dummy_95": np.nan, "nonzero": False})
    pd.DataFrame(sh_rows).to_csv(out / "knn_shapley.csv", index=False)
    pd.DataFrame(conv).to_csv(out / "convergence_knn.csv", index=False)
    log("  nearest-neighbour Shapley effects, step 2: done")


def part_surrogate(table, Y, out, sizes, B_s, n_perm, log):
    """Surrogate indices on every nested size, with bootstrap intervals on the
    largest."""
    frames = []
    for N in sizes:
        t0 = time.time()
        frames.append(surrogate_indices(table.iloc[:N], Y[:N], B=B_s if N == sizes[-1] else 0,
                                        n_perm=n_perm))
        log(f"  surrogate, {N} runs: {time.time() - t0:.0f} s")
    pd.concat(frames).to_csv(out / "surrogate.csv", index=False)


def save_bands(out, profiles, z):
    np.savez_compressed(out / "profile_bands.npz", bands=profile_bands(profiles),
                        shape_bands=shape_bands(profiles), percentiles=np.array([5, 50, 95]),
                        countries=z["countries"], day_types=z["day_types"])


def part_profiles(share, table, out, N, B, log):
    z = np.load(Path(share) / "gsa_profiles.npz")
    order = np.argsort(z["runs"])
    profiles = z["profiles"][order][:N].astype(np.float64)
    save_bands(out, profiles, z)
    hourly_ot(table.iloc[:N], profiles, B, seed=7).to_csv(out / "hourly.csv", index=False)
    log("  profiles: done")


def part_uniform(table, uniform, out, B, n_perm, log):
    """The sample with powers uniform in W against the same points of the main
    sample: output summaries, adjusted indices and surrogate Shapley effects."""
    tu = pd.read_csv(Path(uniform) / "gsa_outputs.csv").sort_values("run").reset_index(drop=True)
    base = table.iloc[:len(tu)]
    same = np.allclose(tu[[f"u_{c}" for c in COLUMNS]].to_numpy(),
                       base[[f"u_{c}" for c in COLUMNS]].to_numpy())
    dummy_of = {c: matched_dummy(c) for c in FACTOR_IDS}
    urows = []
    for name, t in (("log-uniform", base), ("uniform", tu)):
        Xs, Ys = t[COLUMNS].to_numpy(dtype=float), t[OUTPUTS].to_numpy(dtype=float)
        raw, (S, O), G, reps, _ = boot_adjusted_indices(Xs, Ys, B, seed=5)
        sur = surrogate_indices(t, Ys, B=0, n_perm=n_perm)
        for o, y in enumerate(OUTPUTS):
            v = Ys[:, o]
            urows.append({"sample": name, "output": y, "factor": "_summary", "mean": v.mean(),
                          "sd": v.std(ddof=1), "median": np.median(v), "p05": np.percentile(v, 5),
                          "p95": np.percentile(v, 95), "same_points": same})
            r, _ = read_out(O[:, o], reps[1][:, :, o], COLUMNS, FACTOR_IDS, dummy_of)
            rs, _ = read_out(S[:, o], reps[0][:, :, o], COLUMNS, FACTOR_IDS, dummy_of)
            for row, row_s in zip(r, rs):
                if row["factor"] in FACTOR_IDS:
                    urows.append({"sample": name, "output": y, "factor": row["factor"],
                                  "S": row_s["point"], "S_nonzero": row_s["nonzero"],
                                  "OT": row["point"], "OT_nonzero": row["nonzero"]})
            for _, r_ in sur[(sur.output == y) & sur.shapley.notna()].iterrows():
                urows.append({"sample": name, "output": y, "factor": r_.player,
                              "Shapley": r_.shapley, "cv_r2": r_.cv_r2})
    pd.DataFrame(urows).to_csv(out / "uniform.csv", index=False)
    log("  uniform-in-watts comparison: done")


def run_all(share, out, uniform=None, B=500, sizes=SIZES, n_perm=10000, B_surrogate=50,
            parts=PARTS, log=print):
    share, out = Path(share), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table, X, Y = load(share)
    sizes = [N for N in sizes if N <= len(table)]
    if "summary" in parts:
        baseline = {}
        if (share / "baseline_check.json").is_file():
            baseline = json.loads((share / "baseline_check.json").read_text())
        summary(table, baseline).to_csv(out / "summary.csv", index=False)
    if "dependence" in parts:
        part_dependence(table, X, out, sizes)
    if "indices" in parts:
        part_indices(X, Y, out, sizes, B, log)
    if "knn" in parts:
        part_knn(table, Y, out, sizes, B, log)
    if "surrogate" in parts:
        part_surrogate(table, Y, out, sizes, B_surrogate, n_perm, log)
    if "profiles" in parts and (share / "gsa_profiles.npz").is_file():
        part_profiles(share, table, out, sizes[-1], min(B, 100), log)
    if "uniform" in parts and uniform is not None:
        part_uniform(table, uniform, out, min(B, 200), n_perm, log)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--share", type=Path, required=True)
    parser.add_argument("--uniform", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--B", type=int, default=500)
    parser.add_argument("--sizes", default="128,256,512,1024")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--B-surrogate", type=int, default=50)
    parser.add_argument("--parts", default=",".join(PARTS),
                        help="comma-separated subset of " + ", ".join(PARTS))
    args = parser.parse_args(argv)
    parts = tuple(args.parts.split(","))
    unknown = set(parts) - set(PARTS)
    if unknown:
        parser.error(f"unknown parts: {', '.join(sorted(unknown))}")
    t0 = time.time()
    run_all(args.share, args.out, args.uniform, B=args.B,
            sizes=tuple(int(s) for s in args.sizes.split(",")), n_perm=args.n_perm,
            B_surrogate=args.B_surrogate, parts=parts)
    print(f"Done in {time.time() - t0:.0f} s; results in {args.out}")


if __name__ == "__main__":
    main()

"""Indices of the sensitivity analysis from the files of gsa.collect.

    python3 -m gsa.analyse --share SHARE [--uniform SHARE_UNIFORM] --out FOLDER
                           [--B 500] [--sizes 128,256,...]

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
  summary.csv      each output over the runs (the uncertainty analysis):
                   canonical value and its percentile, median, 5th to 95th
                   percentile, mean, SD, coefficient of variation, skewness,
                   excess kurtosis, bimodality coefficient and the share of
                   the runs at the most frequent value
  ot_parts.csv     the adjusted first-order and OT indices on the full sample
                   with the mean, spread and shape parts of the OT index and
                   its value for a factor that only shifts a Gaussian output
  brackets.csv     first-order index, Shapley effect and total index of the
                   surrogate on independent points, the appliance powers one
                   by one and as a group (S <= Sh <= T for independent inputs)
  seeds.csv        with --reseed: total index of the redrawn seeds on every
                   output from the reruns of gsa.reseed (Jansen's estimator),
                   by variant, with a bootstrap interval
  budget.csv       with --reseed: the variance of every output from the model
                   alone, as the first-order effects of the factors other than
                   the seeds (sum of their first-order indices from the runs,
                   first_order_sums), the interactions among them (by
                   difference) and the seeds (reruns), which add up to one;
                   with brackets.csv also what the surrogate reproduces of
                   each part, and so the share of the interactions it
                   apportions among the factors
  first_order_sums.csv  with --reseed: that sum, the floor of the estimator,
                   a bound of the leak of the design's dependence, and the
                   bootstrap standard error
  totals_check.csv with --reseed, indices.csv and totals.csv: the total index
                   of the surrogate of every factor against the bracket
                   S_i <= T_i <= S_i + I that the runs and the reruns give
                   without a surrogate (I the share of the interactions)
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
SIZES = tuple(2 ** k for k in range(7, 14))      # nested designs up to the runs present


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


def indices_one_output(cls, K, off, W, y, parts=False):
    """First-order index S_i and OT index iota_i of every column for one
    output y, from the classes of the columns (cls, n x J) with the runs of
    each class weighted by W[:, j] (weights adding up to the number of runs
    of the class; ones for the plain estimators). The squared W2 between a
    class and the whole sample is exact: int Q_c^2 + int Q^2 - 2 int Q_c Q,
    with int Q_c Q taken from the integral of the quantile function of the
    sample, which is linear between the points i/n.

    With parts=True, also the three parts of iota_i (3 x J), from
    W2^2 = (m_c - m)^2 + (s_c - s)^2 + 2 s_c s (1 - r_c), where m and s are
    the mean and standard deviation and r_c the correlation of the quantile
    functions of the class and of the whole sample: the mean part (S_i / 2
    for the plain estimator), the spread part and the shape part, each
    averaged over the classes and divided by 2 Var Y. A factor that only
    shifts a Gaussian output has no shape part and a spread part set by S_i,
    so that iota_i = (S_i + (1 - sqrt(1 - S_i))^2) / 2."""
    n, J = cls.shape
    var = y.var()
    if var == 0:
        return (np.zeros(J), np.zeros(J), np.zeros((3, J))) if parts else (np.zeros(J), np.zeros(J))
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
    P = np.zeros((3, J))
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
        sh_j = share[off[j]:off[j] + K[j]]
        O[j] = np.sum(sh_j * w2) / (2 * var)
        if parts:
            mu = means[off[j]:off[j] + K[j]]
            sd_c = np.sqrt(np.maximum(m2 - mu ** 2, 0.0))
            sd = np.sqrt(M2)
            P[:, j] = [np.sum(sh_j * mu ** 2), np.sum(sh_j * (sd_c - sd) ** 2),
                       np.sum(sh_j * 2 * (sd_c * sd - cross))]
    if parts:
        return S, O, P / (2 * var)
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


def adjusted_indices(X, Y, drivers, parts=False):
    """Class-based S_i and iota_i of every column of X for every output, the
    runs of each class reweighted by raking_weights on the leading drivers of
    the output (drivers[o], positions in X, the column under study set
    aside), so that each class carries the shares of the drivers that the
    whole sample has. Returns S, OT and the imbalance left (columns x
    outputs), and with parts=True also the mean, spread and shape parts of
    iota_i (3 x columns x outputs, see indices_one_output)."""
    n, m = Y.shape
    J = X.shape[1]
    P = np.zeros((3, J, m))
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
        if parts:
            S[:, o], O[:, o], P[:, :, o] = indices_one_output(cls, K, off, W, Y[:, o], parts=True)
        else:
            S[:, o], O[:, o] = indices_one_output(cls, K, off, W, Y[:, o])
    if np.any(S / 2 > O + 1e-9):
        raise AssertionError("S_i / 2 > iota_i: estimator error")
    return (S, O, G, P) if parts else (S, O, G)


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


def boot_plain_indices(X, Y, B, seed):
    """Plain class-based indices (every run of a class weighted alike) with B
    bootstrap resamples, in the form of boot_adjusted_indices: the point
    estimates serve as both the raw and the reported ones, and no imbalance
    is left to report (NaN)."""
    raw, reps = boot_class_indices(X, Y, B, seed)
    return raw, raw, np.full(raw[0].shape, np.nan), reps, leading_drivers(raw[0])


def savage_scores(x):
    """Savage scores of the values x ranked from the largest: sum_{k=r}^n 1/k
    for rank r, ties given the mean score of their ranks."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    tail = np.cumsum(1.0 / np.arange(n, 0, -1))[::-1]          # score of rank r = tail[r - 1]
    order = np.argsort(-x, kind="stable")
    sc = np.empty(n)
    sc[order] = tail
    for v in np.unique(x):                                       # ties: mean score
        m = x == v
        if m.sum() > 1:
            sc[m] = sc[m].mean()
    return sc


def top_down_correlation(a, b):
    """Top-down correlation of Iman and Conover (1987): the Pearson correlation
    of the Savage scores of two rankings, which weighs agreement on the most
    important factors most."""
    return float(np.corrcoef(savage_scores(a), savage_scores(b))[0, 1])


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


def location_ot(S):
    """OT index of a factor that only shifts a Gaussian output, as a function
    of its first-order index: (S + (1 - sqrt(1 - S))^2) / 2."""
    S = np.clip(np.asarray(S, dtype=float), 0.0, 1.0)
    return (S + (1 - np.sqrt(1 - S)) ** 2) / 2


def ot_parts(X, Y, drivers=None, adjusted=False):
    """The first-order and OT indices of every factor on the full sample
    (plain, or adjusted), with the mean, spread and shape parts of the OT
    index and its value for a pure shift of a Gaussian output
    (location_ot)."""
    if adjusted:
        if drivers is None:
            drivers = leading_drivers(class_indices(X, Y)[0])
        S, O, _, P = adjusted_indices(X, Y, drivers, parts=True)
    else:
        cls, K, off = column_classes(X)
        W = np.ones(X.shape)
        S, O, P = np.zeros((X.shape[1], Y.shape[1])), np.zeros((X.shape[1], Y.shape[1])), \
            np.zeros((3, X.shape[1], Y.shape[1]))
        for o in range(Y.shape[1]):
            S[:, o], O[:, o], P[:, :, o] = indices_one_output(cls, K, off, W, Y[:, o], parts=True)
    rows = []
    for o, y in enumerate(OUTPUTS):
        for c in FACTOR_IDS:
            j = COLUMNS.index(c)
            rows.append({"output": y, "factor": c, "S": S[j, o], "OT": O[j, o], "mean": P[0, j, o],
                         "spread": P[1, j, o], "shape": P[2, j, o],
                         "location_OT": float(location_ot(S[j, o]))})
    return pd.DataFrame(rows)


def pick_freeze(f, A, B, groups):
    """First-order (Saltelli et al. 2010) and total (Jansen 1999) variances of
    f for each group of columns, from two independent samples A and B, with
    their Monte Carlo standard errors: V_u = E[f(B) (f(A_B^u) - f(A))] and
    VT_u = E[(f(A) - f(A_B^u))^2] / 2, where A_B^u is A with the columns of u
    taken from B. The values of f are centred on their mean over A and B
    first: this leaves V_u unbiased and keeps its variance from growing with
    the square of the mean of f, which for an output such as Y2b (mean 4.9 h,
    SD 0.13 h) would swamp the estimate."""
    fA, fB = f(A), f(B)
    f0 = float(np.mean(np.r_[fA, fB]))
    fA, fB = fA - f0, fB - f0
    n = len(A)
    out = []
    for cols in groups:
        AB = A.copy()
        AB[:, cols] = B[:, cols]
        fAB = f(AB) - f0
        a, t = fB * (fAB - fA), 0.5 * (fA - fAB) ** 2
        out.append((a.mean(), a.std(ddof=1) / np.sqrt(n), t.mean(), t.std(ddof=1) / np.sqrt(n)))
    return np.array(out)


def surrogate_brackets(table, Y, n_perm=100000, batches=10, n_pf=2 ** 16, seed=0, outputs=OUTPUTS):
    """For every output, the surrogate of surrogate_indices fitted to all the
    runs and, on independent points, its first-order and total indices
    (pick_freeze) of every feature one by one and of the six appliance powers
    as a group (closed indices), and its Shapley effects in two games: every
    feature a player, the powers one by one, and the powers as one player
    (SURROGATE_PLAYERS). Each game takes `batches` independent runs of the
    permutation estimator of Song et al. (2016) with n_perm / batches
    permutations each, whose spread gives the Monte Carlo error. All as shares
    of Var Y. For independent inputs S <= Sh <= T for every player, and the
    sum of the Shapley effects of the powers one by one is at least the
    effect of their group."""
    X = table[SURROGATE_FEATURES].to_numpy(dtype=float)
    pos = {c: i for i, c in enumerate(SURROGATE_FEATURES)}
    single = [(c, [pos[c]]) for c in SURROGATE_FEATURES]
    grouped = [(p, [pos[c] for c in cs]) for p, cs in SURROGATE_PLAYERS]
    group_cols = dict(grouped)["P_appliances"]
    rows = []
    for o, y_name in enumerate(outputs):
        y = Y[:, o]
        var_y = y.var()
        model = make_surrogate(seed).fit(X, y)
        rng = np.random.default_rng(seed + 7)
        A, B = independent_sample(n_pf, rng), independent_sample(n_pf, rng)
        pf = pick_freeze(model.predict, A, B, [cols for _, cols in single] + [group_cols]) / var_y
        pf_of = {name: pf[i] for i, (name, _) in enumerate(single)}
        pf_of["P_appliances"] = pf[-1]
        for game, players in (("single", single), ("grouped", grouped)):
            reps, var_f = [], []
            for b in range(batches):
                sh, vf = est.song_shapley(model.predict, independent_sample, [cols for _, cols in players],
                                          n_perm=n_perm // batches, seed=seed + 1000 * (game == "grouped") + b)
                reps.append(sh * vf / var_y)
                var_f.append(vf)
            reps = np.array(reps)
            for i, (name, _) in enumerate(players):
                S_, S_se, T_, T_se = pf_of[name]
                rows.append({"output": y_name, "game": game, "player": name,
                             "shapley": reps[:, i].mean(), "shapley_se": reps[:, i].std(ddof=1) / np.sqrt(batches),
                             "S": S_, "S_se": S_se, "T": T_, "T_se": T_se,
                             "explained": float(np.mean(var_f) / var_y)})
    return pd.DataFrame(rows)


def surrogate_totals(table, Y, sizes, B=50, n_pf=2 ** 15, n_pf_boot=2 ** 14, seed=0, outputs=OUTPUTS):
    """First-order and total indices of the surrogate of surrogate_indices on
    every nested size, as shares of Var Y of the runs: every feature one by
    one (the six powers and the continuous dummy included) and the six powers
    as a group (closed indices), by pick_freeze on n_pf pairs of independent
    points, the same for every size and output. On the largest size, B
    bootstrap refits of the surrogate (runs resampled with replacement, the
    first n_pf_boot pairs of points) give the intervals and, through the
    dummy, the screening of read_out. Returns the table of indices and the
    statistics of Sarrazin et al. (2016) for the total indices (on the
    largest size) with the largest change of a total index from the size
    before."""
    sizes = [N for N in sizes if N <= len(table)]
    X = table[SURROGATE_FEATURES].to_numpy(dtype=float)
    pos = {c: i for i, c in enumerate(SURROGATE_FEATURES)}
    groups = [(c, [pos[c]]) for c in SURROGATE_FEATURES] + \
        [("P_appliances", [pos[c] for c in dict(SURROGATE_PLAYERS)["P_appliances"]])]
    names = [g for g, _ in groups]
    cols = [c for _, c in groups]
    factors = [c for c in SURROGATE_FEATURES if c != "D_cont"]
    dummy_of = {c: "D_cont" for c in factors + ["P_appliances"]}
    f_idx = [names.index(c) for c in factors]
    rng = np.random.default_rng(seed + 11)
    A, Bm = independent_sample(n_pf, rng), independent_sample(n_pf, rng)
    rows, conv, prev = [], [], {}
    for N in sizes:
        for o, y_name in enumerate(outputs):
            y = Y[:N, o]
            var_y = y.var()
            model = make_surrogate(seed).fit(X[:N], y)
            pf = pick_freeze(model.predict, A, Bm, cols) / var_y
            explained = float(np.var(model.predict(A)) / var_y)
            extra = [{} for _ in names]
            stats = {}
            if N == sizes[-1] and B:
                reps = np.empty((2, B, len(names)))
                brng = np.random.default_rng(seed + 100 + o)
                for b in range(B):
                    idx = brng.integers(0, N, N)
                    mb = make_surrogate(seed).fit(X[:N][idx], y[idx])
                    pfb = pick_freeze(mb.predict, A[:n_pf_boot], Bm[:n_pf_boot], cols) / y[idx].var()
                    reps[0, b], reps[1, b] = pfb[:, 0], pfb[:, 2]
                rT, stats = read_out(pf[:, 2], reps[1], names, factors, dummy_of)
                rS, _ = read_out(pf[:, 0], reps[0], names, factors, dummy_of)
                for i in range(len(names)):
                    extra[i] = {k: rT[i][k] for k in ("lo", "hi", "lo_pct", "hi_pct", "boot_bias",
                                                     "dummy_95", "nonzero")}
                    extra[i].update({"S_lo": rS[i]["lo"], "S_hi": rS[i]["hi"], "S_nonzero": rS[i]["nonzero"]})
            for i, name in enumerate(names):
                rows.append({"N": N, "output": y_name, "player": name, "S": pf[i, 0], "S_se": pf[i, 1],
                             "T": pf[i, 2], "T_se": pf[i, 3], "explained": explained, **extra[i]})
            change = (float(np.max(np.abs(pf[f_idx, 2] - prev[o]))) if o in prev else np.nan)
            prev[o] = pf[f_idx, 2]
            conv.append({"output": y_name, "family": "T", "N": N, **stats, "max_change": change})
    return pd.DataFrame(rows), pd.DataFrame(conv)


def seed_totals(table, reseeded, B=1000, seed=0):
    """Total index of the redrawn seeds on every output by the estimator of
    Jansen (1999), T = mean((y_i - y'_i)^2) / (2 Var Y), over the points of
    `reseeded` (a gsa_outputs table of gsa.reseed, whose run is the index of
    the base point), with Var Y over all the runs of `table`, a bootstrap
    interval over the points, and the share of points whose output changed."""
    base = table.set_index("run")
    r = reseeded.set_index("run")
    idx = r.index.intersection(base.index)
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(idx), (B, len(idx)))
    rows = []
    for y in OUTPUTS:
        var = table[y].astype(float).var(ddof=0)
        d2 = (base.loc[idx, y].astype(float).to_numpy() - r.loc[idx, y].astype(float).to_numpy()) ** 2 / 2
        T = float(d2.mean() / var) if var > 0 else 0.0
        bt = d2[boots].mean(axis=1) / var if var > 0 else np.zeros(B)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        rows.append({"output": y, "T": T, "lo": float(lo), "hi": float(hi), "points": len(idx),
                     "changed": float(np.mean(d2 > 0))})
    return pd.DataFrame(rows)


def seed_dispersion(table, reseeded, bands=None, factors=("X7", "X16", "X2", "X13", "X19"),
                    n_perm=2000, seed=0):
    """Which factors modulate the effect of the redrawn seeds. At every point
    of `reseeded`, d = (y - y')^2 / 2 estimates without bias the variance that
    the seeds leave there, so its mean over the points in a band of a factor,
    over Var Y of all the runs of `table`, is the total index of the seeds
    within that band (the bands average back to the total index). `bands`
    maps a factor to its band edges. Also the rank correlation of d with each
    of `factors` and its two-sided permutation p-value. Returns the bands and
    the correlations as two tables."""
    from scipy.stats import spearmanr
    bands = bands or {"X7": (0.0, 0.1, 0.2, 0.3), "X16": (6, 8, 10, 12)}
    base = table.set_index("run")
    r = reseeded.set_index("run")
    idx = r.index.intersection(base.index)
    x = base.loc[idx]
    rng = np.random.default_rng(seed)
    rows, corr = [], []
    for y in OUTPUTS:
        var = table[y].astype(float).var(ddof=0)
        d = (base.loc[idx, y].astype(float).to_numpy() - r.loc[idx, y].astype(float).to_numpy()) ** 2 / 2
        if var <= 0 or not np.any(d > 0):
            continue
        for f, edges in bands.items():
            v = x[f].astype(float).to_numpy()
            for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                m = (v >= lo) & ((v < hi) if k < len(edges) - 2 else (v <= hi))
                rows.append({"output": y, "factor": f, "lo": lo, "hi": hi, "points": int(m.sum()),
                             "T": float(d[m].mean() / var) if m.any() else np.nan})
        for f in factors:
            v = x[f].astype(float).to_numpy()
            rho = float(spearmanr(v, d).statistic)
            perm = np.array([spearmanr(v, rng.permutation(d)).statistic for _ in range(n_perm)])
            corr.append({"output": y, "factor": f, "rho": rho,
                         "p": float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1))})
    return pd.DataFrame(rows), pd.DataFrame(corr)


def first_order_only(cls, K, off, y):
    """Class-based first-order index of every column for one output, as
    indices_one_output without the OT index (classes from column_classes)."""
    n, J = cls.shape
    var = y.var()
    if var == 0:
        return np.zeros(J)
    y = y - y.mean()
    g = (cls + off).ravel()
    nK = int(K.sum())
    cnt = np.bincount(g, minlength=nK)
    share = cnt / n
    col_of = np.repeat(np.arange(J), K)
    means = np.bincount(g, weights=np.repeat(y, J), minlength=nK) / np.maximum(cnt, 1)
    mcol = np.bincount(col_of, weights=share * means, minlength=J)
    return np.bincount(col_of, weights=share * (means - mcol[col_of]) ** 2, minlength=J) / var


def first_order_sums(X, Y, B=300, seed=0, outputs=OUTPUTS):
    """The first-order part of the variance of every output from the runs,
    without a surrogate: the sum over the factors other than the seeds of
    their class-based first-order indices (first_order_raw), less the floor
    of the estimator (first_order_runs). The floor of factor i is the index
    of its matched dummy, averaged over the outputs, times 1 - S_i: about
    0.001 per factor with 25 classes on 4,096 Sobol' points, against
    (K - 1)/n on independent points, because the points of each class are
    spread evenly over the other columns. Also leak_bound, the sum over the
    pairs of factors of d(i -> j) S_j, with d(i -> j) the index of column j
    on the classes of column i (design_dependence): an upper bound of the
    part of the effect of a factor j that the residual dependence of the
    design counts again in S_i (an upper bound because d holds the noise
    the floor already removes); the first-order part may be smaller by that
    much. With the standard error of the raw sum over B resamples of the
    runs (classes drawn again on each, as in the bootstrap of the
    indices)."""
    factors = [c for c in FACTOR_IDS if c not in SEEDS]
    fi = [COLUMNS.index(c) for c in factors]
    cls, K, off = column_classes(X)
    S = np.column_stack([first_order_only(cls, K, off, Y[:, o]) for o in range(Y.shape[1])])
    level = {d: float(S[COLUMNS.index(d)].mean()) for d in design.DUMMIES}
    floor = np.array([level[matched_dummy(c)] for c in factors])[:, None] * (1 - S[fi])
    dep = np.column_stack([first_order_only(cls, K, off, X[:, j]) for j in fi])[fi]   # d(i -> j)
    np.fill_diagonal(dep, 0.0)
    leak = dep @ S[fi]
    rng = np.random.default_rng(seed)
    boot = np.empty((B, Y.shape[1]))
    for b in range(B):
        idx = rng.integers(0, len(X), len(X))
        cb, Kb, ob = column_classes(X[idx][:, fi])
        boot[b] = [first_order_only(cb, Kb, ob, Y[idx, o]).sum() for o in range(Y.shape[1])]
    rows = []
    for o, y in enumerate(outputs):
        raw = float(S[fi, o].sum())
        rows.append({"output": y, "first_order_raw": raw, "floor": float(floor[:, o].sum()),
                     "first_order_runs": raw - float(floor[:, o].sum()),
                     "leak_bound": float(leak[:, o].sum()),
                     "first_order_se": float(boot[:, o].std(ddof=1))})
    return pd.DataFrame(rows)


def variance_budget(brackets, seeds, kmeans=None, first=None):
    """Budget of the variance of every output. From the model alone, with
    `first` (first_order_sums): the first-order effects of the factors other
    than the seeds (from the runs), the interactions among them by
    difference, I = 1 - T_seeds - first-order, with an interval from the
    standard errors of both, and the seeds (their total index from the
    reruns, variant seeds, split into the K-means seed and the draw of the
    country multipliers when the variant kmeans is given); the three add up
    to one. With the surrogate's brackets.csv, what it reproduces: its
    first-order effects (sum of its first-order indices) and interactions
    (the share it reproduces less that sum), the share of I that it
    apportions among the factors, and the rest of the factors' variance,
    split into first-order variance it compresses and interactions it does
    not apportion (not_attributed, by difference: a surrogate that
    reproduced E[Y | other factors] exactly would reproduce 1 - T_seeds)."""
    b = brackets[(brackets.game == "single") & (brackets.player != "D_cont")]
    explained = brackets.groupby("output").explained.mean()
    sur_first = b.groupby("output").S.sum()
    t = seeds.set_index("output")
    k = kmeans.set_index("output") if kmeans is not None else None
    f = first.set_index("output") if first is not None else None
    rows = []
    for y in OUTPUTS:
        if y not in t.index or (y not in explained.index and f is None):
            continue
        row = {"output": y, "seeds": float(t.loc[y, "T"]),
               "seeds_lo": float(t.loc[y, "lo"]), "seeds_hi": float(t.loc[y, "hi"])}
        if k is not None:
            row["kmeans"] = float(k.loc[y, "T"])
            row["multipliers"] = row["seeds"] - row["kmeans"]
        if f is not None and y in f.index:
            fo = f.loc[y]
            I = 1.0 - row["seeds"] - float(fo.first_order_runs)
            se = float(np.hypot(fo.first_order_se, (row["seeds_hi"] - row["seeds_lo"]) / (2 * 1.96)))
            row.update({"first_order_runs": float(fo.first_order_runs), "interactions_runs": I,
                        "interactions_lo": I - 1.96 * se, "interactions_hi": I + 1.96 * se,
                        "first_order_raw": float(fo.first_order_raw), "floor": float(fo.floor),
                        "leak_bound": float(fo.leak_bound), "first_order_se": float(fo.first_order_se)})
        if y in explained.index:
            row.update({"surrogate_first_order": float(sur_first[y]),
                        "surrogate_interactions": float(explained[y] - sur_first[y]),
                        "explained": float(explained[y]),
                        "not_attributed": 1.0 - float(explained[y]) - row["seeds"]})
            if "first_order_runs" in row:
                row["surrogate_share"] = (row["surrogate_interactions"] / row["interactions_runs"]
                                          if row["interactions_runs"] > 0 else np.nan)
                row["first_order_compressed"] = row["first_order_runs"] - row["surrogate_first_order"]
                row["interactions_not_apportioned"] = (row["interactions_runs"]
                                                       - row["surrogate_interactions"])
        rows.append(row)
    return pd.DataFrame(rows)


def totals_check(indices, totals, budget):
    """The total index of the surrogate of every factor other than the seeds
    against the bracket S_i <= T_i <= S_i + I of the model averaged over the
    seeds, from the first-order index of the runs (largest size) and the
    share I of the interactions in the budget: T_i adds to S_i only the
    interaction terms that hold factor i, which cannot exceed all of them.
    The flags say where the surrogate's T_i falls below S_i, beyond the
    intervals of both or not, and whether S_i reads as non-zero."""
    ind = indices[(indices.family == "S") & (indices.N == indices.N.max())]
    tot = totals[totals.N == totals.N.max()]
    I = budget.set_index("output")["interactions_runs"]
    rows = []
    for y in OUTPUTS:
        if y not in I.index:
            continue
        a = ind[ind.output == y].set_index("factor")
        t = tot[tot.output == y].set_index("player")
        for c in [c for c in FACTOR_IDS if c not in SEEDS]:
            if c not in a.index or c not in t.index:
                continue
            s, tt = a.loc[c], t.loc[c]
            rows.append({"output": y, "factor": c, "S": float(s.point), "S_lo": float(s.lo),
                         "S_hi": float(s.hi), "S_nonzero": str(s.nonzero) == "True",
                         "T": float(tt["T"]), "T_lo": float(tt.lo), "T_hi": float(tt.hi),
                         "upper": float(s.point + I[y]),
                         "T_below_S": bool(tt["T"] < s.point),
                         "T_below_S_beyond_intervals": bool(tt.hi < s.lo),
                         "T_above_upper": bool(tt["T"] > s.point + I[y])})
    return pd.DataFrame(rows)


def summary(table, baseline):
    """Each output over the runs (the uncertainty analysis), and the
    percentile of the canonical value among them (share of runs below it,
    ties counted half). Besides the mean, standard deviation and percentiles:
    the coefficient of variation, the skewness and excess kurtosis (bias-
    corrected), the bimodality coefficient of Pfister et al. (2013),
    BC = (g^2 + 1) / (k + 3 (n - 1)^2 / ((n - 2)(n - 3))), above 5/9 for a
    bimodal or strongly skewed distribution, and the share of the runs at the
    most frequent value (a spike, such as the zeros of Y2)."""
    from scipy.stats import kurtosis, skew
    rows = []
    for y in OUTPUTS:
        v = table[y].to_numpy(dtype=float)
        n = len(v)
        c = float(baseline.get(y, np.nan))
        g, k = float(skew(v, bias=False)), float(kurtosis(v, bias=False))
        vals, cnt = np.unique(v, return_counts=True)
        rows.append({"output": y, "canonical": c,
                     "canonical_percentile": 100 * (np.mean(v < c) + 0.5 * np.mean(v == c))
                     if np.isfinite(c) else np.nan,
                     "median": np.median(v), "p05": np.percentile(v, 5), "p95": np.percentile(v, 95),
                     "mean": v.mean(), "sd": v.std(ddof=1),
                     "min": v.min(), "max": v.max(), "cv": v.std(ddof=1) / abs(v.mean()),
                     "skewness": g, "excess_kurtosis": k,
                     "bimodality": (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))),
                     "mode": vals[cnt.argmax()], "mode_share": cnt.max() / n})
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


def hourly_ot(table, profiles, B, seed, adjusted=False):
    """OT and first-order indices (plain, or adjusted) of the factors and
    dummies for the hourly means of the pooled profile, by day type, with the
    95th percentile of the matched dummies over B resamples."""
    from gsa.benchmarks import CLOCK_HOUR
    X = table[COLUMNS].to_numpy(dtype=float)
    dummy_of = {c: matched_dummy(c) for c in FACTOR_IDS}
    rows = []
    for d, day in enumerate(["working", "non_working", "all"]):
        prof = profiles[:, -1, d, :]                                      # ALL countries
        H = np.column_stack([prof[:, CLOCK_HOUR == h].mean(axis=1) for h in range(24)])
        boot = boot_adjusted_indices if adjusted else boot_plain_indices
        raw, (S, O), G, reps, _ = boot(X, H, B, seed + d)
        for j, c in enumerate(COLUMNS):
            dj = COLUMNS.index(dummy_of.get(c, c))
            for h in range(24):
                rows.append({"day_type": day, "hour": h, "factor": c, "ot": O[j, h],
                             "dummy_95": float(np.percentile(reps[1, :, dj, h], 95)),
                             "S": S[j, h], "S_dummy_95": float(np.percentile(reps[0, :, dj, h], 95)),
                             "ot_raw": raw[1][j, h], "imbalance": G[j, h]})
    return pd.DataFrame(rows)


PARTS = ("summary", "dependence", "indices", "knn", "surrogate", "profiles", "uniform", "otparts",
         "brackets", "seeds", "totals")


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


def part_indices(X, Y, out, sizes, B, log, adjusted=False):
    """First-order and OT indices on the nested designs, with the statistics
    of Sarrazin et al. (2016): the plain class-based estimators, or with
    adjusted=True those reweighted by raking on the leading drivers (the raw
    ones beside them)."""
    dummy_of = {c: matched_dummy(c) for c in FACTOR_IDS}
    f_idx = [COLUMNS.index(c) for c in FACTOR_IDS]
    rows, conv, prev = [], [], {}
    for N in sizes:
        t0 = time.time()
        if adjusted:
            raw, adj, G, reps, drivers = boot_adjusted_indices(X[:N], Y[:N], B, seed=N)
        else:
            raw, adj, G, reps, drivers = boot_plain_indices(X[:N], Y[:N], B, seed=N)
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
        log(f"  {'adjusted' if adjusted else 'plain'} class-based indices, {N} runs: "
            f"{time.time() - t0:.0f} s")
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


def part_profiles(share, table, out, N, B, log, adjusted=False):
    z = np.load(Path(share) / "gsa_profiles.npz")
    order = np.argsort(z["runs"])
    profiles = z["profiles"][order][:N].astype(np.float64)
    save_bands(out, profiles, z)
    hourly_ot(table.iloc[:N], profiles, B, seed=7, adjusted=adjusted).to_csv(out / "hourly.csv", index=False)
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


def part_seeds(table, reseed, out, log):
    """Total indices of the seeds from the reruns of gsa.reseed (the share
    folders RESEED/seeds and, if present, RESEED/kmeans and RESEED/same), and
    the variance budget with the surrogate's brackets.csv when it is in OUT."""
    reseed = Path(reseed)
    frames, totals = [], {}
    for which in ("seeds", "kmeans", "same"):
        f = reseed / which / "gsa_outputs.csv"
        if f.is_file():
            totals[which] = seed_totals(table, pd.read_csv(f))
            frames.append(totals[which].assign(variant=which))
    if not frames:
        log(f"  no reruns in {reseed}")
        return
    pd.concat(frames).to_csv(out / "seeds.csv", index=False)
    disp, corr = [], []
    for which in ("seeds", "kmeans"):
        f = reseed / which / "gsa_outputs.csv"
        if f.is_file():
            d, c = seed_dispersion(table, pd.read_csv(f))
            disp.append(d.assign(variant=which))
            corr.append(c.assign(variant=which))
    if disp:
        pd.concat(disp).to_csv(out / "seed_dispersion.csv", index=False)
        pd.concat(corr).to_csv(out / "seed_modulators.csv", index=False)
    if "same" in totals and totals["same"]["T"].abs().max() > 0:
        log("  WARNING: the reruns of variant same differ from the base runs")
    if "seeds" in totals:
        first = first_order_sums(table[COLUMNS].to_numpy(dtype=float), table[OUTPUTS].to_numpy(dtype=float))
        first.to_csv(out / "first_order_sums.csv", index=False)
        brackets = pd.read_csv(out / "brackets.csv") if (out / "brackets.csv").is_file() else \
            pd.DataFrame(columns=["output", "game", "player", "S", "explained"])
        budget = variance_budget(brackets, totals["seeds"], totals.get("kmeans"), first)
        budget.to_csv(out / "budget.csv", index=False)
        if (out / "indices.csv").is_file() and (out / "totals.csv").is_file():
            check = totals_check(pd.read_csv(out / "indices.csv"), pd.read_csv(out / "totals.csv"), budget)
            check.to_csv(out / "totals_check.csv", index=False)
            log(f"  surrogate T below the runs' S in {int(check.T_below_S.sum())} of {len(check)} pairs "
                f"({int((check.T_below_S & check.S_nonzero).sum())} with S read as non-zero, "
                f"{int(check.T_below_S_beyond_intervals.sum())} beyond both intervals); "
                f"above S + I in {int(check.T_above_upper.sum())}")


def run_all(share, out, uniform=None, B=500, sizes=SIZES, n_perm=10000, B_surrogate=50,
            parts=PARTS, log=print, reseed=None, adjusted=False):
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
        part_indices(X, Y, out, sizes, B, log, adjusted=adjusted)
    if "knn" in parts:
        part_knn(table, Y, out, sizes, B, log)
    if "surrogate" in parts:
        part_surrogate(table, Y, out, sizes, B_surrogate, n_perm, log)
    if "profiles" in parts and (share / "gsa_profiles.npz").is_file():
        part_profiles(share, table, out, sizes[-1], min(B, 100), log, adjusted=adjusted)
    if "uniform" in parts and uniform is not None:
        part_uniform(table, uniform, out, min(B, 200), n_perm, log)
    if "otparts" in parts:
        ot_parts(X, Y, adjusted=adjusted).to_csv(out / "ot_parts.csv", index=False)
    if "totals" in parts:
        t0 = time.time()
        tot, conv = surrogate_totals(table, Y, sizes, B=B_surrogate)
        tot.to_csv(out / "totals.csv", index=False)
        conv.to_csv(out / "convergence_totals.csv", index=False)
        log(f"  surrogate total indices: {time.time() - t0:.0f} s")
    if "brackets" in parts:
        t0 = time.time()
        surrogate_brackets(table, Y, n_perm=10 * n_perm).to_csv(out / "brackets.csv", index=False)
        log(f"  surrogate brackets: {time.time() - t0:.0f} s")
    if "seeds" in parts and reseed is not None:
        part_seeds(table, reseed, out, log)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--share", type=Path, required=True)
    parser.add_argument("--uniform", type=Path, default=None)
    parser.add_argument("--reseed", type=Path, default=None,
                        help="folder of the shares of gsa.reseed (subfolders seeds, kmeans, same)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--B", type=int, default=500)
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)),
                        help="nested designs to analyse (those above the number of runs are skipped)")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--B-surrogate", type=int, default=50)
    parser.add_argument("--adjusted", action="store_true",
                        help="reweight the runs of each class by raking on the leading drivers (the "
                             "plain class-based estimators otherwise)")
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
            B_surrogate=args.B_surrogate, parts=parts, reseed=args.reseed, adjusted=args.adjusted)
    print(f"Done in {time.time() - t0:.0f} s; results in {args.out}")


if __name__ == "__main__":
    main()

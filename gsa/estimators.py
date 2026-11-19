"""Given-data estimators of the sensitivity analysis.

All three families of indices are computed from one sample of runs (X, Y):

first_order    first-order Sobol' index S_i, class-based: one class per level
               for factors with at most MAX_LEVELS levels, otherwise
               min(25, floor(sqrt n)) quantile classes (R type 7, right-closed
               as cut(include.lowest = TRUE)), variances with divisor n
ot_index       optimal-transport index iota_i = E[W2^2(P_Y, P_Y|X_i)] / (2 Var Y)
               with the same classes and the exact one-dimensional W2 between
               the whole sample and each class (w2sq_exact), divisor n, so that
               S_i / 2 <= iota_i holds exactly in every sample
shapley_knn    Shapley effects with the nearest-neighbour given-data estimator
               of Broto, Bachoc and Depecker (2020): the cost of a set u of
               players, E[Var(Y | X_-u)], is the mean over the runs of the
               variance of Y among the k nearest runs in the coordinates of
               the other players (the run itself included), and the effects
               follow from the Shapley weights; players may be groups of
               factors

The conventions follow the earlier OT-GSA work (w2_exact, class-based weights)
and the design note of 19 November 2026.
"""
import math

import numpy as np

MAX_LEVELS = 25


def n_quantile_classes(n):
    return max(1, min(25, int(math.floor(math.sqrt(n)))))


def classes(x, max_levels=MAX_LEVELS, n_classes=None):
    """Class of every run for one factor: its level if the factor has at
    most `max_levels` levels, else its quantile class (R type 7 quantiles,
    right-closed intervals, the lowest included)."""
    x = np.asarray(x, dtype=float)
    levels = np.unique(x)
    if len(levels) <= max_levels:
        return np.searchsorted(levels, x)
    m = n_classes or n_quantile_classes(len(x))
    edges = np.quantile(x, np.linspace(0.0, 1.0, m + 1))     # numpy 'linear' = R type 7
    return np.searchsorted(edges[1:-1], x, side="left")


def first_order(x, y, **kw):
    """Class-based first-order index sum_m (n_m/n)(ybar_m - ybar)^2 / Var(Y)."""
    y = np.asarray(y, dtype=float)
    c = classes(x, **kw)
    n = len(y)
    var = y.var()
    if var == 0:
        return 0.0
    counts = np.bincount(c)
    means = np.bincount(c, weights=y) / np.where(counts > 0, counts, 1)
    keep = counts > 0
    return float(np.sum(counts[keep] / n * (means[keep] - y.mean()) ** 2) / var)


def w2sq_exact(a, b):
    """Exact squared 2-Wasserstein distance between the empirical
    distributions of two samples of any sizes: the integral over (0, 1) of
    the squared difference of their quantile functions, which are step
    functions, so the integral is a finite sum over the merged steps
    (integer arithmetic for the breakpoints)."""
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    na, nb = len(a), len(b)
    t = np.union1d(np.arange(1, na + 1, dtype=np.int64) * nb, np.arange(1, nb + 1, dtype=np.int64) * na)
    lengths = np.diff(np.r_[0, t]) / (na * nb)
    ia, ib = (t - 1) // nb, (t - 1) // na
    return float(np.sum(lengths * (a[ia] - b[ib]) ** 2))


def ot_index(x, y, **kw):
    """Class-based OT index sum_m (n_m/n) W2^2(Y, Y_m) / (2 Var Y)."""
    y = np.asarray(y, dtype=float)
    c = classes(x, **kw)
    n = len(y)
    var = y.var()
    if var == 0:
        return 0.0
    total = 0.0
    for m in np.unique(c):
        ym = y[c == m]
        total += len(ym) / n * w2sq_exact(y, ym)
    return float(total / (2 * var))


def indices_all(X, y, **kw):
    """First-order and OT indices of every column of X, with the bound
    S_i / 2 <= iota_i checked."""
    X = np.asarray(X)
    s = np.array([first_order(X[:, j], y, **kw) for j in range(X.shape[1])])
    o = np.array([ot_index(X[:, j], y, **kw) for j in range(X.shape[1])])
    if np.any(s / 2 > o + 1e-12):
        raise AssertionError("S_i / 2 > iota_i for some factor: estimator error")
    return s, o


# ---------------------------------------------------------------- Shapley

def _popcount(v):
    v = np.array(v, dtype=np.int64)          # a copy: the loop shifts it in place
    c = np.zeros_like(v)
    while np.any(v):
        c += v & 1
        v >>= 1
    return c


def shapley_weights(d):
    """Matrix M of shape (d, 2^d) such that Sh = M @ W for a cost vector W
    indexed by bit masks of players: Sh_i = sum over u without i of
    |u|!(d-|u|-1)!/d! (W[u + i] - W[u])."""
    masks = np.arange(2 ** d, dtype=np.int64)
    size = _popcount(masks)
    fact = np.array([math.factorial(k) for k in range(d + 1)], dtype=float)
    M = np.zeros((d, 2 ** d))
    for i in range(d):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        w = fact[size[without]] * fact[d - size[without] - 1] / fact[d]
        M[i, without | bit] += w
        M[i, without] -= w
    return M


def knn_costs(U, y, players, k=3, refresh=512):
    """Per-run local variances of Y among the k nearest runs in the
    coordinates of the players outside u, for every set u of players (bit
    mask), as an array of shape (2^d, n). Row 0 (u empty) is 0, since the
    model is deterministic given all its factors; the row of the full set
    is (y - mean)^2 * n / (n - 1), whose mean is the sample variance with
    divisor n - 1. Subsets are visited in Gray-code order, so each distance
    matrix differs from the previous one by one player's distances."""
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = len(y), len(players)
    full = (1 << d) - 1
    Dp = []
    for cols in players:
        D = np.zeros((n, n))
        for c in np.atleast_1d(cols):
            D += (U[:, c][:, None] - U[:, c][None, :]) ** 2
        Dp.append(D)
    L = np.zeros((2 ** d, n), dtype=np.float64)
    L[full] = (y - y.mean()) ** 2 * n / (n - 1)
    cur = np.sum(Dp, axis=0)        # distances on every player: u empty
    mask = 0
    for g in range(1, 2 ** d):
        bit = (g & -g).bit_length() - 1
        mask ^= 1 << bit
        if mask & (1 << bit):
            cur -= Dp[bit]
        else:
            cur += Dp[bit]
        if g % refresh == 0:        # against drift of the running sum
            cur = np.sum([Dp[j] for j in range(d) if not mask & (1 << j)], axis=0) \
                if mask != full else np.zeros((n, n))
        if mask == full:
            continue
        nn = np.argpartition(cur, k - 1, axis=1)[:, :k]
        L[mask] = y[nn].var(axis=1, ddof=1)
    return L


def shapley_from_local(L, counts=None):
    """Shapley effects from the local variances of knn_costs; `counts`
    (multiplicities of the runs, for a bootstrap) default to ones."""
    n = L.shape[1]
    d = int(round(math.log2(L.shape[0])))
    w = np.ones(n) if counts is None else np.asarray(counts, dtype=float)
    W = L @ w / w.sum()
    M = shapley_weights(d)
    return (M @ W) / W[-1]


def shapley_knn(U, y, players, k=3):
    """Shapley effects of the players (lists of columns of U, the uniform
    coordinates of the factors) for output y. They add up to one."""
    return shapley_from_local(knn_costs(U, y, players, k=k))

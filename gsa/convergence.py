"""Bootstrap intervals, dummies and convergence of the sensitivity indices.

For each nested design size N (128, 256, 512, 1,024, ...), the indices are
computed on the first N runs and on B bootstrap resamples of them. An index
counts as non-zero when the lower end of its 95% interval lies above the 95th
percentile of the bootstrap distribution of its matched dummy. Convergence is
judged by the three statistics of Sarrazin, Pianosi and Wagener (2016):

ranking     95th percentile over the resamples of the adjusted rank
            difference rho_s = sum_j |R_j^b - R_j| max(S_j^b, S_j)^2 /
            sum_k max(S_k^b, S_k)^2 between each resample's ranking and that
            of the N runs; converged below 1
screening   widest 95% interval among the factors read as zero; below 0.05
indices     widest 95% interval among the factors read as non-zero; below 0.05

together with a change of less than 0.02 in every index between the last two
sizes. The iid bootstrap ignores the stratification of the Sobol' design, so
its intervals are conservative.
"""
import numpy as np
from scipy.stats import rankdata

from . import estimators as est


def class_indices(X, y, kind):
    """First-order ('S') or OT ('OT') indices of every column of X."""
    f = est.first_order if kind == "S" else est.ot_index
    return np.array([f(X[:, j], y) for j in range(X.shape[1])])


def bootstrap(X, y, kind, B=500, seed=0):
    """Point estimates and B bootstrap replicates (resampled runs)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = class_indices(X, y, kind)
    reps = np.empty((B, X.shape[1]))
    for b in range(B):
        idx = rng.integers(0, n, n)
        reps[b] = class_indices(X[idx], y[idx], kind)
    return point, reps


def adjusted_rank_difference(s0, s1):
    """rho_s of Sarrazin et al. (2016) between two vectors of indices."""
    r0 = rankdata(-s0, method="average")
    r1 = rankdata(-s1, method="average")
    w = np.maximum(s0, s1) ** 2
    return float(np.sum(np.abs(r1 - r0) * w) / np.sum(w)) if np.sum(w) > 0 else 0.0


def assess(point, reps, factor_cols, dummy_col):
    """Intervals, the non-zero reading against the dummy, and the three
    convergence statistics over the factor columns."""
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    threshold = np.percentile(reps[:, dummy_col], 95)
    nonzero = lo[factor_cols] > threshold
    width = (hi - lo)[factor_cols]
    rho = [adjusted_rank_difference(point[factor_cols], r[factor_cols]) for r in reps]
    return {
        "lo": lo, "hi": hi, "dummy_threshold": float(threshold), "nonzero": nonzero,
        "stat_ranking": float(np.percentile(rho, 95)),
        "stat_screening": float(width[~nonzero].max()) if (~nonzero).any() else 0.0,
        "stat_indices": float(width[nonzero].max()) if nonzero.any() else 0.0,
    }


def nested(X, y, sizes, kind, factor_cols, dummy_col, B=500, seed=0):
    """Indices, intervals and statistics for each nested size."""
    out = []
    for N in sizes:
        point, reps = bootstrap(X[:N], y[:N], kind, B=B, seed=seed + N)
        a = assess(point, reps, factor_cols, dummy_col)
        out.append({"N": N, "point": point, **a})
    last, prev = out[-1]["point"][factor_cols], out[-2]["point"][factor_cols] if len(out) > 1 else None
    for r in out:
        r["converged"] = (r["stat_ranking"] < 1 and r["stat_screening"] < 0.05
                          and r["stat_indices"] < 0.05)
    if prev is not None:
        out[-1]["max_change"] = float(np.max(np.abs(last - prev)))
        out[-1]["converged"] = out[-1]["converged"] and out[-1]["max_change"] < 0.02
    return out

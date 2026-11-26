"""Principal components of the global clustering, as the manuscript reports
them (Section 3.2): how many components 80, 90 and 95% of the variance need,
the share of each of the first five, and the window of the day on which each
loads most strongly.

    python3 tools/pca_summary.py [--processed FOLDER | --bundle BUNDLE] [--n 5]

It reads the power profiles of prepared_clustering_data.csv (Stage 04) from
the processed folder of a run, standardises every slot as Stage 05 does, and
prints aggregates over all household-days only: shares of variance, and for
each component the clock time of its largest absolute loading and the
contiguous block of slots around it whose absolute loadings reach half of
that largest one. The diary day starts at 04:00 (slot 0).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from run_folder import describe, resolve, stop  # noqa: E402

ENERGY = [f"Household_Energy_W_{t}" for t in range(144)]


def clock(t):
    m = (4 * 60 + 10 * int(t)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def window(loading):
    """Slot of the largest absolute loading and the contiguous block around it
    whose absolute loadings reach half of it, as clock times."""
    a = np.abs(np.asarray(loading, dtype=float))
    k = int(a.argmax())
    lo = hi = k
    while lo - 1 >= 0 and a[lo - 1] >= a[k] / 2:
        lo -= 1
    while hi + 1 < len(a) and a[hi + 1] >= a[k] / 2:
        hi += 1
    return k, lo, hi


def summary(X, n=5):
    """Shares of variance and windows of the first n components of the
    standardised profiles X (household-days x 144 slots)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    need = {p: int(np.searchsorted(cum, p / 100) + 1) for p in (80, 90, 95)}
    comps = []
    for i in range(n):
        k, lo, hi = window(pca.components_[i])
        comps.append({"component": i + 1, "share_pct": 100 * pca.explained_variance_ratio_[i],
                      "largest": clock(k), "from": clock(lo), "to": clock(hi)})
    return need, pd.DataFrame(comps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    processed = resolve(args.processed, args.bundle)
    path = Path(processed) / "prepared_clustering_data.csv"
    if not path.is_file():
        stop(f"{path} not found: give the processed folder of the run (--processed or --bundle).")
    print(describe(processed))
    X = pd.read_csv(path, usecols=ENERGY, dtype={c: np.float32 for c in ENERGY})[ENERGY].to_numpy(dtype=float)
    need, comps = summary(X, args.n)
    print(f"\nComponents for 80, 90 and 95% of the variance: {need[80]}, {need[90]}, {need[95]}")
    print("\nFirst components (share of variance, largest absolute loading and its block):")
    for _, r in comps.iterrows():
        print(f"  {int(r.component)}: {r.share_pct:.2f}%, largest at {r.largest}, block {r['from']} to {r.to}")


if __name__ == "__main__":
    main()

"""Stability of the stratified solutions and other clustering methods
(Reviewer 1, comment 3).

Stage 05 assesses the stability of the global solution (silhouette width and
ten half-samples, HETUS_STABILITY=1), and the sensitivity analysis varies K,
the share of variance kept by PCA and the seed of K-means. This script adds
the two checks the reviewer also asks for:

  the same stability measures for every stratified solution of Stage 06 (by
  day type, by country, by country and day type): silhouette width on up to
  10,000 household-days, and over ten half-samples the adjusted Rand index
  between K-means refitted on the half-sample and the solution restricted to
  it, with the share of clusters whose mean Jaccard similarity with their best
  match stays at 0.75 or above (stable) or falls below 0.5 (dissolved), after
  Hennig (2007);
  other clustering methods on the global solution, with the same
  standardisation and PCA as Stage 05: a Gaussian mixture with diagonal
  covariances and K components, K-means on the standardised profiles without
  PCA, and Ward's agglomerative clustering on a random subsample of 8,000
  household-days, each against the K-means partition (adjusted Rand index),
  with its silhouette width.

    python3 tools/clustering_robustness.py [--processed FOLDER] [--reps 10]
                                           [--skip-methods] [--skip-strata]

It reads the *_clustering_results.csv files of Stages 05 and 06 (the power
profiles and the cluster of every household-day) from the processed folder
of a run, so it runs on the machine that holds the microdata, and writes
clustering_robustness.csv next to cluster_stability.csv. Only measures of
whole solutions are written (no cluster sizes or profiles); a solution of
fewer than 50 household-days is left out.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from disclosure import MIN_N_UNFLAGGED  # noqa: E402

ENERGY = [f"Household_Energy_W_{t}" for t in range(144)]
PCA_VARIANCE = float(os.environ.get("HETUS_PCA_VARIANCE", "0.95"))


def scores(sol):
    """Standardised power profiles and their PCA scores, as Stages 05 and 06."""
    X = StandardScaler().fit_transform(sol[ENERGY].to_numpy(dtype=float))
    return X, PCA(n_components=PCA_VARIANCE).fit_transform(X)


def half_sample_stability(Z, labels, reps=10, seed=42):
    """Silhouette, and the ARI and cluster-wise Jaccard similarity of K-means
    refitted on half-samples, as Stage 05 (HETUS_STABILITY=1)."""
    k = int(labels.max()) + 1
    sil = silhouette_score(Z, labels, sample_size=min(10000, len(Z)), random_state=seed) if k > 1 else np.nan
    rng = np.random.default_rng(seed)
    aris, jac = [], np.full((reps, k), np.nan)
    for b in range(reps):
        idx = rng.choice(len(Z), size=len(Z) // 2, replace=False)
        refit = KMeans(n_clusters=k, random_state=100 + b, n_init=5).fit(Z[idx])
        full, new = labels[idx], refit.labels_
        aris.append(adjusted_rand_score(full, new))
        for c in range(k):
            in_c = full == c
            if in_c.any():
                jac[b, c] = max(np.sum(in_c & (new == m)) / np.sum(in_c | (new == m)) for m in range(k))
    jm = np.nanmean(jac, axis=0)
    return {"silhouette": float(sil), "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)),
            "share_stable_clusters": float(np.mean(jm >= 0.75)),
            "share_dissolved_clusters": float(np.mean(jm < 0.5))}


def other_methods(X, Z, labels, seed=42, ward_n=8000):
    """Other methods against the K-means partition of the same data."""
    k = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    out = []
    gm = GaussianMixture(n_components=k, covariance_type="diag", n_init=2, random_state=seed).fit(Z)
    g = gm.predict(Z)
    out.append(("Gaussian mixture, diagonal, on the PCA scores", adjusted_rand_score(labels, g),
                silhouette_score(Z, g, sample_size=min(10000, len(Z)), random_state=seed)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    out.append(("K-means on the standardised profiles, no PCA", adjusted_rand_score(labels, km),
                silhouette_score(X, km, sample_size=min(10000, len(X)), random_state=seed)))
    idx = rng.choice(len(Z), size=min(ward_n, len(Z)), replace=False)
    w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z[idx])
    out.append((f"Ward, agglomerative, on {len(idx):,} household-days", adjusted_rand_score(labels[idx], w),
                silhouette_score(Z[idx], w, random_state=seed)))
    return [{"solution": "clustering_results.csv", "method": m, "ari_with_kmeans": float(a),
             "silhouette": float(s)} for m, a, s in out]


def solution_files(processed):
    p = Path(processed)
    files = [p / "DayType_Analysis" / "Working_Days_clustering_results.csv",
             p / "DayType_Analysis" / "Non_Working_Days_clustering_results.csv"]
    files += [Path(f) for f in sorted(glob.glob(str(p / "Country_Analysis" / "[A-Z][A-Z]_clustering_results.csv")))]
    for tag in ("working", "non_working"):
        files += [Path(f) for f in sorted(glob.glob(str(p / "Country_DayType_Analysis" /
                                                         f"[A-Z][A-Z]_{tag}_days_clustering_results.csv")))]
    return [f for f in files if f.is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--skip-methods", action="store_true")
    parser.add_argument("--skip-strata", action="store_true")
    args = parser.parse_args()
    from paths import PROCESSED_DIR
    processed = Path(args.processed or PROCESSED_DIR)
    rows = []
    glob_file = processed / "clustering_results.csv"
    if not args.skip_methods and glob_file.is_file():
        sol = pd.read_csv(glob_file, usecols=ENERGY + ["cluster"])
        X, Z = scores(sol)
        labels = sol.cluster.to_numpy()
        rows.append({"solution": glob_file.name, "method": "K-means (Stage 05), half-samples",
                     "k": int(labels.max()) + 1, **half_sample_stability(Z, labels, args.reps)})
        for r in other_methods(X, Z, labels):
            rows.append({**r, "k": int(labels.max()) + 1})
            print(f"{r['method']}: ARI with K-means {r['ari_with_kmeans']:.3f}, silhouette {r['silhouette']:.3f}")
    if not args.skip_strata:
        for f in solution_files(processed):
            sol = pd.read_csv(f, usecols=ENERGY + ["cluster"])
            if len(sol) < MIN_N_UNFLAGGED or sol.cluster.nunique() < 2:
                continue
            _, Z = scores(sol)
            labels = sol.cluster.to_numpy()
            st = half_sample_stability(Z, labels, args.reps)
            rows.append({"solution": f.name, "method": "K-means (Stage 06), half-samples",
                         "k": int(labels.max()) + 1, **st})
            print(f"{f.name}: K {labels.max() + 1}, ARI {st['ari_mean']:.3f} (min {st['ari_min']:.3f}), "
                  f"silhouette {st['silhouette']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(processed / "clustering_robustness.csv", index=False)
    print(f"Written to {processed / 'clustering_robustness.csv'}")


if __name__ == "__main__":
    main()

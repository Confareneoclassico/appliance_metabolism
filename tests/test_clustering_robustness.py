"""tools/clustering_robustness.py on three well-separated groups of profiles:
every method finds them and every half-sample keeps them."""
import importlib.util
import sys

import numpy as np
import pandas as pd

from conftest import ROOT

spec = importlib.util.spec_from_file_location("clustering_robustness", ROOT / "tools" / "clustering_robustness.py")
cr = importlib.util.module_from_spec(spec)
sys.modules["clustering_robustness"] = cr
spec.loader.exec_module(cr)


def synthetic(rng, n=600):
    labels = rng.integers(0, 3, n)
    prof = np.full((n, 144), 50.0)
    for k, (a, b) in enumerate(((20, 40), (60, 80), (100, 120))):
        prof[labels == k, a:b] += 800
    prof += rng.normal(0, 20, prof.shape)
    return pd.DataFrame(prof, columns=cr.ENERGY).assign(cluster=labels)


def test_separated_groups_are_stable_and_found_by_every_method():
    sol = synthetic(np.random.default_rng(2))
    X, Z = cr.scores(sol)
    labels = sol.cluster.to_numpy()
    st = cr.half_sample_stability(Z, labels, reps=3)
    assert st["ari_mean"] > 0.99 and st["share_stable_clusters"] == 1.0
    assert st["share_dissolved_clusters"] == 0.0 and st["silhouette"] > 0.2
    for r in cr.other_methods(X, Z, labels):
        assert r["ari_with_kmeans"] > 0.99, r


def test_files_of_the_strata(tmp_path):
    for sub, name in (("DayType_Analysis", "Working_Days_clustering_results.csv"),
                      ("Country_Analysis", "BE_clustering_results.csv"),
                      ("Country_DayType_Analysis", "BE_working_days_clustering_results.csv")):
        (tmp_path / sub).mkdir(exist_ok=True)
        (tmp_path / sub / name).write_text("x\n")
    (tmp_path / "Country_Analysis" / "PL_ALL_DATA_clustering_results.csv").write_text("x\n")
    names = [f.name for f in cr.solution_files(tmp_path)]
    assert names == ["Working_Days_clustering_results.csv", "BE_clustering_results.csv",
                     "BE_working_days_clustering_results.csv"]

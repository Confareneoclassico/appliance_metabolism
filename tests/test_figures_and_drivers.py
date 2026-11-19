"""Country cluster figure with up to ten clusters (L8) and the merged groups of
Option A, and the driver figures of Stage 11: clusters on the x axis with bars
that add up to 100% (D8), variables from the Stage 07 ranking (L16)."""
import importlib

import numpy as np
import pandas as pd

import plot_country_cluster_profiles as country_figure

stage11 = importlib.import_module("11_visualize_cluster_drivers")


def test_ordinals_and_legend_beyond_eight_ranks():
    assert [country_figure.ordinal(k) for k in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]
    colours, labels = country_figure.legend_entries(10)
    assert len(colours) == len(labels) == 9
    assert labels[0] == "1st (largest)" and labels[-1] == "9th and smaller"
    assert len(set(colours)) == 9              # no colour reused


def test_country_figure_with_ten_clusters(tmp_path, monkeypatch):
    rows = []
    rng = np.random.default_rng(0)
    for rank in range(1, 11):
        row = {"stratum": "PL", "stratum_n": 5500, "cluster": rank - 1, "size_rank": rank,
               "n_small_clusters_merged": 2 if rank == 1 else 0, "n": 1000 - 50 * rank,
               "n_class": "", "withheld": False}
        row.update({f"P_{t:03d}": float(rng.gamma(2, 100)) for t in range(144)})
        rows.append(row)
    path = tmp_path / "export.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(country_figure, "FIGURES_DIR", tmp_path)
    country_figure.main(path)                   # used to raise IndexError
    assert (tmp_path / "country_cluster_profiles_small_multiples.png").is_file()


def test_driver_composition_adds_up_by_cluster():
    df = pd.DataFrame({"cluster": np.repeat([0, 1, 2], [300, 200, 10]),
                       "HHC5": np.r_[np.tile([1, 2, 3], 100), np.tile([1, 2], 100),
                                     np.full(10, 3)]})
    pct, small, largest = stage11.cluster_composition(df, "HHC5")
    assert list(pct.index) == [0, 1]           # cluster 2 is shown with cluster 0
    assert small == [2] and largest == 0
    assert np.allclose(pct.sum(axis=1), 100)
    assert np.isclose(pct.loc[1, 2], 50.0)     # half of cluster 1 has two adults


def test_driver_variables_from_the_stage07_ranking(tmp_path):
    tests = pd.DataFrame({
        "file": ["clustering_results.csv"] * 4 + ["PL_clustering_results.csv"],
        "Feature": ["HHC5", "COUNTRY", "INC2", "HHQ9_1", "INC1"],
        "Cramers_V": [0.17, 0.14, 0.07, np.nan, 0.9],
        "Publication_Class": ["", "", "", "withheld", ""],
    })
    path = tmp_path / "chi2.csv"
    tests.to_csv(path, index=False)
    assert stage11.top_features(path, top_n=2) == ["HHC5", "COUNTRY"]
    assert stage11.top_features(path, top_n=6) == ["HHC5", "COUNTRY", "INC2"]
    assert stage11.top_features(tmp_path / "none.csv") == stage11.FALLBACK_FEATURES

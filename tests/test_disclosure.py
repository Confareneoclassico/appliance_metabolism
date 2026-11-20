"""Publication rules (disclosure.py): Option A (D1, D2), the intruder tests run
before every export, the stratum guard of Stage 06 and the size checks of the
summary writers (L5)."""
import numpy as np
import pandas as pd
import pytest

from disclosure import (MIN_N_UNFLAGGED, assert_sizes, check_export, check_strata,
                        pool_tail, protect, published_groups, recovered_size)


def profile_table(sizes, rng, stratum="PL"):
    """Mean profiles of whole-watt household-days for groups of the given
    sizes, as an intruder would see them."""
    rows = []
    for g, n in enumerate(sizes):
        e = rng.integers(0, 3000, size=(n, 144))
        row = {"stratum": stratum, "stratum_n": int(sum(sizes)), "cluster": g, "n": n}
        row.update({f"P_{t:03d}": e[:, t].mean() for t in range(144)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_published_groups_merge_small_clusters_into_the_largest():
    labels = [0] * 300 + [1] * 120 + [2] * 30 + [3] * 7
    groups, small, largest = published_groups(labels)
    assert largest == 0 and small == [2, 3]
    sizes = pd.Series(groups).value_counts()
    assert sizes.to_dict() == {0: 337, 1: 120}
    assert sizes.min() >= MIN_N_UNFLAGGED and sizes.sum() == len(labels)


def test_published_groups_with_every_cluster_small():
    groups, small, largest = published_groups([0] * 30 + [1] * 25)
    assert set(groups) == {0} and small == [1]


def test_integer_means_reveal_the_group_size():
    rng = np.random.default_rng(0)
    e = rng.integers(0, 3000, size=(37, 144))
    assert recovered_size(e.mean(axis=0), n_max=500) == 37


def test_check_export_rejects_means_of_a_small_group():
    rng = np.random.default_rng(1)
    table = profile_table([400, 37], rng)
    table.loc[1, "n"] = np.nan            # the size withheld, the means not
    with pytest.raises(ValueError, match="reveal a group of 37"):
        check_export(table, n_col="n", stratum_col="stratum", stratum_total_col="stratum_n")


def test_check_export_rejects_a_small_remainder():
    rng = np.random.default_rng(2)
    table = profile_table([400, 120, 9], rng).iloc[:2].copy()   # the third group left out
    for col in [c for c in table if c.startswith("P_")]:
        table[col] = np.nan                                       # sizes only
    with pytest.raises(ValueError, match="leave 9 household-days"):
        check_export(table, n_col="n", stratum_col="stratum", stratum_total_col="stratum_n")


def test_check_export_rejects_a_small_stratum():
    rng = np.random.default_rng(3)
    table = profile_table([49], rng, stratum="LU")
    for col in [c for c in table if c.startswith("P_")]:
        table[col] = np.nan
    with pytest.raises(ValueError, match="stratum of fewer than"):
        check_export(table, n_col="n", stratum_col="stratum", stratum_total_col="stratum_n")


def test_check_export_accepts_option_a_groups():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(4), [300, 120, 30, 7])
    groups, _, _ = published_groups(labels)
    sizes = pd.Series(groups).value_counts().sort_index()
    table = profile_table(sizes.tolist(), rng)
    check_export(table, n_col="n", stratum_col="stratum", stratum_total_col="stratum_n")


def test_protect_blanks_the_profiles_of_flagged_groups():
    rng = np.random.default_rng(5)
    table = profile_table([400, 30], rng)
    kept = protect(table, n_col="n")
    blanked = protect(table, n_col="n", flagged_results=False)
    assert kept.loc[1, "P_000"] == table.loc[1, "P_000"]    # flagged, results kept
    assert np.isnan(blanked.loc[1, "P_000"])                # results left empty
    assert pd.isna(blanked.loc[1, "n"]) and blanked.loc[1, "n_class"] == "20-49"
    assert blanked.loc[0, "n"] == 400                         # rounded to 10


def test_check_strata_stops_on_a_small_stratum():
    counts = pd.Series({("BE", "working"): 946, ("LU", "non_working"): 17,
                        ("LU", "undetermined"): 0})
    with pytest.raises(ValueError, match="17"):
        check_strata(counts)
    check_strata(counts.drop(("LU", "non_working")))


def test_assert_sizes():
    assert_sizes(pd.DataFrame({"n": [50, 0, 1000, np.nan]}), ["n"])
    with pytest.raises(ValueError, match="column n"):
        assert_sizes(pd.DataFrame({"n": [50, 49]}), ["n"])


def test_pool_tail_keeps_every_class_at_50_or_more():
    # a long tail of small classes is pooled from the first value at which
    # the pooled class reaches 50
    counts = pd.Series([5000, 3000, 800, 400, 120, 60, 31, 10, 6, 2, 1], index=range(1, 12))
    pooled, k = pool_tail(counts)
    assert k == 7 and list(pooled.index) == ["1", "2", "3", "4", "5", "6", "7+"]
    assert pooled["7+"] == 50 and pooled.sum() == counts.sum()
    # a small class below a tail that is large enough joins the tail
    pooled, k = pool_tail(pd.Series([300, 30, 200, 60], index=[1, 2, 3, 4]))
    assert k == 2 and list(pooled.index) == ["1", "2+"] and pooled["2+"] == 290
    # nothing to pool
    pooled, k = pool_tail(pd.Series([300, 80, 60], index=[1, 2, 3]))
    assert k == 3 and list(pooled.index) == ["1", "2", "3"]

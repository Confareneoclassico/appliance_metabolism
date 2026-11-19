"""Stage 07: item non-response among the units an item applies to (D5), no
continuity correction (L2), sizes of 20 to 49 given as their class (L3), and
tests on fewer than 50 observations withheld."""
import importlib

import numpy as np
import pandas as pd

stage07 = importlib.import_module("07_statistical_validation")


def frame(cluster, values, country=0):
    return pd.DataFrame({"cluster": cluster, "IND6_1": values, "COUNTRY": country})


def test_nonresponse_among_the_units_the_item_applies_to():
    # 200 employed (answers 1/2), 300 without a job (-98), 50 refusals (-99)
    rng = np.random.default_rng(0)
    values = np.r_[rng.integers(1, 3, 200), np.full(300, -98), np.full(50, -99)]
    df = frame(rng.integers(0, 2, len(values)), values)
    res = stage07.validate_categorical_drivers(df, ["IND6_1"], "x.csv").iloc[0]
    assert res.Item_Nonresponse_Pct == 20.0             # 50 of 250, not 350 of 550
    assert res.N_Observations == 200
    assert res.Publication_Class == ""


def test_no_continuity_correction_on_2x2_tables():
    # 2 x 2 table [[20, 10], [10, 20]]: V = 1/3 without Yates' correction
    cluster = np.r_[np.zeros(30), np.ones(30)].astype(int)
    values = np.r_[np.full(20, 1), np.full(10, 2), np.full(10, 1), np.full(20, 2)]
    res = stage07.validate_categorical_drivers(frame(cluster, values), ["IND6_1"], "x.csv").iloc[0]
    assert abs(res.Cramers_V - 1 / 3) < 1e-12


def test_small_tests_are_withheld_and_sized_by_class():
    cluster = np.r_[np.zeros(15), np.ones(15)].astype(int)
    values = np.r_[np.full(10, 1), np.full(5, 2), np.full(5, 1), np.full(10, 2)]
    res = stage07.validate_categorical_drivers(frame(cluster, values), ["IND6_1"], "x.csv").iloc[0]
    assert res.N_Class == "20-49" and pd.isna(res.N_Observations)
    assert res.Publication_Class == "withheld"
    assert np.isnan(res.Chi2_Statistic) and np.isnan(res.Cramers_V)

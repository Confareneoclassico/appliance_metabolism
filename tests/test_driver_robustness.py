"""tools/driver_robustness.py on synthetic solutions whose answer is known:
the clusters follow the number of diarists of the household-day only, the
number of adults (HHC5) goes with the number of diarists, every household
contributes two days with the same cluster, and DDV6 is unrelated noise."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("driver_robustness", ROOT / "tools" / "driver_robustness.py")
dr = importlib.util.module_from_spec(spec)
sys.modules["driver_robustness"] = dr
spec.loader.exec_module(dr)


def synthetic(folder, rng, n_households=1500):
    nd = rng.choice([1, 2, 3, 4], size=n_households, p=[0.3, 0.4, 0.2, 0.1])
    hhc5 = np.where(rng.random(n_households) < np.select([nd == 1, nd == 2], [0.2, 0.8], 0.9), 2, 1)
    cl = np.where(rng.random(n_households) < 0.8, np.minimum(nd, 3) - 1, rng.integers(0, 3, n_households))
    wide, recs, sol = [], [], []
    for h in range(n_households):
        for d in (1, 2):
            e = np.zeros(144)
            e[(7 * h + d) % 144] = 1 + h % 50 + 100 * d
            wide.append({"HID": f"AA_{h}", "DIARY": d, "COUNTRY": "AA", "DDV1": 3 if d == 1 else 7,
                         "DDV7": 1 if d == 1 else 3, **dict(zip(dr.ENERGY, e))})
            for p in range(nd[h]):
                recs.append({"HID": f"AA_{h}", "PID": p + 1, "DIARY": d})
            hhc3 = -99 if rng.random() < 0.1 else int(rng.integers(0, 2))
            sol.append({**dict(zip(dr.ENERGY, e)), "COUNTRY": "AA", "HHC5": hhc5[h],
                        "DDV6": int(rng.integers(1, 3)), "HHC3": hhc3, "cluster": cl[h]})
    pd.DataFrame(wide).to_csv(folder / "household_interval_timeseries.csv", index=False)
    pd.DataFrame(recs).to_csv(folder / "person_daily_metabolic_totals.csv", index=False)
    s = pd.DataFrame(sol)
    s.to_csv(folder / "clustering_results.csv", index=False)
    (folder / "Country_Analysis").mkdir()
    s.to_csv(folder / "Country_Analysis" / "AA_clustering_results.csv", index=False)
    return s


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("processed")
    sol = synthetic(folder, np.random.default_rng(5))
    ids = dr.load_households(folder)
    files = dr.solution_files(folder, "countries")
    assert [f[0].name for f in files] == ["clustering_results.csv", "AA_clustering_results.csv"]
    sub = dr.match(sol, ids, None, None)
    res = dr.analyse(sol, sub.HID.to_numpy(), sub.n_diarists.to_numpy(), B=100, draws=20,
                     name="clustering_results.csv").set_index("Feature")
    return res, sol, ids, folder


def test_adults_vanish_at_a_fixed_number_of_diarists(run):
    res = run[0]
    assert res.loc["HHC5", "V_corrected"] > 0.3
    for label in ("1", "2", "3+"):
        assert res.loc[f"HHC5 | diarists {label}", "V_corrected"] < 0.06
    assert res.loc["HHC5 | diarists, pooled", "V_corrected"] < 0.05
    assert res.loc["N_DIARISTS", "V_corrected"] > res.loc["HHC5", "V_corrected"]
    assert res.loc["DDV6", "V_corrected"] < 0.05
    assert res.loc["HHC5", "rank_corrected"] == 1      # among the characteristics, diarists aside


def test_plain_v_as_stage07(run):
    res, sol = run[0], run[1]
    t = pd.crosstab(sol.cluster, sol.HHC5)
    chi2 = chi2_contingency(t, correction=False)[0]
    assert np.isclose(res.loc["HHC5", "V_plain"], np.sqrt(chi2 / (t.values.sum() * 1)))


def test_households_and_missing_answers(run):
    res = run[0]
    r = res.loc["HHC5"]
    assert r.V_corrected_lo <= r.V_corrected <= r.V_corrected_hi
    assert abs(r.V_one_day_per_household - r.V_corrected) < 0.05
    assert np.isfinite(res.loc["HHC3", "V_missing_as_category"])
    assert np.isclose(res.loc["HHC3", "Item_Nonresponse_Pct"], 10, atol=2)
    assert np.isnan(res.loc["HHC5"].get("V_missing_as_category", np.nan))


def test_order_is_checked(run):
    _, sol, ids, _ = run
    with pytest.raises(ValueError):
        dr.match(sol.sample(frac=1, random_state=1), ids, None, None)
    assert len(dr.match(sol, ids, "AA", None)) == len(sol)

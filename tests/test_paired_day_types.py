"""tools/paired_day_types.py on a synthetic Stage 03 output whose answer is
known: in country AA the same households raise their coincident and own peaks
by a fifth on non-working days, while households seen on one day type only
change the comparison of all household-days (composition); BB has too few
paired households to publish; CC has one diary day per household; DD has
several diary days of each type per household (as the Netherlands)."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("paired_day_types", ROOT / "tools" / "paired_day_types.py")
pdt = importlib.util.module_from_spec(spec)
sys.modules["paired_day_types"] = pdt
spec.loader.exec_module(pdt)

BASE = 10


def day(level, working):
    p = np.full(144, BASE, dtype=float)
    if working:
        p[84:96] += level                      # evening
    else:
        p[48:60] += round(1.2 * level)         # midday, a fifth higher
    return p


def synthetic(rng):
    rows, levels = [], {}
    def add(country, hid, diary, working, level):
        rows.append({"HID": f"{country}_{hid}", "DIARY": diary, "COUNTRY": country,
                     "DDV1": 3 if working else 7, "DDV7": 1 if working else 3,
                     **{f"Household_Energy_W_{t}": v for t, v in enumerate(day(level, working))}})
    for h in range(400):                       # AA: paired households
        a = int(rng.integers(100, 300))
        levels[("AA", h)] = a
        add("AA", h, 1, True, a)
        add("AA", h, 2, False, a)
    for h in range(400, 700):                  # AA: working days only, heavy users
        add("AA", h, 1, True, 600)
    for h in range(700, 1000):                 # AA: non-working days only, light users
        add("AA", h, 1, False, 50)
    for h in range(30):                        # BB: 30 paired households
        add("BB", h, 1, True, 150)
        add("BB", h, 2, False, 150)
    for h in range(200):                       # CC: one diary day per household
        add("CC", h, 1, h % 2 == 0, 150)
    for h in range(60):                        # DD: three working and two non-working days
        for d in range(3):
            add("DD", h, d + 1, True, 100 + 10 * d + h)
        for d in range(2):
            add("DD", h, d + 4, False, 100 + h)
    return pd.DataFrame(rows), levels


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    folder = tmp_path_factory.mktemp("processed")
    table, levels = synthetic(np.random.default_rng(3))
    table.to_csv(folder / "household_interval_timeseries.csv", index=False)
    ids, power = pdt.load(folder)
    return pdt.compare(ids, power, B=200).set_index("country"), levels, ids, power


def test_paired_changes_are_those_of_the_same_households(result):
    res, levels, _, _ = result
    a = np.array([v for (c, _), v in levels.items() if c == "AA"], dtype=float)
    w, nw = a.mean() + BASE, np.round(1.2 * a).mean() + BASE
    r = res.loc["AA"]
    assert np.isclose(r.paired_coincident_w, w) and np.isclose(r.paired_coincident_nw, nw)
    assert np.isclose(r.paired_coincident_change_pct, 100 * (nw / w - 1))
    assert np.isclose(r.paired_own_peak_change_pct, 100 * (nw / w - 1))   # one block a day
    assert r.paired_coincident_change_lo < r.paired_coincident_change_pct < r.paired_coincident_change_hi
    assert r.paired_coincident_time_w == "18:00" and r.paired_coincident_time_nw == "12:00"
    # the households seen on one day type only drag the comparison of all days down
    assert r.all_coincident_change_pct < r.paired_coincident_change_pct - 10
    assert r.n_paired_households == 400
    assert np.isclose(r.share_paired_w_pct, 100 * 400 / 700)
    assert np.isclose(r.share_paired_nw_pct, 100 * 400 / 700)
    assert r.own_peak_wilcoxon_p < 1e-6


def test_all_household_days_as_stage06(result):
    res, _, ids, power = result
    m = (ids.COUNTRY == "AA").to_numpy()
    w = power[m & (ids.day_type == 0).to_numpy()].astype(float)
    n = power[m & (ids.day_type == 1).to_numpy()].astype(float)
    assert np.isclose(res.loc["AA", "all_coincident_change_pct"],
                      100 * (n.mean(axis=0).max() / w.mean(axis=0).max() - 1))
    assert np.isclose(res.loc["AA", "all_own_peak_change_pct"],
                      100 * (n.max(axis=1).mean() / w.max(axis=1).mean() - 1))


def test_small_and_unpaired_countries_are_withheld(result):
    res = result[0]
    assert np.isnan(res.loc["BB", "n_paired_households"]) and res.loc["BB", "N_Class"] == "20-49"
    assert np.isnan(res.loc["BB"].get("paired_coincident_change_pct", np.nan))
    assert res.loc["CC", "N_Class"] == "none"
    assert "ALL" in res.index and res.loc["ALL", "n_paired_households"] == 460


def test_several_days_of_a_type_count_once(result):
    res = result[0]
    h = np.arange(60)
    w = (100 + h[:, None] + 10 * np.arange(3)[None, :]).mean(axis=1)   # mean working level
    nw = np.round(1.2 * (100 + h))
    assert np.isclose(res.loc["DD", "paired_own_peak_w"], w.mean() + BASE)
    assert np.isclose(res.loc["DD", "paired_own_peak_nw"], nw.mean() + BASE)


def test_weekday_rule(tmp_path):
    table, _ = synthetic(np.random.default_rng(3))
    table["DDV7"] = -9                         # no DDV7: the weekday decides
    table.to_csv(tmp_path / "household_interval_timeseries.csv", index=False)
    ids, power = pdt.load(tmp_path, "ddv7")
    ids1, _ = pdt.load(tmp_path, "ddv1")
    assert (ids.day_type == ids1.day_type).all()
    assert set(ids.day_type) == {0, 1}

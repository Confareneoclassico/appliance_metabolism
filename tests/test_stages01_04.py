"""Stages 01 to 04 run on a small synthetic raw file: country codes from the
file name (L1), a code recorded as main and secondary activity counted once
(L10), metabolic rates looked up by activity code (L11), not applicable kept
apart from non-response (D5), daily totals by household-day (D9) and the
household metabolic rate summed over persons (D10)."""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, run_stage
from synthetic import write_raw

N_HOUSEHOLDS = 80          # 80 household-days x 144 slots > 10,000 rows (D9)
MAPPING = pd.read_csv(ROOT / "data" / "HETUS_mapped_data.csv")
POWER = MAPPING.set_index("activity")["Appliance_Power_Watts"].to_dict()
MR_ADULT = MAPPING.set_index("activity")["MR_adult_Watts"].to_dict()


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    base = tmp_path_factory.mktemp("stages01_04")
    dirs = {"HETUS_DATA_DIR": base / "raw", "HETUS_PROCESSED_DIR": base / "processed",
            "HETUS_FIGURES_DIR": base / "figures"}
    for d in dirs.values():
        d.mkdir()
    write_raw(dirs["HETUS_DATA_DIR"], countries=("EE",), n_households=N_HOUSEHOLDS)
    for script in ("01_calculate_diarist_metrics.py", "02_calculate_dominant_activity.py",
                   "03_aggregate_household.py", "04_prepare_clustering_data.py"):
        done = run_stage(script, dirs, {"HETUS_COUNTRIES": "EE"})
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return dirs["HETUS_PROCESSED_DIR"]


def parse(s):
    """A series written by Stages 01 and 02 ('[1.0, nan, ...]')."""
    if not isinstance(s, str):
        return s
    return [np.nan if x.strip() in ("nan", "") else float(x) for x in s.strip("[]").split(",")]


def series(frame, col):
    return frame[col].apply(parse)


def test_country_code_from_the_file_name(processed):
    d = pd.read_csv(processed / "lu_data_at_home_with_diarist_metrics.csv")
    assert set(d["COUNTRY"]) == {2}        # EE is the third country of the list


def test_missing_file_stops_stage01(stage_dirs):
    done = run_stage("01_calculate_diarist_metrics.py", stage_dirs, {"HETUS_COUNTRIES": "BE,EE"})
    assert done.returncode != 0
    assert "BE.csv, EE.csv" in done.stderr


def test_main_and_secondary_code_counted_once(processed):
    d = pd.read_csv(processed / "lu_data_at_home_with_diarist_metrics.csv")
    first = d[(d.HID == "EE_1") & (d.PID == "EE_1")].iloc[0]
    energy = series(d.loc[[first.name]], "Diarist_Energy_Demand_Series").iloc[0]
    assert energy[0] == POWER[821]          # TV as main and secondary: once


def test_dominant_activity_by_metabolic_rate(processed):
    d = pd.read_csv(processed / "lu_data_at_home_with_dominant_activity.csv")
    dom = series(d, "Diarist_Dominant_Activity_Series")
    checked = 0
    for i, row in d.iterrows():
        for t in range(144):
            m, s, w = row[f"Mact{t + 1}"], row[f"Sactn{t + 1}"], row[f"Wherep{t + 1}"]
            if w == 1 and m == 11 and s == 821:
                assert dom.iloc[i][t] == 821   # TV (96 W) over sleep (74 W)
                checked += 1
    assert checked > 0


def test_not_applicable_kept_apart(processed):
    d1 = pd.read_csv(processed / "lu_data_at_home_with_diarist_metrics.csv")
    h2 = d1[d1.HID == "EE_2"].set_index("PID")
    assert h2.loc["EE_1", "IND6_1"] == -1 and np.isnan(h2.loc["EE_2", "IND6_1"])
    prep = pd.read_csv(processed / "prepared_clustering_data.csv")
    wide = pd.read_csv(processed / "household_interval_timeseries.csv", usecols=["HID", "DIARY"])
    codes = dict(zip(wide.HID, prep["IND6_1"]))
    assert codes["EE_2"] == -98             # first-listed diarist: not applicable
    assert codes["EE_3"] == -99             # first-listed diarist: refused


def test_daily_totals_by_household_day(processed):
    totals = pd.read_csv(processed / "household_daily_totals.csv")
    wide = pd.read_csv(processed / "household_interval_timeseries.csv")
    assert len(totals) == len(wide) == N_HOUSEHOLDS
    energy = wide[[f"Household_Energy_W_{t}" for t in range(144)]].sum(axis=1) / 6
    merged = totals.merge(wide[["HID", "DIARY"]].assign(e=energy), on=["HID", "DIARY"])
    assert np.allclose(merged.total_energy_Wh, merged.e)


def test_household_metabolic_rate_sums_persons(processed):
    wide = pd.read_csv(processed / "household_interval_timeseries.csv")
    h1 = wide[wide.HID == "EE_1"].iloc[0]
    # slot 1: both members asleep at home count twice, not once
    assert h1["Household_Metabolic_W_1"] == pytest.approx(2 * MR_ADULT[11])

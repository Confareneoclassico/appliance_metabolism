"""Age (INC2) in the 15 groups whichever release the raw files come from: the
release of 5 November 2026 gives completed years (10 for 10 to 17, 80 for 80
and over), the earlier one the groups (pipeline/age.py). Stages 01 to 04 give
the same files for both codings of the same ages."""
import numpy as np
import pandas as pd
import pytest

from age import (GROUP_LABELS, check_groups, coding_from_environment, detect_coding,
                 group_of_age, harmonise_inc2)
from conftest import ROOT, run_stage
from synthetic import write_raw

MAPPING = pd.read_csv(ROOT / "data" / "HETUS_mapped_data.csv").set_index("activity")


def bounds(label):
    """Ages covered by a label of the variable list ('20-24', '80+')."""
    if label.endswith("+"):
        return int(label[:-1]), 85
    lo, hi = label.split("-")
    return int(lo), int(hi)


def test_labels_are_those_of_the_variable_list():
    assert list(GROUP_LABELS) == list(range(1, 16))
    assert GROUP_LABELS[1] == "10-17" and GROUP_LABELS[2] == "18-19"
    assert GROUP_LABELS[3] == "20-24" and GROUP_LABELS[14] == "75-79" and GROUP_LABELS[15] == "80+"


@pytest.mark.parametrize("age", range(10, 86))
def test_every_completed_year_falls_in_its_group(age):
    group = int(group_of_age([age])[0])
    lo, hi = bounds(GROUP_LABELS[group])
    assert lo <= age <= hi


def test_codes_of_the_release_of_november_2026():
    years = pd.Series([10, 18, 19, 20, 24, 25, 64, 65, 79, 80, -7, -9])
    out, coding = harmonise_inc2(years)
    assert coding == "years"
    assert out.tolist() == [1, 2, 2, 3, 3, 4, 11, 12, 14, 15, -7, -9]
    assert out.dtype == np.int64


def test_empty_cells_stay_empty():
    out, coding = harmonise_inc2(pd.Series([45.0, np.nan, -6.0, 10.0]))
    assert coding == "years"
    assert out.iloc[0] == 8 and np.isnan(out.iloc[1]) and out.iloc[2] == -6 and out.iloc[3] == 1


def test_groups_are_left_as_they_are():
    groups = pd.Series([1, 2, 5, 15, -7], index=[10, 11, 12, 13, 14])
    out, coding = harmonise_inc2(groups)
    assert coding == "groups"
    pd.testing.assert_series_equal(out, groups)


def test_coding_is_read_from_the_values():
    assert detect_coding([1, 2, 15, -9]) == "groups"
    assert detect_coding([10, 18, 80, -9]) == "years"
    assert detect_coding([-9, np.nan]) == "groups"          # nothing to map


@pytest.mark.parametrize("values, coding", [
    ([3, 45], None),             # groups mixed with years
    ([0, 45], None),             # 0 is neither a group nor an age in the files
    ([45, 86], None),            # above the top-coded ages
    ([2.5, 45], None),           # not whole
    ([1, 45], "groups"),         # years read as groups
    ([5, 45], "years"),          # a group code read as an age
])
def test_values_that_fit_neither_coding_stop_the_run(values, coding):
    with pytest.raises(ValueError, match="HETUS_INC2_CODING"):
        harmonise_inc2(pd.Series(values), coding)


def test_coding_set_in_the_environment(monkeypatch):
    monkeypatch.delenv("HETUS_INC2_CODING", raising=False)
    assert coding_from_environment() is None
    monkeypatch.setenv("HETUS_INC2_CODING", " Years ")
    assert coding_from_environment() == "years"
    monkeypatch.setenv("HETUS_INC2_CODING", "auto")
    assert coding_from_environment() is None
    monkeypatch.setenv("HETUS_INC2_CODING", "decades")
    with pytest.raises(ValueError):
        coding_from_environment()


def test_check_of_processed_files():
    check_groups(pd.Series([1, 15, np.nan, -7]), "test")
    with pytest.raises(ValueError, match="rerun Stage 01"):
        check_groups(pd.Series([1, 45]), "test")


# ---------------------------------------------------------------------------
# Stages 01 to 04 on the same synthetic ages in both codings

def ages(n, rng):
    """Completed years covering every group, with 10 for 10 to 17 and 80 for
    80 and over as in the release of 5 November 2026, and missing codes."""
    a = np.concatenate([[10, 18, 19], 20 + 5 * np.arange(12) + rng.integers(0, 5, 12), [80]])
    a = np.resize(rng.permutation(np.tile(a, n // len(a) + 1)), n)
    a[rng.random(n) < 0.05] = -7
    return a


@pytest.fixture(scope="module")
def both_codings(tmp_path_factory):
    rng = np.random.default_rng(11)
    out = {}
    years = None
    for coding in ("groups", "years"):
        base = tmp_path_factory.mktemp(f"age_{coding}")
        dirs = {"HETUS_DATA_DIR": base / "raw", "HETUS_PROCESSED_DIR": base / "processed",
                "HETUS_FIGURES_DIR": base / "figures"}
        for d in dirs.values():
            d.mkdir()
        (path,) = write_raw(dirs["HETUS_DATA_DIR"], countries=("EE",), n_households=60)
        raw = pd.read_csv(path)
        if years is None:
            years = ages(len(raw), rng)
        raw["INC2"] = years if coding == "years" else np.where(years > 0, group_of_age(np.abs(years)), years)
        raw.to_csv(path, index=False)
        logs = []
        for script in ("01_calculate_diarist_metrics.py", "02_calculate_dominant_activity.py",
                       "03_aggregate_household.py", "04_prepare_clustering_data.py"):
            done = run_stage(script, dirs, {"HETUS_COUNTRIES": "EE"})
            assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
            logs.append(done.stdout)
        out[coding] = (dirs["HETUS_PROCESSED_DIR"], "".join(logs))
    return out


def test_stage01_names_the_coding(both_codings):
    assert "[age] EE: INC2 in the 15 age groups (release of November 2020)" in both_codings["groups"][1]
    assert ("[age] EE: INC2 in completed years (release of 5 November 2026), mapped to the 15 age "
            "groups") in both_codings["years"][1]


@pytest.mark.parametrize("name", ["lu_data_at_home_with_diarist_metrics.csv",
                                  "lu_data_at_home_with_dominant_activity.csv",
                                  "household_interval_timeseries.csv",
                                  "person_daily_metabolic_totals.csv",
                                  "household_daily_totals.csv",
                                  "prepared_clustering_data.csv"])
def test_both_codings_give_the_same_files(both_codings, name):
    groups = (both_codings["groups"][0] / name).read_bytes()
    years = (both_codings["years"][0] / name).read_bytes()
    assert groups == years


def test_metabolic_rate_follows_the_age_group(both_codings):
    d = pd.read_csv(both_codings["years"][0] / "lu_data_at_home_with_diarist_metrics.csv")
    assert set(d["INC2"].dropna()) <= set(range(1, 16)) and d["INC2"].isna().any()
    rate = {1: "MR_teenager_Watts", 5: "MR_adult_Watts", 12: "MR_senior_Watts"}
    seen = set()
    for _, row in d[d["INC2"].isin(list(rate))].iterrows():
        series = [float(x) for x in row["Diarist_Metabolic_Rate_Series"].strip("[]").split(",")]
        for t in range(144):
            if row[f"Wherep{t + 1}"] == 1 and row[f"Mact{t + 1}"] == 11 and row[f"Sactn{t + 1}"] not in (821, 831):
                assert series[t] == pytest.approx(MAPPING.loc[11, rate[int(row["INC2"])]])
                seen.add(int(row["INC2"]))
    assert seen == set(rate)

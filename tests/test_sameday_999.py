"""Stage 01 after the check of the codes against the documentation (November
2026): household-days whose members kept their diaries on different days are
left out (same-day rule), and secondary code 999 is no secondary activity."""
import numpy as np
import pandas as pd
import pytest

from conftest import run_stage
from synthetic import BASE_COLUMNS, SERIES, SLOTS, person_day

DAY = {"DDV1": 2, "DDV3": 5, "DDV4": 2010}      # a Monday in May 2010


def raw_file(folder):
    """Households of two diarists (diary 1 unless stated):
    1 same day; 2 other weekday; 3 same weekday, other month; 4 same weekday
    and month, other year; 5 one diarist; 6 same day on diary 1, other
    weekdays on diary 2; 7 same day for the two members at home, a third
    member never at home on another weekday. Person 1 of household 1 has
    secondary code 999 in the first ten slots, all at home."""
    rng = np.random.default_rng(3)
    rows = []

    def add(hid, pid, diary=1, **day):
        rows.append(person_day(hid, pid, diary, rng, "EE", **{**DAY, **day}))

    add(1, 1); add(1, 2)
    add(2, 1); add(2, 2, DDV1=7)
    add(3, 1); add(3, 2, DDV3=6)
    add(4, 1); add(4, 2, DDV4=2011)
    add(5, 1)
    add(6, 1); add(6, 2); add(6, 1, 2, DDV1=7); add(6, 2, 2, DDV1=1)
    add(7, 1); add(7, 2); add(7, 3, DDV1=4)
    df = pd.DataFrame(rows, columns=BASE_COLUMNS + ["DDV4"]
                      + [f"{s}{t}" for s in SERIES for t in range(1, SLOTS + 1)])
    away = (df.HID == 7) & (df.PID == 3)
    df.loc[away, [f"Wherep{t}" for t in range(1, SLOTS + 1)]] = 2
    first = (df.HID == 1) & (df.PID == 1)
    for t in range(1, 11):
        df.loc[first, [f"Mact{t}", f"Sactn{t}", f"Wherep{t}"]] = [11, 999, 1]
    df.to_csv(folder / "EE.csv", index=False)


def run(stage_dirs, rule="1"):
    raw_file(stage_dirs["HETUS_DATA_DIR"])
    done = run_stage("01_calculate_diarist_metrics.py", stage_dirs,
                     {"HETUS_COUNTRIES": "EE", "HETUS_SAME_DAY_RULE": rule})
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    out = pd.read_csv(stage_dirs["HETUS_PROCESSED_DIR"] / "lu_data_at_home_with_diarist_metrics.csv")
    return out, done.stdout


def test_household_days_on_different_days_are_left_out(stage_dirs):
    out, log = run(stage_dirs)
    kept = set(zip(out.HID, out.DIARY))
    assert kept == {("EE_1", 1), ("EE_5", 1), ("EE_6", 1), ("EE_7", 1)}
    assert len(out[(out.HID == "EE_7")]) == 2          # the member never at home is not a record
    # counts of the multi-diarist household-days, in classes below 50
    assert "[same-day rule] EE: <20 multi-diarist, <20 other weekday, <20 same weekday" in log


def test_rule_can_be_switched_off(stage_dirs):
    out, log = run(stage_dirs, rule="0")
    kept = set(zip(out.HID, out.DIARY))
    assert {("EE_2", 1), ("EE_3", 1), ("EE_4", 1), ("EE_6", 2)} <= kept
    assert "kept: HETUS_SAME_DAY_RULE=0" in log


def test_secondary_999_is_no_secondary_activity(stage_dirs):
    out, log = run(stage_dirs)
    first = out[(out.HID == "EE_1") & (out.PID == "EE_1")].iloc[0]
    assert all(np.isnan(first[f"Sactn{t}"]) for t in range(1, 11))
    assert "[secondary 999] EE:" in log
    share = float(log.split("[secondary 999] EE:")[1].split("%")[0])
    assert share > 0

"""Small synthetic inputs in the format of the HETUS 2010 scientific use files.

No microdata are used: every value is drawn at random or set by hand, so the
tests can run anywhere. The activity codes are those of the mapping table in
data/, so the appliance powers and metabolic rates are the real ones.
"""
import numpy as np
import pandas as pd

SLOTS = 144
BASE_COLUMNS = ["YEAR", "HID", "PID", "DIARY", "COUNTRY", "DDV1", "DDV3", "DDV6", "DDV7",
                "INC1", "INC2", "INC3", "INC4_1", "IND6_1", "HHQ5", "HHQ9_1", "IND22_1",
                "IND44", "HHC1", "HHC3", "HHC4", "HHC5"]
SERIES = ["Mact", "Sactn", "Wherep", "Wpartner", "Wchild", "Wotherh"]
CODES = [11, 21, 311, 321, 331, 721, 821, 831, 910]


def person_day(hid, pid, diary, rng, country="BE", messy=False, **attrs):
    """One diarist record with random codes, at home except for a block of
    travel and work in the middle of the day. With `messy`, secondary codes
    are also empty, negative or absent from the mapping, some locations are
    empty, and the members are at home at random times."""
    row = {"YEAR": 2010, "HID": hid, "PID": pid, "DIARY": diary, "COUNTRY": country,
           "DDV1": 3, "DDV3": 5, "DDV6": 1, "DDV7": 1, "INC1": 1 + pid % 2, "INC2": 5,
           "INC3": 1, "INC4_1": 1, "IND6_1": 2, "HHQ5": 1, "HHQ9_1": 3, "IND22_1": 2,
           "IND44": 1, "HHC1": 2, "HHC3": 0, "HHC4": 0, "HHC5": 2}
    row.update(attrs)
    away = rng.integers(30, 60)
    for t in range(SLOTS):
        home = not (away <= t < away + 40)
        if messy:
            home = rng.random() < 0.8
        row[f"Mact{t + 1}"] = int(rng.choice(CODES)) if home else 910
        if messy:
            row[f"Sactn{t + 1}"] = rng.choice([np.nan, np.nan, 0, -1, 821, 831, 7777])
            row[f"Wherep{t + 1}"] = 1 if home else rng.choice([2, np.nan])
        else:
            row[f"Sactn{t + 1}"] = int(rng.choice([0, 821, 831])) if home else 0
            row[f"Wherep{t + 1}"] = 1 if home else 2
        for s in ("Wpartner", "Wchild", "Wotherh"):
            row[f"{s}{t + 1}"] = 0
    return row


def raw_country(n_households, rng, country="BE", sizes=(2,), messy=False):
    """One diary day per household, with sizes[h % len(sizes)] diarists in
    household h (two by default), and a few records built by hand at the
    start:

    household 1: person 1 watches TV as both main and secondary activity in
        slot 0 (L10), and both persons sleep at home in slot 1 (D10);
    household 2: person 1 has the job items not applicable (-1) and person 2
        refused them (-7) (D5); person 1 is listed first;
    household 3: person 1 refused the job items (-7).
    """
    rows = []
    for h in range(1, n_households + 1):
        for p in range(1, max(2, sizes[h % len(sizes)]) + 1):
            rows.append(person_day(h, p, 1, rng, country, messy=messy))
    df = pd.DataFrame(rows, columns=BASE_COLUMNS + [f"{s}{t}" for s in SERIES
                                                      for t in range(1, SLOTS + 1)])
    # household 1
    first = (df.HID == 1)
    df.loc[first & (df.PID == 1), ["Mact1", "Sactn1", "Wherep1"]] = [821, 821, 1]
    df.loc[first & (df.PID == 2), ["Mact1", "Sactn1", "Wherep1"]] = [11, 0, 1]
    df.loc[first, ["Mact2", "Sactn2", "Wherep2"]] = [11, 0, 1]
    # household 2
    df.loc[(df.HID == 2) & (df.PID == 1), ["IND6_1", "IND44"]] = [-1, -1]
    df.loc[(df.HID == 2) & (df.PID == 2), ["IND6_1", "IND44"]] = [-7, -7]
    # household 3
    df.loc[(df.HID == 3) & (df.PID == 1), ["IND6_1", "IND44"]] = [-7, -7]
    return df


def write_raw(folder, countries=("BE",), n_households=40, seed=1, sizes=(2,), messy=False):
    """Raw files for `countries` in `folder`, and return their paths."""
    rng = np.random.default_rng(seed)
    paths = []
    for c in countries:
        path = folder / f"{c}.csv"
        raw_country(n_households, rng, c, sizes=sizes, messy=messy).to_csv(path, index=False)
        paths.append(path)
    return paths


def prepared_data(sizes, seed=2, whole_watts=True):
    """A prepared_clustering_data.csv table (Stage 04 format) with the given
    number of household-days per (country code, DDV1) cell, for instance
    {(0, 3): 120, (0, 1): 30} for 120 Belgian weekdays and 30 Sundays."""
    rng = np.random.default_rng(seed)
    blocks = []
    for (country, ddv1), n in sizes.items():
        e = rng.gamma(2.0, 150.0, size=(n, SLOTS))
        if whole_watts:
            e = np.round(e)
        b = pd.DataFrame(e, columns=[f"Household_Energy_W_{t}" for t in range(SLOTS)])
        b["COUNTRY"] = country
        b["DDV1"] = ddv1
        b["DDV7"] = -99
        for col in ["DDV3", "DDV6", "INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
                    "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"]:
            b[col] = rng.integers(1, 4, size=n)
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)

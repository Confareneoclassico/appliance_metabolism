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


# ---------------------------------------------------------------------------
# Inputs for the sensitivity analysis (gsa.bundle, gsa.model): households of
# one to three diarists with a varied day, two day types, weights, diary
# months and years, and the ownership items.
GSA_SIZES = {"BE": (1, 2, 3), "FR": (2, 1, 3, 2), "NL": (1,)}


def _block(codes, start, length, code):
    codes[max(0, start):max(0, min(len(codes), start + length))] = code


def gsa_person(rng, hid, pid, house, adult=True):
    """One diary day of one diarist of a household described by `house`."""
    employed = adult and rng.random() < 0.6
    row = {"YEAR": house["year"], "HID": hid, "PID": pid, "DIARY": 1, "COUNTRY": house["country"],
           "DDV1": house["ddv1"], "DDV3": house["month"], "DDV4": house["year"],
           "DDV6": 1 if employed else 2,
           "DDV7": (house["ddv7_work"] if employed else -9),
           "INC1": 1 + int(rng.random() < 0.5), "INC2": int(rng.integers(4, 9)) if adult else 2,
           "INC3": 1, "INC4_1": 1 if employed else int(rng.integers(2, 6)),
           "IND6_1": int(rng.integers(1, 4)) if employed else -1,
           "IND44": int(rng.integers(1, 3)) if employed else -1,
           "IND22_1": int(rng.integers(1, 4)), "HHQ5": house["hhq5"], "HHQ9_1": house["income"],
           "HHC1": house["size"], "HHC3": house["hhc3"], "HHC4": house["hhc4"],
           "HHC5": house["hhc5"], "WGHT1": float(rng.lognormal(0.0, 0.3)),
           "HHQ6c": house["hhq6c"], "HHQ6h": house["hhq6h"], "HHQ6o": house["hhq6o"]}
    m = np.full(SLOTS, 11)                         # sleep
    s = np.zeros(SLOTS)
    where = np.ones(SLOTS)
    wake, bed = int(rng.integers(10, 30)), int(rng.integers(108, 132))
    day = rng.choice([21, 31, 39, 329, 351, 363, 513, 529, 619, 719, 732, 999], size=bed - wake)
    m[wake:bed] = day
    working_day = house["ddv1"] in (2, 3, 4, 5, 6) and employed and house["ddv7_work"] != 3
    if working_day:                                 # away at work
        out = wake + int(rng.integers(6, 12))
        span = int(rng.integers(40, 60))
        _block(m, out, span, 910)
        where[out:out + span] = 3
    if rng.random() < house["tv"]:
        _block(m, int(rng.integers(84, 100)), int(rng.integers(8, 24)), 821)
    if rng.random() < 0.7:
        _block(m, int(rng.integers(46, 54)), int(rng.integers(3, 7)), 311)
        _block(m, int(rng.integers(86, 94)), int(rng.integers(3, 7)), 311)
    if rng.random() < 0.3:
        _block(m, int(rng.integers(wake, max(wake + 1, bed - 6))), int(rng.integers(2, 6)), 331)
    for code, prob in ((321, 0.3), (332, 0.15), (721, 0.4), (733, 0.15), (371, 0.2), (31, 0.2)):
        if rng.random() < prob:
            _block(m, int(rng.integers(wake, max(wake + 1, bed - 6))), int(rng.integers(2, 9)), code)
    m[where != 1] = np.where(m[where != 1] == 11, 910, m[where != 1])
    sec_pick = rng.random(SLOTS)
    s[(sec_pick < 0.08) & (m != 821)] = 821
    s[(sec_pick > 0.92)] = 831
    s[(sec_pick > 0.5) & (sec_pick < 0.52)] = 331
    s[(sec_pick > 0.6) & (sec_pick < 0.62)] = 999
    s[where != 1] = 0
    for t in range(SLOTS):
        row[f"Mact{t + 1}"] = int(m[t])
        row[f"Sactn{t + 1}"] = int(s[t])
        row[f"Wherep{t + 1}"] = int(where[t])
        for prefix in ("Wpartner", "Wchild", "Wotherh"):
            row[f"{prefix}{t + 1}"] = 0
    return row


def raw_country_gsa(n_households, rng, country):
    sizes = GSA_SIZES.get(country, (2,))
    rows = []
    for h in range(1, n_households + 1):
        size = sizes[h % len(sizes)]
        weekend = h % 2 == 0
        house = {"country": country, "size": size, "year": 2010 + int(rng.random() < 0.4),
                 "month": int(rng.integers(1, 13)),
                 "ddv1": int(rng.choice([1, 7])) if weekend else int(rng.integers(2, 7)),
                 "ddv7_work": (3 if weekend else (3 if rng.random() < 0.15 else 1)),
                 "hhq5": int(rng.integers(1, 3)),
                 "income": int(rng.integers(1, 11)) if rng.random() < 0.85 else -7,
                 "hhc3": int(rng.integers(0, 2)), "hhc4": int(rng.integers(0, 3)),
                 "hhc5": 1 if size == 1 else 2,
                 "hhq6c": (-9 if country == "NL" else int(rng.random() < 0.9)),
                 "hhq6h": 2 if rng.random() < 0.15 else 1,
                 "hhq6o": 2 if rng.random() < 0.3 else 1,
                 "tv": 0.4 + 0.15 * size}
        for p in range(1, size + 1):
            rows.append(gsa_person(rng, h, p, house, adult=(p <= 2)))
    return pd.DataFrame(rows)


def write_raw_gsa(folder, countries=("BE", "FR", "NL"), n_households=130, seed=7):
    """Raw files for the tests of the sensitivity analysis, with the extra
    columns that gsa.bundle reads (WGHT1, DDV4, HHQ6c, HHQ6h, HHQ6o)."""
    rng = np.random.default_rng(seed)
    for c in countries:
        raw_country_gsa(n_households, rng, c).to_csv(folder / f"{c}.csv", index=False)


def raw_country_two_days(n_households, rng, country, share_two=0.8):
    """Households of one to three diarists who keep a diary on a weekday
    (DIARY 1) and, for `share_two` of the households, also on a weekend day
    (DIARY 2), as in most HETUS countries; the employed work on the weekday
    only, so the two day types differ within the same households."""
    sizes = GSA_SIZES.get(country, (2,))
    rows = []
    for h in range(1, n_households + 1):
        size = sizes[h % len(sizes)]
        weekday = {"country": country, "size": size, "year": 2010, "month": int(rng.integers(1, 13)),
                   "ddv1": int(rng.integers(2, 7)), "ddv7_work": 1, "hhq5": int(rng.integers(1, 3)),
                   "income": int(rng.integers(1, 11)), "hhc3": int(rng.integers(0, 2)),
                   "hhc4": int(rng.integers(0, 3)), "hhc5": 1 if size == 1 else 2,
                   "hhq6c": int(rng.random() < 0.9), "hhq6h": 1, "hhq6o": 1, "tv": 0.4 + 0.15 * size}
        days = [(1, weekday)]
        if rng.random() < share_two:
            days.append((2, {**weekday, "ddv1": int(rng.choice([1, 7])), "ddv7_work": 3,
                             "tv": min(1.0, weekday["tv"] + 0.3)}))
        for diary, house in days:
            for p in range(1, size + 1):
                row = gsa_person(rng, h, p, house, adult=(p <= 2))
                row["DIARY"] = diary
                rows.append(row)
    return pd.DataFrame(rows)


def write_raw_two_days(folder, countries=("BE", "FR", "NL"), n_households=120, seed=11):
    """Raw files with two diary days for most households (tools/ checks)."""
    rng = np.random.default_rng(seed)
    for c in countries:
        raw_country_two_days(n_households, rng, c).to_csv(folder / f"{c}.csv", index=False)

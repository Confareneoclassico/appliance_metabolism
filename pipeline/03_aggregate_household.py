"""Stage 03: household power and metabolic rate per household-day and slot.

The diarist records of a household with the same diary number (HID, DIARY)
form a household-day. In every 10-minute slot its power is the sum of the
appliance powers of the distinct activity codes, main or secondary, of the
diarists at home (Wherep = 1): an activity shared by several members counts
once (the shared-appliance rule). This is a sparse product p = A w, where A
has one row per household-day and slot and one column per activity code, with
a 1 where the code is in use at home, and w holds the appliance powers of the
mapping table, so that other powers (the sensitivity analysis) need only
another w. The metabolic rate is the sum of the rates of the diarists at home,
from the series of Stage 01 (audit of November 2026, D10), and the
person-level attributes are those of the first-listed diarist.

Versions up to 1.1.0 computed the same quantities with a loop over
household-days and slots, which took 10 to 15 hours on HETUS 2010. The loop is
kept in aggregate_household_loop_reference.py, and tools/compare_stage03.py
compares the two. Power is a sum of whole watts and is reproduced exactly; the
metabolic rate is summed in the order of the loop (numpy's for up to seven
diarists at home, numpy's own for more), so it is reproduced to the last bit
as well.

Outputs (working files, not to be shared):
    household_interval_timeseries.csv   one row per household-day: the
                                        attributes and the 144 slots of power
                                        and of metabolic rate (Stage 04 reads it)
    person_daily_metabolic_totals.csv   one row per diarist record (Stage 10)
    household_daily_totals.csv          daily energy and metabolic totals and
                                        the mean numbers of diarists and of
                                        distinct dominant activities at home
    household_interval_timeseries_long.csv
                                        one row per household-day and slot,
                                        only with HETUS_WRITE_LONG=1 (no stage
                                        reads it; versions up to 1.1.0 always
                                        wrote it, about 28 million rows)
    activity_matrix.npz                 the matrices of the sensitivity
                                        analysis, only with
                                        HETUS_SAVE_ACTIVITY_MATRIX=1: codes in
                                        use at home (union), used as a main
                                        activity, and the number of diarists
                                        using them, with the activity codes and
                                        the household-day of every row block
"""
import os
import time

import numpy as np
import pandas as pd

from paths import PROCESSED_DIR, MAPPING_FILE

SLOTS = 144
INTERVAL_HOURS = 10.0 / 60.0
EXTRA_COLS = ["DDV1", "DDV3", "DDV6", "DDV7", "COUNTRY", "HHC1", "HHC3", "HHC4", "HHC5",
              "INC1", "INC2", "INC3", "INC4_1", "IND6_1", "HHQ5", "HHQ9_1", "IND22_1", "IND44"]
MACT = [f"Mact{t}" for t in range(1, SLOTS + 1)]
SACT = [f"Sactn{t}" for t in range(1, SLOTS + 1)]
WHERE = [f"Wherep{t}" for t in range(1, SLOTS + 1)]
MR_SERIES = "Diarist_Metabolic_Rate_Series"
DOM_SERIES = "Diarist_Dominant_Activity_Series"
NO_CODE = np.int32(-2 ** 31)        # an empty activity cell
BLOCK = int(os.environ.get("HETUS_STAGE03_BLOCK", "20000"))   # household-days per block
WRITE_LONG = os.environ.get("HETUS_WRITE_LONG", "0") == "1"
SAVE_MATRIX = os.environ.get("HETUS_SAVE_ACTIVITY_MATRIX", "0") == "1"
INPUT = PROCESSED_DIR / "lu_data_at_home_with_dominant_activity.csv"


def parse_series(column):
    """The 144 values of each list written by Stages 01 and 02 ('[74.3, 0,
    nan, ...]'), as the loop of versions up to 1.1.0 parsed them: 'nan',
    'np.nan', empty and unreadable elements are NaN, a missing list is 144
    NaN. Returns the values and whether each list had exactly 144 elements."""
    text = column.astype("string")
    parts = text.str.strip("[]").str.split(",", expand=True)
    n = len(column)
    values = np.full((n, SLOTS), np.nan)
    if parts.shape[1]:
        width = min(parts.shape[1], SLOTS)
        block = parts.iloc[:, :width].apply(lambda c: pd.to_numeric(c.str.strip(), errors="coerce"))
        values[:, :width] = block.to_numpy(dtype="float64")
    lengths = parts.notna().sum(axis=1).to_numpy() if parts.shape[1] else np.zeros(n, int)
    missing = text.isna().to_numpy()
    return values, missing | (lengths == SLOTS)


def as_codes(frame):
    """Activity codes as int32, truncated as int() does, NO_CODE where empty."""
    a = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    out = np.full(a.shape, NO_CODE, dtype=np.int32)
    ok = ~np.isnan(a)
    out[ok] = np.trunc(a[ok]).astype(np.int32)
    return out


def read_records(path=INPUT):
    """Identifiers, attributes, codes, location and series of every diarist
    record, in file order."""
    header = pd.read_csv(path, nrows=0).columns
    extra = [c for c in EXTRA_COLS if c in header]
    cols = ["HID", "PID", "DIARY"] + extra + MACT + SACT + WHERE + [MR_SERIES, DOM_SERIES]
    meta, M, S, W, MR, MR_ok, DOM = [], [], [], [], [], [], []
    for ch in pd.read_csv(path, usecols=[c for c in cols if c in header], chunksize=20000,
                          low_memory=False):
        meta.append(ch[["HID", "PID", "DIARY"] + extra])
        M.append(as_codes(ch[MACT]))
        S.append(as_codes(ch[SACT]))
        W.append(ch[WHERE].apply(pd.to_numeric, errors="coerce").to_numpy() == 1)
        mr, ok = parse_series(ch[MR_SERIES])
        MR.append(mr)
        MR_ok.append(ok)
        dom, _ = parse_series(ch[DOM_SERIES])
        # the loop counted distinct dominant activities with NaN as -1 and
        # the codes truncated to integers
        DOM.append(np.where(np.isnan(dom), -1, np.trunc(np.nan_to_num(dom, nan=-1))).astype(np.int32))
    meta = pd.concat(meta, ignore_index=True)
    if "COUNTRY" not in meta.columns:
        meta["COUNTRY"] = meta["HID"].map(
            lambda x: x.split("_")[0] if isinstance(x, str) and "_" in x else "Unknown")
    return (meta, np.vstack(M), np.vstack(S), np.vstack(W), np.vstack(MR),
            np.concatenate(MR_ok), np.vstack(DOM))


def group_records(meta):
    """Household-day of every record, numbered in order of first appearance
    (the order of the loop and of the output files); -1 for records without
    HID or DIARY, which the loop never matched."""
    ng = meta.groupby(["HID", "DIARY"], sort=False, dropna=True).ngroup()
    return ng.fillna(-1).astype(np.int64).to_numpy()


def metabolic_sum(key, values, n_keys):
    """Sum of values by key as pandas summed the metabolic rates of the
    diarists at home in the loop: in record order, sequentially (numpy's sum
    for fewer than eight terms), and with numpy's own summation for eight or
    more terms. Empty sums are 0; NaN terms count as 0, as in pandas."""
    terms = np.nan_to_num(values, nan=0.0)
    out = np.bincount(key, weights=terms, minlength=n_keys)
    count = np.bincount(key, minlength=n_keys)
    for k in np.flatnonzero(count >= 8):
        out[k] = np.sum(terms[key == k])
    return out


def household_block(gl, M, S, W, MR, DOM, power_of, codes):
    """Slot quantities of a block of household-days. gl: block-local
    household-day of every record (records sorted by it, file order kept).
    Returns arrays of shape (n, 144) and the sparse entries of A."""
    n = int(gl.max()) + 1
    r, t = np.nonzero(W)                         # records at home, row-major
    key = gl[r].astype(np.int64) * SLOTS + t
    nk = n * SLOTS
    n_home = np.bincount(key, minlength=nk)
    metab = metabolic_sum(key, MR[r, t], nk)
    # distinct dominant activities of the diarists at home
    dom = DOM[r, t].astype(np.int64)
    dk = np.unique(key * (2 ** 31) + (dom + 2 ** 30))
    n_dom = np.bincount(dk // (2 ** 31), minlength=nk)
    # distinct codes, main or secondary, of the diarists at home
    m, s = M[r, t], S[r, t]
    has_m, has_s = m != NO_CODE, s != NO_CODE
    ent_key = np.r_[key[has_m], key[has_s]]
    ent_code = np.searchsorted(codes, np.r_[m[has_m], s[has_s]])
    ent_main = np.r_[np.ones(has_m.sum(), bool), np.zeros(has_s.sum(), bool)]
    ent_rec = np.r_[r[has_m], r[has_s]]
    nc = len(codes)
    union = np.unique(ent_key * nc + ent_code)
    u_key, u_code = union // nc, union % nc
    n_codes = np.bincount(u_key, minlength=nk)
    power = np.bincount(u_key, weights=power_of[u_code], minlength=nk)
    entries = None
    if SAVE_MATRIX:
        main = np.unique(ent_key[ent_main] * nc + ent_code[ent_main])
        # diarists using a code, each counted once per code
        per_rec = np.unique((ent_rec.astype(np.int64) * SLOTS + np.r_[t[has_m], t[has_s]]) * nc
                            + ent_code)
        rec, rest = per_rec // (SLOTS * nc), per_rec % (SLOTS * nc)
        uk, cnt = np.unique(gl[rec].astype(np.int64) * SLOTS * nc + rest, return_counts=True)
        entries = (union, main, uk, cnt)
    shape = (n, SLOTS)
    return (power.reshape(shape), metab.reshape(shape), n_home.reshape(shape),
            n_dom.reshape(shape), n_codes.reshape(shape), entries)


def person_totals(MR, MR_ok, W):
    """Daily metabolic energy of each record at home, in Wh, accumulated slot
    by slot as in the loop (0 where its list did not have 144 values)."""
    contrib = np.where(W, MR * INTERVAL_HOURS, 0.0)
    total = np.cumsum(contrib, axis=1)[:, -1]
    return np.where(MR_ok, total, 0.0)


def write(frame, path, first):
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)


def main():
    t0 = time.time()
    print("Starting aggregate_household_metrics.py (sparse products)")
    hetus = pd.read_csv(MAPPING_FILE)
    activity_power = hetus.set_index("activity")["Appliance_Power_Watts"].to_dict()

    print("Loading data...")
    meta, M, S, W, MR, MR_ok, DOM = read_records()
    print(f"Loaded {len(meta)} rows")
    print("\nCountry distribution in raw data:")
    print(meta["COUNTRY"].value_counts())

    g = group_records(meta)
    keep = np.flatnonzero(g >= 0)
    order = keep[np.argsort(g[keep], kind="stable")]       # by household-day, file order kept
    meta, M, S, W, MR, MR_ok, DOM = (meta.iloc[order].reset_index(drop=True), M[order], S[order],
                                     W[order], MR[order], MR_ok[order], DOM[order])
    g = g[order]
    n_days = int(g.max()) + 1 if len(g) else 0
    print(f"Found {n_days} unique household-day combinations")

    # Secondary code 999 is set to no secondary activity in Stage 01, which
    # prints its share by country (89.9% of the Dutch at-home slots)

    codes = np.unique(np.r_[M[W & (M != NO_CODE)], S[W & (S != NO_CODE)], 0])
    power_of = np.array([float(activity_power.get(int(c), 0.0)) for c in codes])

    wide_path = PROCESSED_DIR / "household_interval_timeseries.csv"
    person_path = PROCESSED_DIR / "person_daily_metabolic_totals.csv"
    totals_path = PROCESSED_DIR / "household_daily_totals.csv"
    long_path = PROCESSED_DIR / "household_interval_timeseries_long.csv"
    if not WRITE_LONG and long_path.exists():
        long_path.unlink()          # a stale long file from an earlier run
        print(f"Removed {long_path}, left by an earlier run (set HETUS_WRITE_LONG=1 to write it)")
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1, len(g)]
    persons = person_totals(MR, MR_ok, W)
    matrix_parts = []
    for b0 in range(0, n_days, BLOCK):
        b1 = min(b0 + BLOCK, n_days)
        r0, r1 = starts[b0], starts[b1]
        gl = g[r0:r1] - b0
        power, metab, n_home, n_dom, n_codes, entries = household_block(
            gl, M[r0:r1], S[r0:r1], W[r0:r1], MR[r0:r1], DOM[r0:r1], power_of, codes)
        if entries is not None:
            matrix_parts.append((b0, b1 - b0, entries))
        first_rec = starts[b0:b1] - r0
        attrs = meta.iloc[r0:r1].iloc[first_rec].reset_index(drop=True)
        ids = attrs[["HID", "DIARY"]]
        extra = [c for c in EXTRA_COLS if c in attrs.columns]
        wide = pd.concat([attrs[["HID", "DIARY"] + extra],
                          pd.DataFrame(power, columns=[f"Household_Energy_W_{t}" for t in range(SLOTS)]),
                          pd.DataFrame(metab, columns=[f"Household_Metabolic_W_{t}" for t in range(SLOTS)])],
                         axis=1)
        write(wide, wide_path, b0 == 0)
        recs = meta.iloc[r0:r1]
        write(pd.DataFrame({"HID": recs["HID"].values, "PID": recs["PID"].values,
                            "DIARY": recs["DIARY"].values,
                            "diarist_total_metabolic_Wh": persons[r0:r1],
                            "note": "Person-level total includes all at-home intervals"}),
              person_path, b0 == 0)
        totals = pd.concat([attrs[["HID", "DIARY"] + extra], pd.DataFrame({
            "total_energy_Wh": (power * INTERVAL_HOURS).sum(axis=1),
            "total_metabolic_Wh": (metab * INTERVAL_HOURS).sum(axis=1),
            "avg_diarists_at_home": n_home.sum(axis=1) / SLOTS,
            "avg_unique_activities": n_dom.sum(axis=1) / SLOTS})], axis=1)
        write(totals, totals_path, b0 == 0)
        if WRITE_LONG:
            n = b1 - b0
            long = pd.DataFrame({
                "HID": np.repeat(ids["HID"].values, SLOTS),
                "DIARY": np.repeat(ids["DIARY"].values, SLOTS),
                "Interval": np.tile(np.arange(SLOTS), n),
                "Household_Energy_W": power.ravel(), "Household_Metabolic_W": metab.ravel(),
                "Has_At_Home_Activities": n_home.ravel() > 0,
                "n_diarists_at_home": n_home.ravel(),
                "n_unique_activities_after_dedup": n_dom.ravel(),
                "n_total_activities": n_codes.ravel()})
            for c in extra:
                long[c] = np.repeat(attrs[c].values, SLOTS)
            write(long, long_path, b0 == 0)
        print(f"  household-days {b0 + 1}-{b1} of {n_days} done ({time.time() - t0:.0f} s)")

    if SAVE_MATRIX:
        save_matrices(matrix_parts, codes, power_of, meta.iloc[starts[:-1]][["HID", "DIARY"]])

    daily = pd.read_csv(totals_path)
    print(f"Exported {totals_path} with {len(daily)} rows")
    print(f"Average daily household energy: {daily['total_energy_Wh'].mean():.1f} Wh")
    print(f"Average daily household metabolic: {daily['total_metabolic_Wh'].mean():.1f} Wh")
    print(f"Average diarists at home: {daily['avg_diarists_at_home'].mean():.2f}")
    print(f"Average unique activities after dedup: {daily['avg_unique_activities'].mean():.2f}")
    print("\nProcessing complete!")
    print(f"- Wide format: {n_days} rows in {wide_path}")
    print(f"- Person data: {len(meta)} rows in {person_path}")
    if WRITE_LONG:
        print(f"- Long format: {n_days * SLOTS} rows in {long_path}")
    print(f"Finished aggregate_household_metrics.py in {time.time() - t0:.0f} s")


def save_matrices(parts, codes, power_of, days):
    """Write the matrices of the sensitivity analysis: rows are household-day
    x slot (household-day h, slot t at row 144 h + t, household-days in the
    order of the wide file), columns the activity codes in `codes`."""
    from scipy import sparse
    nc = len(codes)
    mats = {"union": [], "main": [], "count": []}
    for b0, n, (union, main, uk, cnt) in parts:
        shape = (n * SLOTS, nc)
        for name, keys, data in (("union", union, None), ("main", main, None), ("count", uk, cnt)):
            rows, cols = keys // nc, keys % nc
            vals = np.ones(len(keys), np.int8) if data is None else data.astype(np.int16)
            mats[name].append(sparse.csr_matrix((vals, (rows, cols)), shape=shape))
    out = {name: sparse.vstack(blocks, format="csr") for name, blocks in mats.items()}
    np.savez_compressed(PROCESSED_DIR / "activity_matrix.npz", codes=codes, power=power_of,
                        **{f"{name}_{part}": getattr(m, part) for name, m in out.items()
                           for part in ("data", "indices", "indptr")},
                        shape=np.array(out["union"].shape))
    days.to_csv(PROCESSED_DIR / "activity_matrix_households.csv", index=False)
    print(f"Activity matrices saved to {PROCESSED_DIR / 'activity_matrix.npz'} "
          f"({out['union'].nnz} codes in use at home)")


if __name__ == "__main__":
    main()

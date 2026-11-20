"""Build the inputs of the sensitivity analysis from the outputs of a run.

Run on the machine that holds the microdata, after Stages 01 to 05 of a run
and with the same HETUS_* settings:

    python3 -m gsa.bundle --out FOLDER [--reference CLUSTERING_RESULTS.csv]

The bundle holds household-day microdata in a compact form (activity codes in
use at home, weights, background variables) and must never leave that
machine, like the processed folder it comes from. Only the outputs of the
model runs (gsa.run, gsa.collect) are aggregates that can be shared.

Every array is an uncompressed .npy file, so that the workers of gsa.run
memory-map one copy instead of loading one each. Household-days are in the
order of household_interval_timeseries.csv (h = 0 .. n - 1), and slot t of
household-day h is row 144 h + t:

  entries/     one element per household-day, slot and activity code in use at
               home, for the 28 codes with a power in some run (X1 to X6):
               row, day (h), code (position in codes.npy), cm (diarists at home
               using the code as main activity), cs (diarists at home using it
               as secondary activity only, that is with another main activity),
               fm and fs (the same for the first-listed diarist, as booleans)
  slots/       lit_all, lit_first: whether a diarist at home, or the
               first-listed diarist at home, is in an activity, main or
               secondary, with Light_use = 1; runon_all, runon_first: the number
               of slots since laundry (331) was last in use (1 to 12, otherwise
               0), and runon_*_main: whether that last slot had laundry as a
               main activity, all diarists or the first-listed one (n x 144)
  households/  country, day types (by DDV7 then DDV1 as in Stage 06, and by
               DDV1 only: 0 working, 1 non-working, -1 undetermined), survey
               weights (WGHT1 of the first-listed diarist and the mean over the
               household-day's diarists, each scaled to mean 1 within its
               country, 1 where missing), the row of the darkness table, the
               ownership items HHQ6c, HHQ6h and HHQ6o of the first-listed
               diarist as in the raw files (negative where not asked or not
               answered, -99 for an empty cell),
               the 16 background variables of the tests (as in
               prepared_clustering_data.csv, -98 not applicable, -99 missing),
               the cluster of the reference partition (output Y8; -1 if none)
               and the number of diarists
  darkness/    table: share of the days of each diary month on which each slot
               is dark at the capital (gsa.sun), and keys: its rows (country,
               year, month; month 0 where the month is unknown, which averages
               the twelve months)
  meta.json    sizes, codes, feature names, sources and the result of the
               check below

The bundle is checked against the run: with the powers of the mapping table it
must give Household_Energy_W_* of household_interval_timeseries.csv exactly in
every household-day and slot, or the build stops.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from paths import PROCESSED_DIR, RAW_DIR, MAPPING_FILE  # noqa: E402
from countries import COUNTRIES  # noqa: E402

from gsa.design import POWER  # noqa: E402
from gsa import sun  # noqa: E402

s03 = importlib.import_module("03_aggregate_household")

SLOTS = 144
# The codes whose power varies in some run, in a fixed order
POWER_CODES = np.array(sorted({c for codes in POWER.values() for c in codes}), dtype=np.int32)
LAUNDRY = 331
RUNON_SLOTS = 12            # X20 goes up to 120 minutes
FEATURES = ["DDV1", "DDV3", "DDV6", "COUNTRY", "INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
            "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"]
NOT_APPLICABLE, MISSING = -98, -99
# Raw columns read for the bundle; the identifiers as Stage 01 reads them
RAW_COLUMNS = ["HID", "PID", "DIARY", "WGHT1", "DDV3", "DDV4", "HHQ6C", "HHQ6H", "HHQ6O"]
ITEMS = {"HHQ6C": "hhq6c", "HHQ6H": "hhq6h", "HHQ6O": "hhq6o"}
GROUPS = ("entries", "slots", "households", "darkness")
BLOCK = s03.BLOCK                       # household-days per block, as in Stage 03


def count_label(n):
    """A count as it may be published: exact from 50, a class below."""
    n = int(n)
    if n == 0:
        return "0"
    return "<20" if n < 20 else ("20-49" if n < 50 else f"{n:,}")


def save_group(folder, group, **arrays):
    d = Path(folder) / group
    d.mkdir(parents=True, exist_ok=True)
    for name, a in arrays.items():
        np.save(d / f"{name}.npy", np.ascontiguousarray(a))


def load_group(folder, group, mmap=True):
    """The arrays of one group of a bundle, memory-mapped by default."""
    d = Path(folder) / group
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found: build the bundle with python3 -m gsa.bundle")
    return {p.stem: np.load(p, mmap_mode="r" if mmap else None) for p in sorted(d.glob("*.npy"))}


def code_index(codes, values):
    """Position of each value in `codes`, -1 where it is not one of them."""
    pos = np.searchsorted(codes, values)
    pos = np.minimum(pos, len(codes) - 1)
    return np.where(codes[pos] == values, pos, -1)


def by_ddv1(ddv1):
    """Stage 06's weekday rule: DDV1 2 to 6 (Monday to Friday) working, 1 or 7
    non-working, otherwise undetermined."""
    ddv1 = np.asarray(ddv1, dtype=float)
    return np.where(np.isin(ddv1, [2, 3, 4, 5, 6]), 0,
                    np.where(np.isin(ddv1, [1, 7]), 1, -1)).astype(np.int8)


def day_type_ddv7_then_ddv1(ddv7, ddv1):
    """Stage 06's rule (determine_day_type): DDV7 1 or 2 working, 3 to 6
    non-working, DDV1 where DDV7 is missing or negative, and undetermined
    otherwise; 0 working, 1 non-working, -1 undetermined."""
    ddv7 = np.asarray(ddv7, dtype=float)
    out = np.full(len(ddv7), -1, dtype=np.int8)
    missing = ~np.isfinite(ddv7) | (ddv7 < 0)
    out[missing] = by_ddv1(ddv1)[missing]
    out[np.isin(ddv7, [1, 2])] = 0
    out[np.isin(ddv7, [3, 4, 5, 6])] = 1
    return out


def feature_codes(column):
    """A background variable of prepared_clustering_data.csv as integers:
    numeric codes as they are (-98, -99 included), an empty cell -99, and any
    non-numeric code as a code of its own from 10,000 up, so that the tests
    see the same categories as Stage 07."""
    s = column.astype("string").str.strip()
    num = pd.to_numeric(s, errors="coerce")
    empty = s.isna() | (s == "") | (s.str.lower() == "nan")
    other = num.isna() & ~empty
    out = num.fillna(MISSING)
    if other.any():
        codes, _ = pd.factorize(s[other], sort=True)
        out[other] = 10000 + codes
    return out.round().astype(np.int16).to_numpy()


def diary_key(values):
    """DIARY as text without a trailing .0, so that integer and float columns
    match."""
    d = pd.to_numeric(pd.Series(values), errors="coerce")
    return d.astype("Int64").astype("string").fillna("NA").to_numpy()


def read_raw(country, raw_dir):
    """Identifiers, weight, month, year and ownership items of every record of
    a raw country file. Column names are matched as in Stage 01
    (harmonise_column_names): the exact name if present, otherwise the one
    that differs only in case or surrounding spaces; HID and PID are prefixed
    with the country as in Stage 01."""
    path = Path(raw_dir) / f"{country}.csv"
    header = list(pd.read_csv(path, nrows=0).columns)
    use = {}
    for c in RAW_COLUMNS:
        exact = [h for h in header if str(h) == c]
        loose = [h for h in header if str(h).strip().upper() == c]
        if exact or loose:
            use[c] = (exact or loose)[0]
    if not {"HID", "PID", "DIARY"} <= set(use):
        raise ValueError(f"{path}: HID, PID or DIARY not found")
    df = pd.read_csv(path, usecols=list(use.values()))
    df = df.rename(columns={v: k for k, v in use.items()})
    for c in RAW_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df["HID"] = country + "_" + df["HID"].astype(str)
    df["PID"] = country + "_" + df["PID"].astype(str)
    df["DIARY"] = diary_key(df["DIARY"])
    return df


def run_length_since(use, main):
    """For every slot of every row, the number of slots since the last slot in
    use (1 to RUNON_SLOTS after an episode, otherwise 0, in use included) and
    whether that last slot had the activity as a main activity."""
    n, t = use.shape
    slots = np.arange(t, dtype=np.int16)[None, :]
    idx = np.where(use, slots, np.int16(-1))
    last = np.maximum.accumulate(idx, axis=1)
    k = slots - last
    valid = (last >= 0) & ~use & (k <= RUNON_SLOTS)
    was_main = np.take_along_axis(main, np.maximum(last, 0), axis=1) & valid
    return np.where(valid, k, 0).astype(np.int8), was_main


def block_entries(gl, first, M, S, W, light_codes):
    """Entries and light of a block of household-days (gl: the block-local
    household-day of every record, records sorted by it). Returns the entry
    identifiers (144 h + t) * len(POWER_CODES) + code, sorted, with cm, cs, fm
    and fs, and the slots (144 h + t) in which a diarist at home, and the
    first-listed one, is in an activity with Light_use = 1."""
    r, t = np.nonzero(W)                         # records at home
    key = gl[r].astype(np.int64) * SLOTS + t
    m, s, fr = M[r, t], S[r, t], first[r]
    nc = len(POWER_CODES)
    km, ks = code_index(POWER_CODES, m), code_index(POWER_CODES, s)
    main_sel = km >= 0
    sec_sel = (ks >= 0) & (s != m)               # secondary only: another main activity
    id_m = key[main_sel] * nc + km[main_sel]
    id_s = key[sec_sel] * nc + ks[sec_sel]
    ids = np.unique(np.r_[id_m, id_s])
    pos_m, pos_s = np.searchsorted(ids, id_m), np.searchsorted(ids, id_s)
    cm = np.bincount(pos_m, minlength=len(ids))
    cs = np.bincount(pos_s, minlength=len(ids))
    fm = np.bincount(pos_m, weights=fr[main_sel], minlength=len(ids)) > 0
    fs = np.bincount(pos_s, weights=fr[sec_sel], minlength=len(ids)) > 0
    lit = (code_index(light_codes, m) >= 0) | (code_index(light_codes, s) >= 0)
    return ids, cm, cs, fm, fs, np.unique(key[lit]), np.unique(key[lit & fr])


def read_records(processed):
    """Identifiers and the codes and locations of every diarist record of
    Stage 02's output, sorted by household-day as in Stage 03."""
    path = Path(processed) / s03.INPUT.name
    print(f"Reading the diarist records of {path}")
    header = pd.read_csv(path, nrows=0).columns
    cols = ["HID", "PID", "DIARY", "COUNTRY"] + s03.MACT + s03.SACT + s03.WHERE
    meta, M, S, W = [], [], [], []
    for ch in pd.read_csv(path, usecols=[c for c in cols if c in header], chunksize=20000,
                          low_memory=False):
        meta.append(ch[["HID", "PID", "DIARY"]])
        M.append(s03.as_codes(ch[s03.MACT]))
        S.append(s03.as_codes(ch[s03.SACT]))
        W.append(ch[s03.WHERE].apply(pd.to_numeric, errors="coerce").to_numpy() == 1)
    meta = pd.concat(meta, ignore_index=True)
    M, S, W = np.vstack(M), np.vstack(S), np.vstack(W)
    g = s03.group_records(meta)
    keep = np.flatnonzero(g >= 0)
    order = keep[np.argsort(g[keep], kind="stable")]        # by household-day, file order kept
    return meta.iloc[order].reset_index(drop=True), M[order], S[order], W[order], g[order]


def build(out, processed=PROCESSED_DIR, raw_dir=RAW_DIR, mapping=MAPPING_FILE, reference=None):
    t0 = time.time()
    out, processed = Path(out), Path(processed)
    out.mkdir(parents=True, exist_ok=True)
    meta, M, S, W, g = read_records(processed)
    n = int(g.max()) + 1
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    first = np.zeros(len(g), bool)
    first[starts] = True
    print(f"  {len(meta):,} diarist records, {n:,} household-days")

    # codes in use at home, as entries (household-day x slot, code), and
    # light, in blocks of household-days as in Stage 03 to keep memory low
    table = pd.read_csv(mapping)
    table.columns = [str(c).replace("\ufeff", "").strip() for c in table.columns]
    light_codes = np.array(sorted(table.loc[table["Light_use"] == 1, "activity"].astype(int)),
                           dtype=np.int32)
    nc = len(POWER_CODES)
    bounds = np.r_[starts, len(g)]
    parts = []
    lit_all, lit_first = np.zeros(n * SLOTS, bool), np.zeros(n * SLOTS, bool)
    for b0 in range(0, n, BLOCK):
        b1 = min(b0 + BLOCK, n)
        r0, r1 = bounds[b0], bounds[b1]
        ids, cm, cs, fm, fs, la, lf = block_entries(g[r0:r1] - b0, first[r0:r1], M[r0:r1],
                                                    S[r0:r1], W[r0:r1], light_codes)
        parts.append((ids + b0 * SLOTS * nc, cm, cs, fm, fs))
        lit_all[la + b0 * SLOTS] = True
        lit_first[lf + b0 * SLOTS] = True
    ids, cm, cs, fm, fs = (np.concatenate(a) for a in zip(*parts))
    del parts, M, S, W
    row, code = (ids // nc).astype(np.int32), (ids % nc).astype(np.int8)
    if cm.max(initial=0) > 127 or cs.max(initial=0) > 127:
        raise ValueError("more than 127 diarists in one household-day")
    save_group(out, "entries", row=row, day=(row // SLOTS).astype(np.int32), code=code,
               cm=cm.astype(np.int8), cs=cs.astype(np.int8), fm=fm, fs=fs, codes=POWER_CODES)
    print(f"  {len(ids):,} entries for the {nc} codes with a power ({time.time() - t0:.0f} s)")

    # laundry, by household-day and slot
    laundry = int(np.flatnonzero(POWER_CODES == LAUNDRY)[0])
    is_l = code == laundry
    use_all, main_all = np.zeros(n * SLOTS, bool), np.zeros(n * SLOTS, bool)
    use_all[row[is_l]] = True
    main_all[row[is_l & (cm > 0)]] = True
    use_first, main_first = np.zeros(n * SLOTS, bool), np.zeros(n * SLOTS, bool)
    use_first[row[is_l & (fm | fs)]] = True
    main_first[row[is_l & fm]] = True
    k_all, km_all = run_length_since(use_all.reshape(n, SLOTS), main_all.reshape(n, SLOTS))
    k_first, km_first = run_length_since(use_first.reshape(n, SLOTS), main_first.reshape(n, SLOTS))
    save_group(out, "slots", lit_all=lit_all.reshape(n, SLOTS), lit_first=lit_first.reshape(n, SLOTS),
               runon_all=k_all, runon_all_main=km_all, runon_first=k_first, runon_first_main=km_first)
    del lit_all, lit_first, use_all, main_all, use_first, main_first

    # check: the powers of the mapping table reproduce Stage 03 exactly
    power_of = (table.set_index("activity")["Appliance_Power_Watts"]
                .reindex(POWER_CODES).fillna(0).to_numpy(dtype=float))
    p = np.bincount(row, weights=power_of[code], minlength=n * SLOTS).reshape(n, SLOTS)
    energy_cols = [f"Household_Energy_W_{i}" for i in range(SLOTS)]
    wide = pd.read_csv(processed / "household_interval_timeseries.csv",
                       usecols=["HID", "DIARY", "COUNTRY"] + energy_cols,
                       float_precision="round_trip", low_memory=False)
    first_meta = meta.iloc[starts].reset_index(drop=True)
    same_ids = (len(wide) == n
                and (wide["HID"].astype(str).to_numpy() == first_meta["HID"].astype(str).to_numpy()).all()
                and (diary_key(wide["DIARY"]) == diary_key(first_meta["DIARY"])).all())
    if not same_ids:
        raise ValueError("the household-days differ from household_interval_timeseries.csv: "
                         "rerun Stage 03 with the same settings")
    diff = np.abs(wide[energy_cols].to_numpy(dtype=float) - p)
    power_check = {"slot_values": int(p.size), "differing": int((diff > 0).sum()),
                   "largest_difference_w": float(diff.max(initial=0))}
    print(f"  power check against Stage 03: {power_check['differing']:,} of {p.size:,} "
          f"slot values differ")
    if power_check["differing"]:
        raise ValueError("the bundle does not reproduce the power of Stage 03")
    country = pd.to_numeric(wide["COUNTRY"], errors="raise").to_numpy().astype(np.int8)
    del wide, diff, p

    # household-day attributes, from Stage 04 (same rows, same order)
    prepared = pd.read_csv(processed / "prepared_clustering_data.csv",
                           usecols=FEATURES + ["DDV7"], dtype=str, keep_default_na=False)
    if len(prepared) != n:
        raise ValueError("prepared_clustering_data.csv does not match the household-days of Stage 03")
    feats = np.column_stack([feature_codes(prepared[f]) for f in FEATURES])
    if not (feats[:, FEATURES.index("COUNTRY")] == country).all():
        raise ValueError("prepared_clustering_data.csv is not in the order of Stage 03")
    ddv7 = pd.to_numeric(prepared["DDV7"], errors="coerce").to_numpy(dtype=float)
    ddv1 = pd.to_numeric(prepared["DDV1"], errors="coerce").to_numpy(dtype=float)

    print("Reading weights, diary months and ownership items from the raw files")
    present = sorted(set(country.tolist()))
    raw = pd.concat([read_raw(COUNTRIES[c], raw_dir) for c in present], ignore_index=True)
    if raw.duplicated(["HID", "PID", "DIARY"]).any():
        raise ValueError("repeated (HID, PID, DIARY) records in the raw files")
    keys = pd.DataFrame({"HID": meta["HID"].astype(str).to_numpy(),
                         "PID": meta["PID"].astype(str).to_numpy(),
                         "DIARY": diary_key(meta["DIARY"])})
    rec = keys.merge(raw, on=["HID", "PID", "DIARY"], how="left", indicator=True)
    unmatched = int((rec["_merge"] != "both").sum())
    if unmatched or len(rec) != len(meta):
        raise ValueError(f"{count_label(unmatched)} diarist records not found in the raw files")
    wght = pd.to_numeric(rec["WGHT1"], errors="coerce").to_numpy(dtype=float, copy=True)
    wght[~(wght > 0)] = np.nan
    w_first = wght[starts]
    valid = np.bincount(g, weights=np.isfinite(wght), minlength=n)
    w_mean = np.bincount(g, weights=np.nan_to_num(wght), minlength=n) / np.maximum(valid, 1)
    w_mean[valid == 0] = np.nan

    def scaled(w):
        w = w.copy()
        for c in present:
            sel = country == c
            ok = sel & np.isfinite(w)
            w[sel] = w[sel] / w[ok].mean() if ok.any() else 1.0
            w[sel & ~ok] = 1.0
        return w

    weights_found = {COUNTRIES[c]: count_label(np.isfinite(w_first[country == c]).sum())
                     for c in present}
    month = pd.to_numeric(rec["DDV3"], errors="coerce").to_numpy(dtype=float)[starts]
    year = pd.to_numeric(rec["DDV4"], errors="coerce").to_numpy(dtype=float)[starts]
    month = np.where(np.isfinite(month) & (month >= 1) & (month <= 12), month, 0).astype(np.int16)
    year_ok = np.isfinite(year) & (year > 1990) & (year < 2100)
    year = np.where(year_ok, year, 0).astype(np.int16)
    for c in present:                       # a missing year: the country's most common one
        sel = country == c
        common = np.bincount(year[sel & year_ok]).argmax() if (sel & year_ok).any() else 2010
        year[sel & ~year_ok] = common
    cym = list(zip(country.tolist(), year.tolist(), month.tolist()))
    combos = sorted(set(cym))
    dark = np.array([sun.darkness(COUNTRIES[c], y, mo) if mo else
                     np.mean([sun.darkness(COUNTRIES[c], y, k) for k in range(1, 13)], axis=0)
                     for c, y, mo in combos], dtype=np.float64)
    lookup = {k: i for i, k in enumerate(combos)}
    dark_row = np.array([lookup[k] for k in cym], dtype=np.int32)
    save_group(out, "darkness", table=dark, keys=np.array(combos, dtype=np.int16))

    own = {}
    for item, name in ITEMS.items():
        v = pd.to_numeric(rec[item], errors="coerce").to_numpy(dtype=float)[starts]
        own[name] = np.where(np.isfinite(v), v, MISSING).astype(np.int16)

    reference = Path(reference) if reference else processed / "clustering_results.csv"
    labels = np.full(n, -1, dtype=np.int16)
    if reference.is_file():
        ref = pd.read_csv(reference, usecols=["cluster"])
        if len(ref) == n:
            labels = ref["cluster"].to_numpy().astype(np.int16)
        else:
            print(f"  {reference} does not have one row per household-day: Y8 will be missing")
    else:
        print(f"  {reference} not found: Y8 will be missing")

    save_group(out, "households", country=country,
               day_type_ddv7=day_type_ddv7_then_ddv1(ddv7, ddv1), day_type_ddv1=by_ddv1(ddv1),
               weight_first=scaled(w_first), weight_mean=scaled(w_mean), dark_row=dark_row,
               features=feats, reference=labels,
               n_diarists=np.bincount(g, minlength=n).astype(np.int16), **own)
    info = {"household_days": n, "records": int(len(meta)), "entries": int(len(ids)),
            "power_codes": POWER_CODES.tolist(), "light_codes": light_codes.tolist(),
            "feature_names": FEATURES, "countries": [COUNTRIES[c] for c in present],
            "first_listed_weight_found": weights_found, "power_check": power_check,
            "unknown_month": count_label((month == 0).sum()),
            "reference": str(reference), "reference_found": bool((labels >= 0).any()),
            "processed": str(processed), "mapping": str(mapping),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(info, indent=1))
    print(f"  first-listed diarists with a weight, by country: {weights_found}")
    print(f"  household-days with an unknown diary month: {info['unknown_month']}")
    print(f"Bundle written to {out} in {time.time() - t0:.0f} s. It holds microdata: "
          f"keep it on this machine.")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="folder of the bundle")
    parser.add_argument("--reference", type=Path, default=None,
                        help="clustering_results.csv of the reference partition of Y8 "
                             "(default: the run's own)")
    args = parser.parse_args()
    build(args.out, reference=args.reference)


if __name__ == "__main__":
    main()

"""Compare the outputs of Stage 03 with those of a reference run.

The reference is a folder with the files of an earlier run of Stage 03: the
loop version (pipeline/aggregate_household_loop_reference.py writes to
processed/stage03_loop_reference/), or a copy of the processed folder of an
earlier version. The current files are read from the processed folder of
pipeline/paths.py.

    python3 tools/compare_stage03.py --reference FOLDER [--power-only] [--long]

Every comparison is exact (floats are read back to the last bit), except the
daily totals, which are sums of 144 rounded products and are compared to
1e-9 Wh. With --power-only, only the identifiers and the 144 power slots of
household_interval_timeseries.csv are compared, for a reference from a
version whose attributes or metabolic rates differ by design (for instance
1.0.0, before the corrections D5, D10 and L11 of the audit of November 2026).
With --long, the long files are compared as well, if both exist.

The report gives counts of differing values, with counts below 50 given as a
class, and maximal differences, so it can be shared. The exit status is 1 if
anything differs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from paths import PROCESSED_DIR  # noqa: E402

SLOTS = 144
POWER = [f"Household_Energy_W_{t}" for t in range(SLOTS)]
METAB = [f"Household_Metabolic_W_{t}" for t in range(SLOTS)]


def count_label(n):
    n = int(n)
    if n == 0:
        return "0"
    if n < 20:
        return "1 to 19"
    if n < 50:
        return "20 to 49"
    return f"{n:,}"


def read(path, usecols=None):
    return pd.read_csv(path, usecols=usecols, float_precision="round_trip", low_memory=False)


def differing(a, b):
    """Number of differing cells of two numeric frames (NaN equal to NaN),
    and the largest absolute difference."""
    x = a.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    y = b.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    both_nan = np.isnan(x) & np.isnan(y)
    diff = ~both_nan & ~(x == y)
    with np.errstate(invalid="ignore"):
        gap = np.where(diff, np.abs(x - y), 0.0)
    return int(diff.sum()), (float(np.nanmax(gap)) if diff.any() else 0.0), diff


class Report:
    def __init__(self):
        self.lines, self.ok = [], True

    def add(self, text, ok=True):
        self.lines.append(("   " if ok else "!! ") + text)
        self.ok &= ok


def same_keys(cur, ref, keys, name, report):
    n_cur, n_ref = len(cur), len(ref)
    report.add(f"{name}: {n_cur:,} rows here, {n_ref:,} in the reference", n_cur == n_ref)
    if n_cur != n_ref:
        return False
    k_cur = cur[keys].astype(str).agg("|".join, axis=1).to_numpy()
    k_ref = ref[keys].astype(str).agg("|".join, axis=1).to_numpy()
    in_order = bool((k_cur == k_ref).all())
    same_set = in_order or set(k_cur) == set(k_ref)
    report.add(f"{name}: identifiers {'in the same order' if in_order else 'the same set, other order' if same_set else 'differ'}",
               same_set)
    return same_set


def align(cur, ref, keys):
    """Reference rows in the order of the current file."""
    k_cur = cur[keys].astype(str).agg("|".join, axis=1)
    k_ref = ref[keys].astype(str).agg("|".join, axis=1)
    pos = pd.Series(np.arange(len(ref)), index=k_ref.values)
    return ref.iloc[pos.loc[k_cur.values].to_numpy()].reset_index(drop=True)


def compare_wide(ref_dir, power_only, report):
    name = "household_interval_timeseries.csv"
    cur, ref = read(PROCESSED_DIR / name), read(Path(ref_dir) / name)
    keys = ["HID", "DIARY"]
    if not same_keys(cur, ref, keys, name, report):
        return
    ref = align(cur, ref, keys)
    n, gap, diff = differing(cur[POWER], ref[POWER])
    days = int(diff.any(axis=1).sum())
    report.add(f"power: {count_label(n)} of {cur[POWER].size:,} slot values differ "
               f"(household-days: {count_label(days)}; largest difference {gap:g} W)", n == 0)
    if power_only:
        return
    n, gap, diff = differing(cur[METAB], ref[METAB])
    report.add(f"metabolic rate: {count_label(n)} slot values differ "
               f"(largest difference {gap:.3g} W)", n == 0)
    attrs = [c for c in cur.columns if c not in keys + POWER + METAB]
    missing = [c for c in attrs if c not in ref.columns]
    report.add(f"attribute columns: {len(attrs)}" + (f", missing in the reference: {missing}" if missing else ""),
               not missing)
    for col in [c for c in attrs if c in ref.columns]:
        n, gap, _ = differing(cur[[col]], ref[[col]])
        if n:
            report.add(f"attribute {col}: {count_label(n)} household-days differ", False)


def compare_person(ref_dir, report):
    name = "person_daily_metabolic_totals.csv"
    cur, ref = read(PROCESSED_DIR / name), read(Path(ref_dir) / name)
    keys = ["HID", "PID", "DIARY"]
    if not same_keys(cur, ref, keys, name, report):
        return
    if cur.duplicated(keys).any() or ref.duplicated(keys).any():
        ref_aligned = ref                      # repeated records: keep file order
    else:
        ref_aligned = align(cur, ref, keys)
    n, gap, _ = differing(cur[["diarist_total_metabolic_Wh"]], ref_aligned[["diarist_total_metabolic_Wh"]])
    report.add(f"person metabolic totals: {count_label(n)} differ (largest difference {gap:.3g} Wh)", n == 0)


def compare_totals(ref_dir, report):
    name = "household_daily_totals.csv"
    cur, ref = read(PROCESSED_DIR / name), read(Path(ref_dir) / name)
    keys = ["HID", "DIARY"]
    if not same_keys(cur, ref, keys, name, report):
        return
    ref = align(cur, ref, keys)
    for col in ["total_energy_Wh", "total_metabolic_Wh", "avg_diarists_at_home", "avg_unique_activities"]:
        x, y = cur[col].to_numpy(float), ref[col].to_numpy(float)
        gap = float(np.nanmax(np.abs(x - y))) if len(x) else 0.0
        report.add(f"daily totals, {col}: largest difference {gap:.3g}", gap <= 1e-9 * max(1.0, np.nanmax(np.abs(y))))


def compare_long(ref_dir, report, chunk=1_000_000):
    name = "household_interval_timeseries_long.csv"
    a, b = PROCESSED_DIR / name, Path(ref_dir) / name
    if not (a.is_file() and b.is_file()):
        report.add(f"{name}: not in both folders, not compared")
        return
    total, bad = 0, 0
    for x, y in zip(pd.read_csv(a, chunksize=chunk, float_precision="round_trip"),
                    pd.read_csv(b, chunksize=chunk, float_precision="round_trip")):
        if len(x) != len(y) or list(x.columns) != list(y.columns):
            report.add(f"{name}: chunks differ in shape or columns", False)
            return
        same_id = (x[["HID", "DIARY", "Interval"]].astype(str).to_numpy()
                   == y[["HID", "DIARY", "Interval"]].astype(str).to_numpy()).all()
        if not same_id:
            report.add(f"{name}: rows are not in the same order", False)
            return
        num = [c for c in x.columns if c not in ("HID", "DIARY")]
        n, _, _ = differing(x[num].astype(float), y[num].astype(float))
        bad += n
        total += len(x)
    report.add(f"{name}: {count_label(bad)} differing values in {total:,} rows", bad == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--power-only", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    report = Report()
    print(f"Current files: {PROCESSED_DIR}\nReference: {args.reference}\n")
    compare_wide(args.reference, args.power_only, report)
    if not args.power_only:
        compare_person(args.reference, report)
        compare_totals(args.reference, report)
        if args.long:
            compare_long(args.reference, report)
    print("\n".join(report.lines))
    print("\nIdentical." if report.ok else "\nDifferences found (lines marked !!).")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()

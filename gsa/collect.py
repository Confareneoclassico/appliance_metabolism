"""Gather the results of the runs into the files that can be shared.

    python3 -m gsa.collect --runs RUNS [--out SHARE] [--check REPORT]

Writes to SHARE (default: a folder share/ next to RUNS):

  gsa_outputs.csv    one row per run: its index, the coordinates of the factors
                     and dummies (u_*), the factor values as numbers
                     (gsa.design.analysis_matrix: levels as indices), the
                     outputs Y1 to Y11 and Y2b, and the diagnostics of the run
  gsa_profiles.npz   mean profiles of every run by country (and ALL) and day
                     type (working, non-working, all), float32, runs x
                     countries x 3 x 144; a profile of fewer than 50
                     household-days would be left empty (none is expected)
  manifest.txt       the settings of the sample, which runs are present, and
                     the checks below

Every value is an aggregate over at least 50 household-days, the publication
rules of the pipeline (disclosure.py); the numbers of household-days stay in
the run files. The report of gsa.check is copied if found (--check, or
baseline_check.txt and .json in SHARE already).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsa import design  # noqa: E402
from gsa.model import OUTPUTS, MIN_N_UNFLAGGED  # noqa: E402

DIAGNOSTICS = ["Y5_variable", "solutions", "k_global", "k_fallbacks", "pca_components",
               "clusters_merged", "mean_daily_energy_wh"]


def ranges(idx):
    """Text of a sorted list of integers as ranges ('0-127, 130')."""
    out, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = i
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(out)


def collect(runs, out, check=None):
    runs, out = Path(runs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((runs / "design.json").read_text())
    files = sorted(runs.glob("run_[0-9][0-9][0-9][0-9][0-9].json"))
    records = [json.loads(f.read_text()) for f in files]
    records = [r for r in records if (runs / f"run_{r['run']:05d}.npz").is_file()]
    if not records:
        sys.exit(f"no complete runs in {runs}")
    idx = np.array([r["run"] for r in records])
    order = np.argsort(idx)
    records, idx = [records[i] for i in order], idx[order]
    U = np.array([r["coordinates"] for r in records])
    X = design.analysis_matrix(U)
    table = pd.DataFrame({"run": idx})
    for j, name in enumerate(design.COLUMNS):
        table[f"u_{name}"] = U[:, j]
    for j, name in enumerate(design.COLUMNS):
        table[name] = X[:, j]
    for y in OUTPUTS + DIAGNOSTICS:
        table[y] = [r["outputs"].get(y) for r in records]
    table["seconds"] = [r["seconds"] for r in records]

    profiles, blanked, countries = [], 0, None
    for i in idx:
        z = np.load(runs / f"run_{i:05d}.npz")
        prof = z["profiles"].astype(np.float32)
        small = z["counts"] < MIN_N_UNFLAGGED
        blanked += int((small & (z["counts"] > 0)).sum())
        prof[small] = np.nan
        profiles.append(prof)
        countries = [str(c) for c in z["countries"]]
    profiles = np.stack(profiles)

    table.to_csv(out / "gsa_outputs.csv", index=False)
    np.savez_compressed(out / "gsa_profiles.npz", profiles=profiles, runs=idx,
                        countries=np.array(countries), day_types=np.array(["working", "non_working", "all"]))
    if check and Path(check).is_file():
        for suffix in (".txt", ".json"):
            src = Path(check).with_suffix(suffix)
            if src.is_file() and src.resolve() != (out / f"baseline_check{suffix}").resolve():
                shutil.copy(src, out / f"baseline_check{suffix}")

    n = len(idx)
    prefix = int(np.max(np.flatnonzero(idx == np.arange(n))) + 1) if idx[0] == 0 else 0
    nested = [2 ** k for k in range(7, 14) if 2 ** k <= prefix]
    lines = [f"Sample: scrambled Sobol' sequence, seed {settings['sample_seed']}, up to 2^{settings['m']} "
             f"points; powers {'uniform in W' if settings['uniform_in_watts'] else 'log-uniform'}; "
             f"K-means n_init {settings['n_init']}; bundle of {settings['household_days']:,} "
             f"household-days built {settings['bundle_created']}",
             f"Runs present: {n} ({ranges(idx.tolist())}); complete nested designs: "
             f"{', '.join(map(str, nested)) or 'none'}",
             f"Seconds per run: median {np.median(table['seconds']):.0f}, "
             f"largest {np.max(table['seconds']):.0f}"]
    for y in OUTPUTS:
        v = pd.to_numeric(table[y], errors="coerce")
        lines.append(f"{y}: {int(v.isna().sum())} missing, range {v.min():.4g} to {v.max():.4g}")
    sol = table["solutions"].value_counts().sort_index()
    lines.append("Solutions with a strongest correlate, by run: "
                 + ", ".join(f"{int(k)} in {int(c)} runs" for k, c in sol.items()))
    lines.append(f"Profiles of fewer than {MIN_N_UNFLAGGED} household-days left empty: {blanked}")
    lines.append("Every value is an aggregate over at least 50 household-days; no microdata.")
    (out / "manifest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Files to share written to {out}")
    return table, profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: share/ next to RUNS")
    parser.add_argument("--check", type=Path, default=None, help="report of gsa.check to include")
    args = parser.parse_args(argv)
    out = args.out or args.runs.resolve().parent / "share"
    collect(args.runs, out, args.check)


if __name__ == "__main__":
    main()

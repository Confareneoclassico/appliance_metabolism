"""Collect the files of a run that may leave the machine holding the microdata.

The working files of the pipeline (for instance clustering_results.csv, one
row per household-day) must never leave that machine. This script copies only
the aggregate exports listed in EXPORTS, after testing each table with the
rules of pipeline/disclosure.py, into one folder that can be zipped and
shared, and writes a manifest of what was copied and what was not. A table
that fails a test, or a file that is missing, is not copied.

    python3 tools/collect_exports.py [--out FOLDER] [--log RUN_LOG [RUN_LOG ...]]

The folder defaults to exports_for_sharing/ at the root of the repository
(ignored by git). The tables are read from the folders of pipeline/paths.py,
so the same HETUS_* settings as for the run apply. With --log, the lines of
the saved console logs (several for a run made in parts) that hold only
solution-level figures (PCA, WCSS, the number of clusters and how it was
chosen, stability) are written to log_extract.txt; read it before sharing it.
The logs themselves are working files and are not copied.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from paths import (PROCESSED_DIR, COUNTRY_DIR, COUNTRY_DAYTYPE_DIR,  # noqa: E402
                   FIGURES_DIR)
from disclosure import MIN_N_UNFLAGGED, check_export  # noqa: E402

PROFILE = re.compile(r"P_\d{3}$")

# (folder, file, size columns, profile test): the size columns may hold no
# value from 1 to MIN_N_UNFLAGGED - 1, and tables of mean profiles pass
# check_export, as one table ("plain") or stratum by stratum ("stratified")
EXPORTS = [
    (PROCESSED_DIR, "cluster_summary.csv", ["size"], None),
    (PROCESSED_DIR, "cluster_mean_profiles.csv", ["n"], "plain"),
    (PROCESSED_DIR, "cluster_stability.csv", ["size"], None),
    (PROCESSED_DIR, "consolidated_anova_stats.csv", [], None),
    (PROCESSED_DIR, "consolidated_chi2_stats.csv", ["N_Observations"], None),
    (PROCESSED_DIR, "consolidated_circular_stats.csv", ["n_households", "n_zero_demand"], None),
    (PROCESSED_DIR, "stratified_cluster_mean_profiles.csv", ["n", "stratum_n"], "stratified"),
    (PROCESSED_DIR, "same_activity_by_diarists.csv", ["household_days"], None),
    (PROCESSED_DIR, "activity_combination_counts.csv", [], None),
    (PROCESSED_DIR, "shared_activities_frequency.csv", ["household_days"], None),
    (COUNTRY_DIR, "country_comparison_summary.csv", ["n_households"], None),
    (COUNTRY_DAYTYPE_DIR, "country_daytype_combined_summary.csv",
     ["n_households", "largest_group_size"], None),
    (COUNTRY_DAYTYPE_DIR, "mean_profiles_country_daytype.csv", ["n"], "plain"),
    (COUNTRY_DAYTYPE_DIR, "peak_power_daytype_comparison.csv",
     ["n_households_working", "n_households_non_working"], None),
    (COUNTRY_DAYTYPE_DIR, "working_days_cross_country.csv", ["n_households"], None),
    (COUNTRY_DAYTYPE_DIR, "non_working_days_cross_country.csv", ["n_households"], None),
    # checks for the revision (tools/paired_day_types.py, driver_robustness.py,
    # clustering_robustness.py), copied when they have been run
    (COUNTRY_DAYTYPE_DIR, "paired_daytype_comparison.csv", ["n_paired_households"], None),
    (COUNTRY_DAYTYPE_DIR, "paired_daytype_comparison_ddv1.csv", ["n_paired_households"], None),
    (PROCESSED_DIR, "driver_robustness.csv", ["N_Observations"], None),
    (PROCESSED_DIR, "clustering_robustness.csv", [], None),
]
# Optional exports: not listed as missing in the manifest when absent
OPTIONAL = {"paired_daytype_comparison.csv", "paired_daytype_comparison_ddv1.csv",
            "driver_robustness.csv", "clustering_robustness.csv",
            "paired_daytype_comparison.png", "paired_daytype_comparison_ddv1.png"}
# Figures drawn from the published groups only
FIGURES = [
    (FIGURES_DIR, "cluster_energy_profiles_10min_combined.png"),
    (FIGURES_DIR, "cluster_energy_profiles_10min_panels.png"),
    (FIGURES_DIR, "country_cluster_profiles_small_multiples.png"),
    (FIGURES_DIR, "elbow_method_improved.png"),
    (COUNTRY_DAYTYPE_DIR, "country_daytype_profiles_small_multiples.png"),
    (FIGURES_DIR, "paired_daytype_comparison.png"),
    (FIGURES_DIR, "paired_daytype_comparison_ddv1.png"),
]
# Console lines with solution-level figures only, and the tagged lines of
# Stage 01 (country shares and counts in classes below 50)
LOG_PATTERNS = re.compile(
    r"PCA|WCSS|[Oo]ptimal number of clusters|optimal k|[Kk]nee|Silhouette|\bARI\b|"
    r"PERFORMING CLUSTERING ANALYSIS|Number of households: \d|Clustering features|"
    r"\[same-day rule\]|\[secondary 999\]")


def problems(table, size_cols, profile_test):
    """Reasons why a table may not be shared (empty if none)."""
    found = []
    for col in size_cols:
        if col in table:
            v = pd.to_numeric(table[col], errors="coerce")
            if ((v > 0) & (v < MIN_N_UNFLAGGED)).any():
                found.append(f"{col} holds a size below {MIN_N_UNFLAGGED}")
    if profile_test:
        prof = [c for c in table.columns if PROFILE.fullmatch(str(c))]
        keep = ["n"] + (["stratum", "stratum_n"] if profile_test == "stratified" else [])
        try:
            check_export(table[keep + prof], n_col="n",
                         stratum_col="stratum" if profile_test == "stratified" else None,
                         stratum_total_col="stratum_n" if profile_test == "stratified" else None)
        except ValueError as e:
            found.append(str(e))
    return found


def collect(out, log=None):
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for folder, name, size_cols, profile_test in EXPORTS:
        src = Path(folder) / name
        if not src.is_file():
            if name not in OPTIONAL:
                manifest.append(f"MISSING  {name}")
            continue
        found = problems(pd.read_csv(src), size_cols, profile_test)
        if found:
            manifest.append(f"REFUSED  {name}: " + "; ".join(found))
            continue
        shutil.copy2(src, out / name)
        manifest.append(f"copied   {name}")
    for folder, name in FIGURES:
        src = Path(folder) / name
        if src.is_file():
            shutil.copy2(src, out / name)
            manifest.append(f"copied   {name}")
        elif name not in OPTIONAL:
            manifest.append(f"MISSING  {name}")
    logs = [] if not log else ([log] if isinstance(log, (str, Path)) else list(log))
    if logs:
        lines, parts = [], []
        for one in logs:
            found = [line for line in Path(one).read_text(errors="replace").splitlines()
                     if LOG_PATTERNS.search(line)]
            lines += found
            parts.append(f"{len(found)} lines of {one}")
        (out / "log_extract.txt").write_text("\n".join(lines) + "\n")
        manifest.append(f"copied   log_extract.txt ({', '.join(parts)}; read it before sharing)")
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    return manifest


def nothing_found(manifest):
    """True when none of the tables was found, which usually means that
    HETUS_PROCESSED_DIR differs from the one of the run."""
    tables = [line for line in manifest if ".csv" in line.split(":")[0]]
    return bool(tables) and all(line.startswith("MISSING") for line in tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "exports_for_sharing")
    parser.add_argument("--log", type=Path, nargs="+", default=None,
                        help="saved console logs of the run (several for a run made in parts)")
    args = parser.parse_args()
    manifest = collect(args.out, args.log)
    print("\n".join(manifest))
    if nothing_found(manifest):
        print(f"\nNone of the tables was found in {PROCESSED_DIR}. Set HETUS_PROCESSED_DIR "
              "(and HETUS_FIGURES_DIR) as for the run, then collect again.")
        sys.exit(1)
    print(f"\nFolder to share: {args.out}")
    if any(line.startswith("REFUSED") for line in manifest):
        sys.exit(1)


if __name__ == "__main__":
    main()

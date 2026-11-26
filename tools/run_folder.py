"""The processed folder of a run, as the tools that read it find and check it.

The checks for the reviewers (paired_day_types.py, driver_robustness.py,
clustering_robustness.py) read the files of Stages 03 to 06 from the processed
folder of a run. That folder is the one given with --processed, or the one the
bundle of the sensitivity analysis was built from (--bundle, the "processed"
field of its meta.json), or else that of pipeline/paths.py (HETUS_PROCESSED_DIR,
by default processed/ in the repository). A folder that lacks the solutions of
Stages 05 and 06 stops the tools with a message saying what is missing, rather
than with an empty table.
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from countries import NAME  # noqa: E402

HINT = ("Point the tools to the processed folder of the run: --bundle ~/hetus_gsa/bundle reads it "
        "from the bundle of the sensitivity analysis, --processed FOLDER gives it directly (in "
        "quotes if the path has spaces), or set HETUS_PROCESSED_DIR as for the run. If that folder "
        "no longer exists, rerun Stages 01 to 07 with the code of main and the same HETUS_* settings.")


def stop(message):
    """Print a message for the user and exit with status 2 (no traceback)."""
    print(f"\n{message}\n", file=sys.stderr)
    sys.exit(2)


def from_bundle(bundle):
    """The processed folder a bundle of gsa.bundle was built from."""
    meta = Path(bundle).expanduser() / "meta.json"
    if not meta.is_file():
        stop(f"{meta} not found. --bundle takes the folder of the bundle of the sensitivity "
             "analysis, the one that holds meta.json (for the canonical run, ~/hetus_gsa/bundle).")
    p = Path(json.loads(meta.read_text())["processed"]).expanduser()
    if not p.is_absolute():
        p = (meta.parent / p) if (meta.parent / p).is_dir() else p
    return p.resolve()


def resolve(processed=None, bundle=None):
    """The processed folder from --processed, --bundle or pipeline/paths.py."""
    if processed is not None and bundle is not None:
        stop("Give --processed or --bundle, not both.")
    if bundle is not None:
        return from_bundle(bundle)
    if processed is not None:
        return Path(processed).expanduser()
    from paths import PROCESSED_DIR
    return Path(PROCESSED_DIR)


def solution_files(processed, which="all"):
    """The solutions Stage 07 tests, with the filter that selects their rows
    among the household-days: (path, country or None, day type or None), the
    day type 0 for working and 1 for non-working days."""
    p = Path(processed)
    files = [(p / "clustering_results.csv", None, None)]
    if which in ("countries", "all"):
        for f in sorted(glob.glob(str(p / "Country_Analysis" / "[A-Z][A-Z]_clustering_results.csv"))):
            files.append((Path(f), Path(f).name[:2], None))
    if which == "all":
        d = p / "DayType_Analysis"
        files += [(d / "Working_Days_clustering_results.csv", None, 0),
                  (d / "Non_Working_Days_clustering_results.csv", None, 1)]
        for t, tag in ((0, "working"), (1, "non_working")):
            for f in sorted(glob.glob(str(p / "Country_DayType_Analysis" /
                                          f"[A-Z][A-Z]_{tag}_days_clustering_results.csv"))):
                files.append((Path(f), Path(f).name[:2], t))
    return [f for f in files if f[0].is_file()]


def check(processed, solutions=True):
    """Stop unless `processed` holds the household-days of Stages 03 and 04 and,
    if `solutions`, the solutions of Stages 05 and 06."""
    p = Path(processed)
    if not p.is_dir():
        stop(f"The processed folder {p} does not exist.\n{HINT}")
    missing = [name for name in ("household_interval_timeseries.csv", "person_daily_metabolic_totals.csv")
               if not (p / name).is_file()]
    if solutions:
        if not (p / "clustering_results.csv").is_file():
            missing.append("clustering_results.csv (Stage 05)")
        if len(solution_files(p)) < 2:
            missing.append("the stratified solutions of Stage 06 (*_clustering_results.csv in "
                           "Country_Analysis/, DayType_Analysis/, Country_DayType_Analysis/)")
    if missing:
        stop(f"The processed folder {p} does not hold a complete run of Stages 01 to 06. Missing:\n  "
             + "\n  ".join(missing) + f"\n{HINT}")


def household_days(processed):
    """Number of household-days of the run (rows of household_interval_timeseries.csv)."""
    with open(Path(processed) / "household_interval_timeseries.csv", "rb") as f:
        return sum(1 for _ in f) - 1


def describe(processed):
    """One line on the run in `processed`, to print before the results: the
    canonical run of 20 November 2026 has 191,740 household-days and 48
    solutions."""
    p = Path(processed)
    sols = solution_files(p)
    n = household_days(p) if (p / "household_interval_timeseries.csv").is_file() else 0
    return (f"Processed folder: {p}\n  {n:,} household-days, {len(sols)} solutions "
            f"({int(any(f[0].name == 'clustering_results.csv' for f in sols))} global, "
            f"{sum(f[0].name != 'clustering_results.csv' for f in sols)} stratified)")


def country_names(column):
    """Two-letter country names for a column of the codes of Stage 01 (positions
    in countries.COUNTRIES); a column of names is returned as text."""
    x = pd.to_numeric(column, errors="coerce")
    if x.notna().all() and len(x):
        codes = x.astype(int)
        return codes.map(NAME).fillna(codes.astype(str)).astype(str)
    return column.astype(str)

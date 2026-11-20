"""Check that the model reproduces the pipeline at the values of the
canonical run.

    python3 -m gsa.check --bundle FOLDER [--processed FOLDER] [--out FILE]

Runs the model once at BASELINE (gsa.model), with the K of every solution
chosen by the knee as in Stages 05 and 06 and n_init = 20 as in their final
fits, and compares it with the files of the run the bundle was built from:

  global partition         labels of clustering_results.csv (Stage 05)
  stratified partitions    K and labels of the 47 *_clustering_results.csv
                           files of Stage 06
  association tests        Cramer's V of consolidated_chi2_stats.csv (Stage
                           07), and which tests are published
  day types                peak_power_daytype_comparison.csv (Stage 06)
  mean profiles            mean_profiles_country_daytype.csv (Stage 06)
  peak times               circular SD of the global groups in
                           consolidated_circular_stats.csv (Stage 08)

The power of Stage 03 is checked when the bundle is built. The report holds
only aggregate differences and counts of matching solutions, so it can be
shared. It ends with the outputs Y1 to Y11 at the baseline.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsa.model import Model, BASELINE, OUTPUTS, SLOTS  # noqa: E402

TOL = 1e-9


def folders(processed):
    p = Path(processed)
    return {"global": p, "day": p / "DayType_Analysis", "country": p / "Country_Analysis",
            "country_day": p / "Country_DayType_Analysis"}


def solution_file(name, dirs):
    if name == "clustering_results.csv":
        return dirs["global"] / name
    if name.startswith(("Working_Days", "Non_Working_Days")):
        return dirs["day"] / name
    if "_working_days_" in name or "_non_working_days_" in name:
        return dirs["country_day"] / name
    return dirs["country"] / name


def compare(model, out, processed):
    """Lines of the report and the number of failed checks."""
    dirs = folders(processed)
    lines, failed = [], 0
    det = out["detail"]

    def report(ok, text):
        nonlocal failed
        failed += not ok
        lines.append(f"{'ok  ' if ok else 'FAIL'}  {text}")

    # partitions
    same_k = same_labels = n_sol = 0
    worst_ari, missing = 1.0, []
    for name, sol in det["solutions"].items():
        path = solution_file(name, dirs)
        if not path.is_file():
            missing.append(name)
            continue
        n_sol += 1
        ref = pd.read_csv(path, usecols=["cluster"])["cluster"].to_numpy()
        if len(ref) != len(sol["labels"]):
            report(False, f"{name}: {len(ref)} rows in the pipeline, {len(sol['labels'])} in the model")
            continue
        same_k += len(np.unique(ref)) == len(np.unique(sol["labels"]))
        same_labels += bool((ref == sol["labels"]).all())
        worst_ari = min(worst_ari, adjusted_rand_score(ref, sol["labels"]))
    report(not missing, f"solutions found in the run: {n_sol} of {len(det['solutions'])}"
           + (f" (missing: {', '.join(missing[:5])})" if missing else ""))
    report(same_labels == n_sol, f"identical labels in {same_labels} of {n_sol} solutions, same K in "
           f"{same_k}, smallest adjusted Rand index {worst_ari:.6f}")

    # association tests
    chi = pd.read_csv(dirs["global"] / "consolidated_chi2_stats.csv")
    names = model.feature_names
    diffs, pattern, compared = [], 0, 0
    for name, sol in det["solutions"].items():
        rows = chi[chi["file"] == name].set_index("Feature")["Cramers_V"]
        for j, f in enumerate(names):
            pipe = rows.get(f, np.nan)
            mine = sol["plain"][j]
            if np.isfinite(pipe) != np.isfinite(mine):
                pattern += 1
            elif np.isfinite(pipe):
                diffs.append(abs(pipe - mine))
                compared += 1
    worst = max(diffs) if diffs else 0.0
    report(pattern == 0 and worst < 1e-8,
           f"Cramer's V: {compared} published tests compared, largest difference {worst:.2e}, "
           f"{pattern} tests published by only one of the two")

    # day types
    cmp_path = dirs["country_day"] / "peak_power_daytype_comparison.csv"
    if cmp_path.is_file():
        cmp_df = pd.read_csv(cmp_path).set_index("country")
        mine = dict(zip([c for c in det["countries"] if c in cmp_df.index],
                        zip(det["coincident_change"], det["own_change"])))
        d1 = max(abs(cmp_df.loc[c, "mean_profile_peak_percent_difference"] - v[0]) for c, v in mine.items())
        d2 = max(abs(cmp_df.loc[c, "percent_difference"] - v[1]) for c, v in mine.items())
        report(len(mine) == len(cmp_df) and max(d1, d2) < 1e-6,
               f"day types: {len(mine)} countries, largest difference in the change of the coincident "
               f"peak {d1:.2e} and of the mean own peak {d2:.2e} percentage points")

    # mean profiles
    prof_path = dirs["country_day"] / "mean_profiles_country_daytype.csv"
    if prof_path.is_file():
        pr = pd.read_csv(prof_path)
        cols = [f"P_{t:03d}" for t in range(SLOTS)]
        rows = det["countries"] + ["ALL"]
        worst, n_cmp = 0.0, 0
        for i, c in enumerate(rows):
            for d, dt in enumerate(("working", "non_working", "all")):
                r = pr[(pr["country"] == c) & (pr["day_type"] == dt)]
                if r.empty:
                    continue
                n_cmp += 1
                rel = np.abs(r[cols].to_numpy(float)[0] - out["profiles"][i, d]) / np.maximum(
                    1.0, np.abs(out["profiles"][i, d]))
                worst = max(worst, float(np.nanmax(rel)))
        report(worst < 1e-9, f"mean profiles: {n_cmp} profiles, largest relative difference {worst:.2e}")

    # peak times of the global groups
    circ_path = dirs["global"] / "consolidated_circular_stats.csv"
    if circ_path.is_file():
        circ = pd.read_csv(circ_path)
        g = circ[circ["file"] == "clustering_results.csv"].set_index("cluster")["circular_std_rad"]
        sd_rad = np.asarray(det["group_sd_h"]) * 2 * np.pi / 24
        d = [abs(g.get(int(k), np.nan) - v) for k, v in zip(det["groups"], sd_rad)]
        report(len(g) == len(sd_rad) and np.nanmax(d) < 1e-9,
               f"peak times: {len(sd_rad)} global groups, largest difference in the circular SD "
               f"{np.nanmax(d):.2e} rad")
    return lines, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=None,
                        help="processed folder of the run (default: the one the bundle was built from)")
    parser.add_argument("--out", type=Path, default=None,
                        help="report to write (default: share/baseline_check.txt next to the bundle)")
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0, help="threads (0: all cores)")
    args = parser.parse_args()
    if args.threads:
        from threadpoolctl import threadpool_limits
        threadpool_limits(args.threads)
    t0 = time.time()
    model = Model(args.bundle, n_init=args.n_init)
    processed = args.processed or Path(model.meta["processed"])
    print(f"Running the model at the baseline on {model.n:,} household-days, with K chosen by the "
          f"knee in every solution (on HETUS 2010 this takes 10 to 30 minutes)", flush=True)
    done = []

    def progress(name, k):
        done.append(name)
        print(f"  {len(done):2d}. {name}: K = {k} ({time.time() - t0:.0f} s)", flush=True)

    out, prof, counts = model.run(BASELINE, detail=True, progress=progress)
    out["profiles"] = prof
    lines, failed = compare(model, out, processed)
    lines.append("")
    lines.append("Outputs at the baseline: " + ", ".join(
        f"{y} = {out[y]:.4g}" if isinstance(out[y], float) else f"{y} = {out[y]}" for y in OUTPUTS)
        + f"; strongest bias-corrected correlate {out['Y5_variable']}; global K {out['k_global']}; "
          f"{out['k_fallbacks']} of {out['solutions']} solutions with the fallback K")
    lines.append(f"Run in {time.time() - t0:.0f} s; {failed} check(s) failed.")
    text = "\n".join(lines)
    print(text)
    target = args.out or Path(args.bundle).resolve().parent / "share" / "baseline_check.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n")
    summary = {y: out[y] for y in OUTPUTS + ["Y5_variable", "k_global", "k_fallbacks", "solutions",
                                             "pca_components", "clusters_merged",
                                             "mean_daily_energy_wh"]}
    (target.with_suffix(".json")).write_text(json.dumps(summary, indent=1, default=float))
    print(f"Report written to {target}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

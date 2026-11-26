"""Run the checks for the reviewers on one run and collect their results.

One command instead of five. It finds the processed folder of the run (from
the bundle of the sensitivity analysis with --bundle, or given with
--processed), stops unless that folder holds a complete run of Stages 01 to
06, and runs on it

  tools/paired_day_types.py                   Reviewer 2, comment 5
  tools/paired_day_types.py --day-type ddv1   the same with the weekday rule
  tools/driver_robustness.py --files all      Reviewer 1, comment 5
  tools/clustering_robustness.py              Reviewer 1, comment 3

and then collects their tables and figures with tools/collect_exports.py
--checks, after the tests of pipeline/disclosure.py, into one folder to zip
and share:

    python3 tools/reviewer_checks.py --bundle ~/hetus_gsa/bundle
    python3 tools/reviewer_checks.py --processed "/path/to/processed"
                                     [--out exports_checks] [--quick]

A step that fails does not stop the others, and the summary at the end lists
every step and the folder to share. The figures go to the figures folder of
pipeline/paths.py (HETUS_FIGURES_DIR), where the collector looks for them.
--quick runs every step with few resamples, to try the command; its results
are not for the paper.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from run_folder import check, describe, resolve  # noqa: E402


def steps(quick=False):
    """(name, script and arguments) of every check, in the order they run."""
    return [
        ("paired days, day types of Stage 06",
         ["paired_day_types.py"] + (["--boot", "100"] if quick else [])),
        ("paired days, weekday rule",
         ["paired_day_types.py", "--day-type", "ddv1"] + (["--boot", "100"] if quick else [])),
        ("drivers, every solution",
         ["driver_robustness.py", "--files", "all"] + (["--boot", "20", "--draws", "5"] if quick else [])),
        ("clustering, stability and other methods",
         ["clustering_robustness.py"] + (["--reps", "2"] if quick else [])),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processed", type=Path, default=None,
                        help="processed folder of the run (in quotes if the path has spaces)")
    parser.add_argument("--bundle", type=Path, default=None,
                        help="bundle of the sensitivity analysis whose run to use (its meta.json)")
    parser.add_argument("--out", type=Path, default=Path("exports_checks"),
                        help="folder for the results to share (default: exports_checks)")
    parser.add_argument("--quick", action="store_true", help="few resamples, to try the command")
    args = parser.parse_args(argv)
    processed = Path(resolve(args.processed, args.bundle)).resolve()
    check(processed)
    print(describe(processed), flush=True)
    env = dict(os.environ, HETUS_PROCESSED_DIR=str(processed))
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    summary = []
    for name, cmd in steps(args.quick):
        print(f"\n=== {name}: python3 tools/{' '.join(cmd)}", flush=True)
        t0 = time.time()
        done = subprocess.run([sys.executable, str(ROOT / "tools" / cmd[0])] + cmd[1:], env=env)
        summary.append((name, done.returncode, time.time() - t0))
    print("\n=== collecting the results", flush=True)
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "collect_exports.py"), "--checks",
                           "--out", str(out)], env=env)
    summary.append(("collection", done.returncode, 0.0))
    print("\nSummary")
    for name, code, secs in summary:
        state = "done" if code == 0 else f"FAILED (status {code}; its message is above)"
        took = "" if not secs else (f", {secs / 60:.0f} min" if secs >= 90 else ", under 2 min")
        print(f"  {name}: {state}{took}")
    if all(code == 0 for _, code, _ in summary):
        print(f"\nZip and send {out}")
        return 0
    print(f"\nSome steps failed. Send the messages above with {out} (it holds only what passed).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

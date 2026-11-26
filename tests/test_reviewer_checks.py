"""The checks for the reviewers (tools/reviewer_checks.py and the four scripts
it runs) on a synthetic run of Stages 01 to 07 in which most households keep
diaries on a weekday and a weekend day, run as a user runs them:

  every solution of Stages 05 and 06 is matched to its household-days (the
  country files name the countries, household_interval_timeseries.csv codes
  them), the paired comparison finds the paired households of every country
  and names the countries, and the collector copies the four tables and the
  figures after its tests;
  a processed folder without the solutions stops every script with a message
  and no traceback, and nothing empty is written;
  --bundle reads the processed folder from the bundle's meta.json;
  the collector refuses an empty table instead of failing.
"""
import importlib.util
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from conftest import ROOT, run_stage
from synthetic import write_raw_two_days

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kneed") is None,
                                reason="kneed is not installed")
STAGES = ["01_calculate_diarist_metrics.py", "02_calculate_dominant_activity.py",
          "03_aggregate_household.py", "04_prepare_clustering_data.py",
          "05_clustering_analysis_global.py", "06_clustering_country_day.py",
          "07_statistical_validation.py"]
TOOLS = ROOT / "tools"


def tool(script, *args, env=None, cwd=None):
    e = dict(os.environ, MPLBACKEND="Agg", **(env or {}))
    return subprocess.run([sys.executable, str(TOOLS / script), *map(str, args)], env=e, cwd=cwd,
                          capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def two_day_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("checks")
    dirs = {"HETUS_DATA_DIR": base / "raw", "HETUS_PROCESSED_DIR": base / "processed",
            "HETUS_FIGURES_DIR": base / "figures"}
    for d in dirs.values():
        d.mkdir()
    write_raw_two_days(dirs["HETUS_DATA_DIR"])
    for script in STAGES:
        done = run_stage(script, dirs, {"HETUS_COUNTRIES": "BE,FR,NL", "MPLBACKEND": "Agg"})
        assert done.returncode == 0, script + done.stdout[-2000:] + done.stderr[-2000:]
    return base, dirs


@pytest.fixture(scope="module")
def checks(two_day_run):
    base, dirs = two_day_run
    out = base / "exports_checks"
    done = tool("reviewer_checks.py", "--processed", dirs["HETUS_PROCESSED_DIR"], "--out", out, "--quick",
                env={"HETUS_FIGURES_DIR": str(dirs["HETUS_FIGURES_DIR"]), "HETUS_PROCESSED_DIR": ""})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return base, dirs, out, done


def test_every_solution_matched(checks):
    _, dirs, out, done = checks
    sys.path.insert(0, str(TOOLS))
    from run_folder import solution_files
    files = {f[0].name for f in solution_files(dirs["HETUS_PROCESSED_DIR"])}
    table = pd.read_csv(out / "driver_robustness.csv")
    assert len(files) >= 10 and set(table.file) == files
    assert "SKIPPED" not in done.stdout
    assert (table.Feature == "N_DIARISTS").sum() == len(files)


def test_paired_comparison_names_the_countries(checks):
    _, _, out, _ = checks
    for name in ("paired_daytype_comparison.csv", "paired_daytype_comparison_ddv1.csv"):
        table = pd.read_csv(out / name).set_index("country")
        assert set(table.index) == {"BE", "FR", "NL", "ALL"}
        assert table.n_paired_households.notna().all()
        assert (table.n_paired_households % 10 == 0).all()
        assert table.paired_coincident_change_lo.le(table.paired_coincident_change_hi).all()


def test_collected_for_sharing(checks):
    _, _, out, done = checks
    manifest = (out / "manifest.txt").read_text()
    assert "REFUSED" not in manifest and "MISSING" not in manifest
    for name in ("paired_daytype_comparison.csv", "paired_daytype_comparison_ddv1.csv",
                 "driver_robustness.csv", "clustering_robustness.csv", "paired_daytype_comparison.png"):
        assert (out / name).is_file(), name
    table = pd.read_csv(out / "clustering_robustness.csv")
    assert (table.solution == "clustering_results.csv").sum() == 4      # half-samples and three methods
    assert (table.solution != "clustering_results.csv").sum() >= 9
    assert "Zip and send" in done.stdout


def test_bundle_gives_the_folder(two_day_run, tmp_path):
    _, dirs = two_day_run
    (tmp_path / "meta.json").write_text(json.dumps({"processed": str(dirs["HETUS_PROCESSED_DIR"])}))
    sys.path.insert(0, str(TOOLS))
    from run_folder import resolve
    assert resolve(bundle=tmp_path) == dirs["HETUS_PROCESSED_DIR"].resolve()
    done = tool("reviewer_checks.py", "--bundle", tmp_path / "nowhere")
    assert done.returncode == 2 and "meta.json not found" in done.stderr


def test_folder_without_solutions_stops_cleanly(two_day_run, tmp_path):
    _, dirs = two_day_run
    partial = tmp_path / "processed"
    partial.mkdir()
    for name in ("household_interval_timeseries.csv", "person_daily_metabolic_totals.csv"):
        (partial / name).write_bytes((dirs["HETUS_PROCESSED_DIR"] / name).read_bytes())
    for script, args in (("driver_robustness.py", ["--files", "all"]), ("clustering_robustness.py", []),
                         ("reviewer_checks.py", ["--out", tmp_path / "out"])):
        done = tool(script, "--processed", partial, *args)
        assert done.returncode == 2, script
        assert "does not hold a complete run" in done.stderr and "Traceback" not in done.stderr, script
    assert not (partial / "driver_robustness.csv").exists()
    assert not (partial / "clustering_robustness.csv").exists()
    # the paired comparison needs only Stage 03 and still runs
    done = tool("paired_day_types.py", "--processed", partial, "--boot", 50,
                env={"HETUS_FIGURES_DIR": str(tmp_path / "figures")})
    assert done.returncode == 0, done.stderr[-2000:]


def test_collector_refuses_an_empty_table(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "clustering_robustness.csv").write_text("\n")
    done = tool("collect_exports.py", "--checks", "--out", tmp_path / "out",
                env={"HETUS_PROCESSED_DIR": str(processed), "HETUS_FIGURES_DIR": str(tmp_path / "fig")})
    assert "Traceback" not in done.stderr
    assert "REFUSED  clustering_robustness.csv: empty or unreadable file" in done.stdout
    assert done.returncode == 1

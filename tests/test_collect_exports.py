"""tools/collect_exports.py copies only the exports that pass the publication
rules and never the working files."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from conftest import ROOT


def run_collector(dirs, out, log=None):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in dirs.items()})
    cmd = [sys.executable, str(ROOT / "tools" / "collect_exports.py"), "--out", str(out)]
    if log:
        cmd += ["--log", str(log)]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def test_only_publishable_exports_are_copied(stage_dirs, tmp_path):
    proc = stage_dirs["HETUS_PROCESSED_DIR"]
    rng = np.random.default_rng(0)
    # a summary that may be shared, and mean profiles that reveal a group of 37
    pd.DataFrame({"cluster": [0, 1], "size": [400, 120]}).to_csv(proc / "cluster_summary.csv",
                                                                  index=False)
    rows = []
    for g, n in enumerate([400, 37]):
        e = rng.integers(0, 3000, size=(n, 144))
        row = {"cluster": g, "n": n if n >= 50 else np.nan}
        row.update({f"P_{t:03d}": e[:, t].mean() for t in range(144)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(proc / "cluster_mean_profiles.csv", index=False)
    # a working file with one row per household-day, never collected
    pd.DataFrame({"HID": ["BE_1"], "cluster": [0]}).to_csv(proc / "clustering_results.csv",
                                                          index=False)
    log = tmp_path / "run_log.txt"
    log.write_text("PCA: 144 columns reduced to 91 components\n"
                   "Cluster sizes:\n0    111428\n7    50\n"
                   "  Optimal number of clusters: 4 (no knee detected, fallback value)\n")
    out = tmp_path / "share"
    done = run_collector(stage_dirs, out, log)
    assert done.returncode == 1                      # one table refused
    copied = sorted(p.name for p in out.iterdir())
    assert "cluster_summary.csv" in copied
    assert "cluster_mean_profiles.csv" not in copied
    assert "clustering_results.csv" not in copied
    manifest = (out / "manifest.txt").read_text()
    assert "REFUSED  cluster_mean_profiles.csv" in manifest and "group of 37" in manifest
    extract = (out / "log_extract.txt").read_text().splitlines()
    assert extract == ["PCA: 144 columns reduced to 91 components",
                       "  Optimal number of clusters: 4 (no knee detected, fallback value)"]

"""Stage 10 prints and draws the numbers of diarists per household-day under
the publication rules: the largest numbers are pooled into one class of at
least 50 household-days, and no minimum or maximum is given (November 2026)."""
import re

import pandas as pd

from conftest import run_stage


def test_stage10_pools_the_tail_and_gives_no_extremes(tmp_path):
    dirs = {"HETUS_DATA_DIR": tmp_path / "raw", "HETUS_PROCESSED_DIR": tmp_path / "processed",
            "HETUS_FIGURES_DIR": tmp_path / "figures"}
    for d in dirs.values():
        d.mkdir()
    rows, h = [], 0
    for country, sizes in (("BE", {1: 200, 2: 120, 3: 60, 4: 30, 5: 10, 9: 2}),
                           ("FR", {1: 150, 2: 80})):
        for size, n in sizes.items():
            for _ in range(n):
                h += 1
                rows += [{"HID": f"{country}_{h}", "PID": f"{country}_{p}", "DIARY": 1}
                         for p in range(1, size + 1)]
    pd.DataFrame(rows).to_csv(dirs["HETUS_PROCESSED_DIR"] / "person_daily_metabolic_totals.csv",
                              index=False)
    done = run_stage("10_p_h_counting.py", dirs, {"MPLBACKEND": "Agg"})
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    out = done.stdout
    assert "Maximum" not in out and "Minimum" not in out and "_max" not in out
    breakdown = re.findall(r"^\s+(\S+) diarist\(s\): (\d+) household-days", out, re.M)
    assert [label for label, _ in breakdown] == ["1", "2", "3+"]
    assert all(int(n) >= 50 for _, n in breakdown)
    assert dict(breakdown)["3+"] == "102"
    for name in ("diarists_per_household_distribution.png", "diarists_per_household_detailed.png"):
        assert (dirs["HETUS_FIGURES_DIR"] / name).is_file()

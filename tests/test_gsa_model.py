"""The data bundle and the model of the sensitivity analysis on a synthetic run
of Stages 01 to 08 (three countries, households of one to three diarists):

  the bundle reproduces the power of Stage 03 (checked when it is built);
  at the values of the canonical run the model reproduces Stages 05 to 08
  (labels of all solutions, Cramer's V, day types, mean profiles, peak times);
  each rule of the reconstruction against a computation of its own from the
  diarist records: every diarist counted (X9), television counted once,
  the first-listed diarist only (X13), secondary codes left out (X11),
  ownership (X10), laundry run-on (X20), lighting in darkness (X19), base load
  (X12) and country multipliers (X7, X8);
  the outputs: published groups, peak times, the tests and the weights;
  the runner (resume, settings) and the collected files.
"""
import importlib
import importlib.util
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from conftest import ROOT, run_stage
from synthetic import write_raw_gsa

sys.path.insert(0, str(ROOT))
from gsa import bundle, check, collect, design, run as gsa_run, sun  # noqa: E402
from gsa.model import (Model, BASELINE, OUTPUTS, bias_corrected_v, chi2_table,  # noqa: E402
                       circular_sd_hours, group_peak_weights, option_a)
from countries import COUNTRIES  # noqa: E402
import disclosure  # noqa: E402
import peaks  # noqa: E402

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kneed") is None,
                                reason="kneed is not installed")
s03 = importlib.import_module("03_aggregate_household")
MAPPING = pd.read_csv(ROOT / "data" / "HETUS_mapped_data.csv")
POWER = MAPPING.set_index("activity")["Appliance_Power_Watts"].to_dict()
STAGES = ["01_calculate_diarist_metrics.py", "02_calculate_dominant_activity.py",
          "03_aggregate_household.py", "04_prepare_clustering_data.py",
          "05_clustering_analysis_global.py", "06_clustering_country_day.py",
          "07_statistical_validation.py", "08_circular_stats_peak_time.py"]


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("gsa")
    dirs = {"HETUS_DATA_DIR": base / "raw", "HETUS_PROCESSED_DIR": base / "processed",
            "HETUS_FIGURES_DIR": base / "figures"}
    for d in dirs.values():
        d.mkdir()
    write_raw_gsa(dirs["HETUS_DATA_DIR"])
    for script in STAGES:
        done = run_stage(script, dirs, {"HETUS_COUNTRIES": "BE,FR,NL", "MPLBACKEND": "Agg"})
        assert done.returncode == 0, script + done.stdout[-2000:] + done.stderr[-2000:]
        (base / f"log_{script[:2]}.txt").write_text(done.stdout)
    info = bundle.build(base / "bundle", processed=dirs["HETUS_PROCESSED_DIR"],
                        raw_dir=dirs["HETUS_DATA_DIR"], mapping=ROOT / "data" / "HETUS_mapped_data.csv")
    return base, dirs, info


@pytest.fixture(scope="module")
def model(synthetic_run):
    return Model(synthetic_run[0] / "bundle", n_init=20)


@pytest.fixture(scope="module")
def records(synthetic_run):
    """Diarist records sorted by household-day as in Stage 03, with their
    codes, locations and power series."""
    processed = synthetic_run[1]["HETUS_PROCESSED_DIR"]
    d = pd.read_csv(processed / s03.INPUT.name)
    g = s03.group_records(d[["HID", "PID", "DIARY"]])
    order = np.argsort(g, kind="stable")
    d, g = d.iloc[order].reset_index(drop=True), g[order]
    energy, _ = s03.parse_series(d["Diarist_Energy_Demand_Series"])
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    return {"meta": d, "g": g, "starts": starts, "n": int(g.max()) + 1,
            "M": s03.as_codes(d[s03.MACT]), "S": s03.as_codes(d[s03.SACT]),
            "W": d[s03.WHERE].to_numpy() == 1, "E": np.nan_to_num(energy)}


def by_day(rec, values, how="sum"):
    """Values of the records (n_records x 144) summed, or OR-ed, by household-day."""
    out = np.zeros((rec["n"], 144), dtype=bool if how == "any" else float)
    (np.logical_or if how == "any" else np.add).at(out, rec["g"], values)
    return out


def at(x=None, **values):
    """BASELINE with some factors changed, and K fixed for speed."""
    v = dict(BASELINE if x is None else x)
    v.update({"X16": 8, "X17": 4})
    v.update(values)
    return v


# --------------------------------------------------------------- the bundle
def test_baseline_powers_are_those_of_the_mapping():
    for fid in ("X1", "X2", "X3", "X4", "X5", "X6"):
        for code, w in BASELINE[fid].items():
            assert POWER[code] == w, code
    with_power = {int(c) for c, w in POWER.items() if w > 0}
    assert with_power == {c for codes in design.POWER.values() for c in codes}
    assert sorted(design.FIVE_WATT) == sorted(c for c, w in POWER.items() if w == 5 and c != 831)


def test_light_use_of_outdoor_activities():
    # Outdoor activities need no light; 342 and 343 were swapped until
    # November 2026 (docs/PIPELINE.md Section 3.17, K7)
    light = MAPPING.set_index("activity")["Light_use"]
    assert (light[[322, 341, 342, 344, 349]] == 0).all() and light[343] == 1
    assert int((light == 1).sum()) == 81


def test_bundle_reproduces_stage03(synthetic_run):
    info = synthetic_run[2]
    assert info["power_check"]["differing"] == 0 and info["entries"] > 0
    assert info["household_days"] == 390 and info["reference_found"]


def test_bundle_in_blocks(synthetic_run, monkeypatch, tmp_path):
    base, dirs, _ = synthetic_run
    monkeypatch.setattr(bundle, "BLOCK", 37)
    bundle.build(tmp_path / "b", processed=dirs["HETUS_PROCESSED_DIR"], raw_dir=dirs["HETUS_DATA_DIR"],
                 mapping=ROOT / "data" / "HETUS_mapped_data.csv")
    for group in bundle.GROUPS:
        one, other = bundle.load_group(base / "bundle", group), bundle.load_group(tmp_path / "b", group)
        assert one.keys() == other.keys()
        for name in one:
            assert np.array_equal(one[name], other[name]), (group, name)


def test_baseline_reproduces_the_pipeline(model, synthetic_run):
    out, prof, _ = model.run(BASELINE, detail=True)
    out["profiles"] = prof
    lines, failed = check.compare(model, out, synthetic_run[1]["HETUS_PROCESSED_DIR"])
    assert failed == 0, "\n".join(lines)
    assert out["Y8"] == 1.0 and out["solutions"] == 12


def test_stage05_console_follows_the_publication_rules(synthetic_run):
    """The synthetic global solution has clusters of fewer than 50
    household-days, so the console of Stage 05 must report only published
    groups, print no table of shares by cluster and category, and print a share
    only where neither the count behind it nor the rest of its group is below
    20 (the run of 20 November 2026 printed both kinds)."""
    out = (synthetic_run[0] / "log_05.txt").read_text()
    assert "reported together with the largest" in out
    assert "Distribution across clusters" not in out and "Category " not in out
    total = synthetic_run[2]["household_days"]
    sizes = [int(n) for n in re.findall(r"^Size: (\d+) households", out, re.M)]
    assert sizes and min(sizes) >= 50 and sum(sizes) == total
    block = out.split("Sizes of the published groups (household-days):")[1].split("\n\n")[0]
    assert all(int(line.split()[-1]) >= 50 for line in block.strip().splitlines())
    size = None
    for line in out.splitlines():
        m = re.match(r"Size: (\d+) households", line)
        if m:
            size = int(m.group(1))
        m = re.search(r"\(([\d.]+)% in cluster vs ([\d.]+)% overall\)", line)
        if m:
            for share, n in ((float(m.group(1)), size), (float(m.group(2)), total)):
                count = round(share / 100 * n)
                assert count >= 20 and (n - count == 0 or n - count >= 20), line


# --------------------------------------------------------- the power rules
def test_every_diarist_counted(model, records):
    p = model.profiles(at(X9="per_person"))
    assert np.array_equal(p, by_day(records, records["E"]))


def test_first_listed_diarist(model, records):
    p = model.profiles(at(X13="first_listed"))
    assert np.array_equal(p, records["E"][records["starts"]])


def test_television_counted_once(model, records):
    M, S, W = records["M"], records["S"], records["W"]
    users = by_day(records, ((M == 821) | (S == 821)) & W)
    expected = by_day(records, records["E"]) - POWER[821] * np.maximum(users - 1, 0)
    assert np.allclose(model.profiles(at(X9="tv_once")), expected)


def test_secondary_codes_left_out(model, records):
    M, W = records["M"], records["W"]
    expected = np.zeros((records["n"], 144))
    for code in model.codes:
        expected += POWER[int(code)] * by_day(records, (M == code) & W, "any")
    assert np.allclose(model.profiles(at(X11=0.0)), expected)


def test_ownership(model, records, synthetic_run):
    M, S, W = records["M"], records["S"], records["W"]
    h = bundle.load_group(synthetic_run[0] / "bundle", "households")
    p0 = model.profiles(at())
    expected = p0.copy()
    for item, lack, codes in (("hhq6c", 0, [821]), ("hhq6h", 2, [331]),
                              ("hhq6o", 2, [721, 722, 723, 729, 733])):
        lacks = np.asarray(h[item]) == lack
        for c in codes:
            expected[lacks] -= POWER[c] * by_day(records, ((M == c) | (S == c)) & W, "any")[lacks]
    assert (np.asarray(h["hhq6c"]) == 0).any() and (np.asarray(h["hhq6c"]) < 0).any()
    assert np.allclose(model.profiles(at(X10="applied")), expected)


def runon_reference(records, x11):
    """Run-on weights from the records: slots since laundry was last in use
    at home, and v = 1 or X11 by whether it was a main activity then."""
    M, S, W = records["M"], records["S"], records["W"]
    use = by_day(records, ((M == 331) | (S == 331)) & W, "any")
    main = by_day(records, (M == 331) & W, "any")
    lut = np.array([0, 1, 1, 0.5] + [0] * 9)            # X20 = 25 minutes
    out = np.zeros(use.shape)
    for h in range(use.shape[0]):
        last = -1
        for t in range(144):
            if use[h, t]:
                last = t
            elif last >= 0 and t - last <= 12:
                out[h, t] = lut[t - last] * (1.0 if main[h, last] else x11)
    return out


def test_laundry_run_on(model, records):
    for x11 in (1.0, 0.5):
        extra = model.profiles(at(X20=25.0, X11=x11)) - model.profiles(at(X11=x11))
        expected = POWER[331] * runon_reference(records, x11)
        assert expected.sum() > 0 and np.allclose(extra, expected)


def test_lighting_in_darkness(model, records, synthetic_run):
    raw_dir = synthetic_run[1]["HETUS_DATA_DIR"]
    first = records["meta"].iloc[records["starts"]]
    raw = pd.concat([bundle.read_raw(c, raw_dir) for c in ("BE", "FR", "NL")])
    raw = raw.set_index(["HID", "PID", "DIARY"])
    keys = list(zip(first["HID"], first["PID"], bundle.diary_key(first["DIARY"])))
    month, year = raw.loc[keys, "DDV3"].to_numpy(), raw.loc[keys, "DDV4"].to_numpy()
    country = first["COUNTRY"].to_numpy().astype(int)
    dark = np.array([sun.darkness(COUNTRIES[c], y, m) for c, y, m in zip(country, year, month)])
    light = MAPPING.loc[MAPPING["Light_use"] == 1, "activity"].to_numpy()
    M, S, W = records["M"], records["S"], records["W"]
    lit = by_day(records, (np.isin(M, light) | np.isin(S, light)) & W, "any")
    extra = model.profiles(at(X19=100.0)) - model.profiles(at())
    assert np.allclose(extra, 100.0 * dark * lit, rtol=0, atol=1e-9)


def test_base_load_moves_no_peak(model):
    p0, p1 = model.profiles(at()), model.profiles(at(X12=40.0))
    assert np.allclose(p1 - p0, 40.0)
    y0, _, _ = model.run(at())
    y1, _, _ = model.run(at(X12=40.0))
    for y in ("Y1", "Y2", "Y2b", "Y5", "Y6", "Y8", "Y9", "Y10", "Y11"):
        assert y0[y] == pytest.approx(y1[y], abs=1e-12), y
    assert y1["Y7"] != pytest.approx(y0["Y7"])


def test_country_multipliers(model):
    m = model.multipliers(0.3, 5)
    present = np.array(model.countries)
    assert np.exp(np.log(m[present]).mean()) == pytest.approx(1.0)
    p0, p1 = model.profiles(at()), model.profiles(at(X7=0.3, X8=5))
    on = p0 > 0
    ratio = np.where(on, p1 / np.where(on, p0, 1), np.nan)
    for c in model.countries:
        assert np.nanmax(np.abs(ratio[model.country == c] - m[c])) < 1e-12


# --------------------------------------------------------------- the outputs
def test_option_a_as_disclosure():
    rng = np.random.default_rng(3)
    labels = rng.choice(6, size=400, p=[0.4, 0.3, 0.2, 0.06, 0.03, 0.01])
    groups, merged = option_a(labels)
    expected, small, _ = disclosure.published_groups(labels)
    assert np.array_equal(groups, expected) and merged == len(small) > 0


def test_peak_times_as_stage08():
    rng = np.random.default_rng(4)
    e = np.round(rng.gamma(1.0, 100.0, (300, 144)) * (rng.random((300, 144)) < 0.3))
    e[:20] = 0                                            # no demand
    e[20:40, 50:53] = e[20:40, 50:53].max() + 500         # ties
    groups = rng.integers(0, 3, 300)
    slot_w, n_days = group_peak_weights(e, 0.0, groups, 3, np.ones(300))
    sd = circular_sd_hours(slot_w, n_days)
    for g in range(3):
        _, sd_rad, _ = peaks.circular_peak_stats(e[groups == g])
        assert sd[g] == pytest.approx(sd_rad * 24 / (2 * np.pi), rel=1e-12)
    slot_w2, _ = group_peak_weights(e + 30.0, 30.0, groups, 3, np.ones(300))
    assert np.allclose(slot_w, slot_w2)                   # the base load moves no peak


def test_tests_as_stage07():
    rng = np.random.default_rng(5)
    table = rng.integers(0, 40, (5, 4)).astype(float)
    table[2] = 0
    chi2, n, shape = chi2_table(table)
    ref = chi2_contingency(table[table.sum(axis=1) > 0], correction=False)[0]
    assert chi2 == pytest.approx(ref) and shape == (4, 4)
    r, k = shape
    phi2 = max(0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    assert bias_corrected_v(chi2, n, r, k) == pytest.approx(np.sqrt(phi2 / min(rc - 1, kc - 1)))


def test_weights_and_weighted_run(model, synthetic_run):
    h = bundle.load_group(synthetic_run[0] / "bundle", "households")
    for w in ("weight_first", "weight_mean"):
        for c in model.countries:
            assert np.asarray(h[w])[model.country == c].mean() == pytest.approx(1.0)
    y0, _, _ = model.run(at())
    y1, _, _ = model.run(at(X15="mean", X14="ddv1_only"))
    assert all(np.isfinite(y1[y]) for y in OUTPUTS)
    assert y1["Y3"] != pytest.approx(y0["Y3"])


# ------------------------------------------------------ the runner and collect
def test_runner_resume_and_collect(synthetic_run, capsys, monkeypatch):
    base = synthetic_run[0]
    runs = base / "runs"
    args = ["--bundle", str(base / "bundle"), "--runs", str(runs), "--m", "2", "--n-init", "3"]
    gsa_run.main(args + ["--first", "2"])
    gsa_run.main(args)
    assert "2 of 4 runs already done" in capsys.readouterr().out
    gsa_run.main(args)
    assert "4 of 4 runs already done" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        gsa_run.main(args[:-1] + ["5"])                   # other n_init: refused
    real = gsa_run.library_versions()
    with monkeypatch.context() as mp:                     # another scikit-learn: refused
        mp.setattr(gsa_run, "library_versions", lambda: {**real, "scikit-learn": "0.0"})
        with pytest.raises(SystemExit):
            gsa_run.main(args)
    dpath = runs / "design.json"                          # a folder begun without versions
    d = json.loads(dpath.read_text())
    d.pop("versions")
    dpath.write_text(json.dumps(d))
    gsa_run.main(args)
    assert json.loads(dpath.read_text())["versions"] == real
    assert json.loads((runs / "run_00000.json").read_text())["versions"] == real
    rec = json.loads((runs / "run_00003.json").read_text())
    assert rec["coordinates"] == pytest.approx(design.sample(2, seed=gsa_run.SAMPLE_SEED)[3].tolist())
    table, profiles = collect.collect(runs, base / "share")
    assert list(table["run"]) == [0, 1, 2, 3] and profiles.shape == (4, 4, 3, 144)
    assert set(OUTPUTS) <= set(table.columns) and np.isfinite(table[OUTPUTS].to_numpy(float)).all()
    shared = pd.read_csv(base / "share" / "gsa_outputs.csv")
    assert len(shared) == 4 and (base / "share" / "manifest.txt").is_file()
    manifest = (base / "share" / "manifest.txt").read_text()
    assert "in 4 runs" in manifest and "WARNING" not in manifest


def test_reseed_reruns(synthetic_run, capsys, monkeypatch):
    from gsa import reseed, analyse
    base = synthetic_run[0]
    runs = base / "runs"
    gsa_run.main(["--bundle", str(base / "bundle"), "--runs", str(runs), "--m", "2", "--n-init", "3"])
    out = base / "reseed"
    common = ["--bundle", str(base / "bundle"), "--base", str(runs), "--out", str(out)]
    reseed.main(common + ["--which", "same", "--list", "0,1"])
    assert "Reruns identical to the base runs" in capsys.readouterr().out
    reseed.main(common + ["--which", "kmeans", "--points", "2"])
    reseed.main(common + ["--which", "seeds", "--points", "2"])
    reseed.main(common + ["--which", "kmeans", "--points", "2"])
    assert "2 of 2 points already done" in capsys.readouterr().out
    cols = {c: design.COLUMNS.index(c) for c in ("X8", "X21")}
    for which, redrawn in (("kmeans", {"X21"}), ("seeds", {"X8", "X21"})):
        for i in (0, 1):
            a = np.array(json.loads((runs / f"run_{i:05d}.json").read_text())["coordinates"])
            b = np.array(json.loads((out / which / f"run_{i:05d}.json").read_text())["coordinates"])
            changed = {c for c in design.COLUMNS if a[design.COLUMNS.index(c)] != b[design.COLUMNS.index(c)]}
            assert changed == redrawn, (which, i, changed)
            assert all(b[j] != a[j] for c, j in cols.items() if c in redrawn)
    real = gsa_run.library_versions()
    with monkeypatch.context() as mp:                     # another scikit-learn: refused
        mp.setattr(gsa_run, "library_versions", lambda: {**real, "scikit-learn": "0.0"})
        with pytest.raises(SystemExit):
            reseed.main(common + ["--which", "seeds", "--points", "2"])
    with pytest.raises(SystemExit):                       # a point outside the sample
        reseed.main(common + ["--which", "seeds", "--list", "7"])
    table = collect.collect(runs, base / "share_base")[0]
    totals = {}
    for which in ("same", "kmeans", "seeds"):
        t, _ = collect.collect(out / which, base / "reseed_share" / which)
        assert list(t["run"]) == ([0, 1] if which != "same" else [0, 1])
        assert "Reruns of points of the sample" in (base / "reseed_share" / which / "manifest.txt").read_text()
        totals[which] = analyse.seed_totals(table, t).set_index("output")
    assert (totals["same"]["T"] == 0).all()               # the model is deterministic
    profile_outputs = ["Y3", "Y4", "Y7", "Y9", "Y10", "Y11"]   # no clustering in them
    assert (totals["kmeans"].loc[profile_outputs, "T"] == 0).all()

"""The vectorised Stage 03 reproduces the loop of versions up to 1.1.0
exactly, on households of two to nine diarists with empty, negative and
unmapped secondary codes and empty locations, and its activity matrix gives
the household power as p = A w."""
import importlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, run_stage
from synthetic import write_raw

stage03 = importlib.import_module("03_aggregate_household")


@pytest.fixture(scope="module")
def both_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("stage03")
    dirs = {"HETUS_DATA_DIR": base / "raw", "HETUS_PROCESSED_DIR": base / "processed",
            "HETUS_FIGURES_DIR": base / "figures"}
    for d in dirs.values():
        d.mkdir()
    write_raw(dirs["HETUS_DATA_DIR"], countries=("FR",), n_households=40, seed=7,
              sizes=(2, 3, 5, 9), messy=True)
    env = {"HETUS_COUNTRIES": "FR"}
    for script in ("01_calculate_diarist_metrics.py", "02_calculate_dominant_activity.py",
                   "aggregate_household_loop_reference.py"):
        done = run_stage(script, dirs, env)
        assert done.returncode == 0, done.stderr[-2000:]
    done = run_stage("03_aggregate_household.py", dirs,
                     dict(env, HETUS_WRITE_LONG="1", HETUS_SAVE_ACTIVITY_MATRIX="1"))
    assert done.returncode == 0, done.stderr[-2000:]
    return dirs


def test_identical_to_the_loop(both_runs):
    proc = both_runs["HETUS_PROCESSED_DIR"]
    env = dict(os.environ, **{k: str(v) for k, v in both_runs.items()})
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "compare_stage03.py"),
                           "--reference", str(proc / "stage03_loop_reference"), "--long"],
                          env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
    assert "Identical." in done.stdout
    # households with eight or more diarists at home in some slot were tested
    long = pd.read_csv(proc / "household_interval_timeseries_long.csv")
    assert long.n_diarists_at_home.max() >= 8


def test_activity_matrix_gives_the_power(both_runs):
    from scipy import sparse
    proc = both_runs["HETUS_PROCESSED_DIR"]
    z = np.load(proc / "activity_matrix.npz")
    shape = tuple(z["shape"])
    A = {name: sparse.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]),
                                 shape=shape) for name in ("union", "main", "count")}
    wide = pd.read_csv(proc / "household_interval_timeseries.csv")
    power = wide[[f"Household_Energy_W_{t}" for t in range(144)]].to_numpy()
    assert np.array_equal((A["union"] @ z["power"]).reshape(power.shape), power)
    # used as a main activity, or counted per diarist: never outside the union
    assert (A["main"] - A["main"].multiply(A["union"])).nnz == 0
    assert np.array_equal((A["count"] > 0).toarray(), A["union"].toarray() > 0)
    days = pd.read_csv(proc / "activity_matrix_households.csv")
    assert days[["HID", "DIARY"]].equals(wide[["HID", "DIARY"]])


def loop_parse(s):
    """parse_list_string_custom of the loop version, verbatim."""
    if pd.isna(s):
        return [np.nan] * 144
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        s = s.strip("[]")
        elements = s.split(", ")
        parsed_list = []
        for e in elements:
            e = e.strip()
            if e == "nan" or e == "np.nan" or e == "":
                parsed_list.append(np.nan)
            else:
                try:
                    parsed_list.append(float(e))
                except ValueError:
                    parsed_list.append(np.nan)
        return parsed_list
    return [np.nan] * 144


def test_parse_series_like_the_loop():
    texts = ["[1.5, nan, np.nan, , 2" + ", 0" * 139 + "]", None, "[]",
             "[3.0, x" + ", 1" * 142 + "]", "[1e-3" + ", 2.25" * 150 + "]",
             "[74.33426583" + ", 0" * 143 + "]"]
    values, ok = stage03.parse_series(pd.Series(texts))
    for i, text in enumerate(texts):
        expect = loop_parse(text)
        assert ok[i] == (len(expect) == 144)
        padded = (expect + [np.nan] * 144)[:144]
        assert np.array_equal(values[i], np.array(padded, dtype=float), equal_nan=True), i


def test_metabolic_sum_in_the_order_of_pandas():
    rng = np.random.default_rng(3)
    for n in range(1, 13):
        terms = rng.uniform(30, 400, size=n) * rng.choice([1, 1e-3, 1e3], size=n)
        key = np.zeros(n, dtype=np.int64)
        assert stage03.metabolic_sum(key, terms, 1)[0] == pd.Series(terms).sum(), n

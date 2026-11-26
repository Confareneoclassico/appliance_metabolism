"""tools/pca_summary.py on profiles with a known structure."""
import importlib.util
import sys

import numpy as np

from conftest import ROOT

spec = importlib.util.spec_from_file_location("pca_summary", ROOT / "tools" / "pca_summary.py")
ps = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "tools"))
spec.loader.exec_module(ps)


def test_window_and_clock():
    load = np.zeros(144)
    load[45:52] = [0.3, 0.6, 0.9, 1.0, 0.8, 0.55, 0.2]
    k, lo, hi = ps.window(-load)                      # sign does not matter
    assert (k, lo, hi) == (48, 46, 50)
    assert ps.clock(0) == "04:00" and ps.clock(48) == "12:00" and ps.clock(143) == "03:50"


def test_leading_component_finds_the_shared_window():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 144))
    common = rng.normal(size=(n, 1)) * 3.0
    X[:, 90:96] += common                                 # 19:00 to 19:50 vary together
    need, comps = ps.summary(X, n=2)
    assert comps.iloc[0].largest in {ps.clock(t) for t in range(90, 96)}
    assert comps.iloc[0]["from"] == "19:00" and comps.iloc[0].to == "19:50"
    assert comps.iloc[0].share_pct > 3 > 2 * comps.iloc[1].share_pct          # 6 x 0.9 / 144 = 3.75%
    assert need[80] < need[90] < need[95] <= 144

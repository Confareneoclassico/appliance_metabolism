"""The sample of the sensitivity analysis: ranges, comonotone groups, levels,
nesting and the Shapley players."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsa import design  # noqa: E402


def test_nested_and_reproducible():
    big, small = design.sample(10), design.sample(7)
    assert big.shape == (1024, len(design.COLUMNS))
    assert np.array_equal(big[:128], small)                 # the first 128 runs are the small design
    assert np.array_equal(design.sample(10), big)


def test_ranges_and_comonotone_groups():
    runs = design.transform(design.sample(10))
    tv = np.array([r["X4"][821] for r in runs])
    radio = np.array([r["X4"][831] for r in runs])
    assert tv.min() >= 50 and tv.max() <= 200 and radio.min() >= 5 and radio.max() <= 30
    # same quantile of each log range
    assert np.allclose(np.log(tv / 50) / np.log(4), np.log(radio / 5) / np.log(6))
    # log-uniform: median near the geometric mean, about 34% of TV powers below 71 W
    assert abs(np.median(tv) - 100) < 5
    assert abs(np.mean(tv < 71) - np.log(71 / 50) / np.log(4)) < 0.03
    five = np.array([r["X6"][31] for r in runs])
    assert five.min() >= 0 and five.max() <= 20               # uniform from zero
    iron = np.array([r["X2"][332] for r in runs])
    assert iron.max() <= 2800 and iron.min() >= 1000


def test_uniform_in_watts_check():
    runs = design.transform(design.sample(8), uniform_in_watts=True)
    tv = np.array([r["X4"][821] for r in runs])
    assert abs(np.median(tv) - 125) < 8


def test_levels_are_balanced():
    U = design.sample(10)
    X = design.analysis_matrix(U)
    j = [f.id for f in design.FACTORS].index("X9")
    assert np.all(np.abs(np.bincount(X[:, j].astype(int)) - 1024 / 3) < 3)
    k = [f.id for f in design.FACTORS].index("X16")
    assert set(np.unique(X[:, k])) == set(range(6, 13))


def test_players():
    names, cols = design.players()
    assert len(names) == 16 and names[0] == "P_appliances" and names[1] == "P_country"
    assert cols[0] == [0, 1, 2, 3, 4, 5] and cols[1] == [6, 7] and names[-1] == "D_cont"
    assert design.DUMMIES == ["D_cont", "D_2", "D_3", "D_6", "D_7"]

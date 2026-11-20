"""The analysis of the sample (gsa/analyse.py): the dependence between the
columns of a design, the raking that restores the shares of the leading
factors within the classes of another, and the adjusted indices, on designs
built to be dependent."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsa import analyse as an, design, estimators as est  # noqa: E402


def confounded(n=1000, seed=0):
    """A binary factor b that drives Y alone, a continuous factor x of no
    effect whose quantile classes hold unbalanced shares of b, and a
    continuous column z independent of both."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    z = rng.random(n)
    k = est.classes(x)
    p = np.where(k % 2 == 0, 0.25, 0.75)              # share of b = 1 by class of x
    b = (rng.random(n) < p).astype(float)
    y = 10 * b + rng.normal(0, 0.5, n)
    return np.column_stack([x, b, z]), y[:, None]


def test_w2sq_weighted_matches_the_exact_distance():
    rng = np.random.default_rng(1)
    a, b = np.sort(rng.normal(size=37)), np.sort(rng.normal(size=500))
    assert np.isclose(an.w2sq_weighted(a, np.full(37, 1 / 37), b), est.w2sq_exact(a, b))
    # weights that stand for a repeated value
    a2 = np.array([0.0, 1.0, 2.0])
    dup = np.array([0.0, 0.0, 1.0, 2.0])
    assert np.isclose(an.w2sq_weighted(a2, np.array([0.5, 0.25, 0.25]), b), est.w2sq_exact(dup, b))


def test_design_dependence_finds_the_confounded_pair():
    X, _ = confounded()
    d = an.design_dependence(X, names=["x", "b", "z"])
    idx = d.set_index(["classes_of", "column"])["index"]
    assert idx[("x", "b")] > 0.2                      # the classes of x carry b
    assert idx[("z", "b")] < 3 * d.independent.max()  # z does not
    assert idx[("x", "z")] < 3 * d.independent.max()


def test_raking_restores_the_shares_of_the_drivers():
    X, _ = confounded()
    c = est.classes(X[:, 0])
    K = c.max() + 1
    b = X[:, 1].astype(int)
    target = np.bincount(b) / len(b)
    W, gap = an.raking_weights(np.column_stack([c, est.classes(X[:, 2])]), [(b, 2, target)])
    for j, cj in enumerate((c, est.classes(X[:, 2]))):
        Kj = cj.max() + 1
        tot = np.bincount(cj * 2 + b, weights=W[:, j], minlength=2 * Kj).reshape(Kj, 2)
        assert np.allclose(tot / tot.sum(axis=1, keepdims=True), target[None, :], atol=1e-6)
        assert np.allclose(np.bincount(cj, weights=W[:, j]), np.bincount(cj))
    assert np.all(gap < 1e-6)
    # a class without one of the driver's classes keeps an imbalance
    c2 = c.copy()
    c2[(c == 0) & (b == 1)] = 1
    _, gap2 = an.raking_weights(c2, [(b, 2, target)])
    assert gap2[0] > 0.01


def test_adjusted_indices_remove_the_effect_taken_on_from_the_driver():
    X, Y = confounded()
    S_raw, O_raw = an.class_indices(X, Y)
    S, O, G = an.adjusted_indices(X, Y, [[1, 0, 2]])
    assert S_raw[0, 0] > 0.15 and O_raw[0, 0] > 0.15  # x takes on the effect of b
    assert S[0, 0] < 0.03 and O[0, 0] < 0.05          # not once the shares of b are restored
    assert np.isclose(S[1, 0], S_raw[1, 0], atol=0.01)  # the driver itself is unchanged
    assert np.all(S / 2 <= O + 1e-12)


def test_adjusted_indices_leave_an_independent_design_alone():
    rng = np.random.default_rng(3)
    X = rng.random((2000, 3))
    y = (np.sin(2 * np.pi * X[:, 0]) + 2 * X[:, 1] ** 2 + 0.1 * rng.normal(size=2000))[:, None]
    S_raw, O_raw = an.class_indices(X, y)
    S, O, _ = an.adjusted_indices(X, y, an.leading_drivers(S_raw, names=["a", "b", "c"],
                                                           factors=["a", "b", "c"]))
    assert np.allclose(S, S_raw, atol=0.02) and np.allclose(O, O_raw, atol=0.02)


def test_no_power_of_two_classes():
    assert [est.n_quantile_classes(n) for n in (128, 256, 512, 1024)] == [11, 15, 22, 25]


def test_surrogate_on_a_known_model():
    """Shapley effects of the surrogate on a model of two factors, X19 and
    X13, with the continuous dummy near zero."""
    U = design.sample(9)                                    # 512 runs
    X = design.analysis_matrix(U)
    t = pd.DataFrame(X, columns=design.COLUMNS)
    y = 2 * t["X19"] / 150 + t["X13"]                       # Var: 1/3 and 1/4
    out = an.surrogate_indices(t, y.to_numpy()[:, None], B=0, n_perm=2000, n_iid=4096,
                               outputs=["Y"])
    sh = out[out.shapley.notna()].set_index("player")["shapley"]
    assert out.cv_r2.iloc[0] > 0.97
    assert np.isclose(sh["X19"], (1 / 3) / (1 / 3 + 1 / 4), atol=0.06)
    assert np.isclose(sh["X13"], (1 / 4) / (1 / 3 + 1 / 4), atol=0.06)
    assert abs(sh["D_cont"]) < 0.02

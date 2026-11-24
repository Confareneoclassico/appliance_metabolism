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


def test_stability_counts_crossings_and_order():
    from gsa.tables import stability_counts
    a = pd.Series({"X1": 0.30, "X2": 0.10, "X3": 0.04, "X4": 0.06, "X5": 0.01, "X6": 0.0})
    b = pd.Series({"X1": 0.31, "X2": 0.05, "X3": 0.06, "X4": 0.04, "X5": 0.02, "X6": 0.0})
    up, down, kept, top2, top5 = stability_counts(a, b)
    assert (up, down) == (1, 1)          # X3 rises to 0.05, X4 falls below it
    assert kept                          # X1 ahead of X2 at both sizes
    assert not top2 and not top5         # X3 overtakes X2
    same = stability_counts(a, a * 1.01)
    assert same == (0, 0, True, True, True)


def test_ot_parts_add_up_and_mean_part_is_half_s():
    rng = np.random.default_rng(3)
    n = 4000
    X = rng.random((n, 3))
    y = X[:, 0] + 0.6 * (0.2 + X[:, 1]) * rng.normal(size=n)    # X0 shifts, X1 spreads
    cls, K, off = an.column_classes(X)
    S, O, P = an.indices_one_output(cls, K, off, np.ones((n, 3)), y, parts=True)
    assert np.allclose(P.sum(axis=0), O)
    assert np.allclose(P[0], S / 2)
    assert (P >= -1e-12).all()
    assert P[0, 0] > P[1, 0]                 # X0: mostly the mean part
    assert P[1, 1] > 5 * P[0, 1]             # X1: mostly the spread part
    assert O[1] > 2 * an.location_ot(S[1])   # far above the curve of a pure shift


def test_location_ot_matches_a_gaussian_shift():
    rng = np.random.default_rng(4)
    n = 40000
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)               # S = 1/2
    cls, K, off = an.column_classes(x[:, None])
    S, O = an.indices_one_output(cls, K, off, np.ones((n, 1)), y)
    assert abs(S[0] - 0.5) < 0.02
    assert abs(O[0] - an.location_ot(S[0])) < 0.01
    assert abs(an.location_ot(1.0) - 1.0) < 1e-12 and an.location_ot(0.0) == 0.0


def test_top_down_correlation():
    a = np.array([0.5, 0.3, 0.1, 0.05, 0.0])
    assert abs(an.top_down_correlation(a, a) - 1) < 1e-12
    assert np.isclose(an.savage_scores(a).sum(), len(a))
    swap_top = np.array([0.3, 0.5, 0.1, 0.05, 0.0])
    swap_tail = np.array([0.5, 0.3, 0.1, 0.0, 0.05])
    assert an.top_down_correlation(a, swap_tail) > an.top_down_correlation(a, swap_top)


def test_pick_freeze_on_an_analytic_function():
    rng = np.random.default_rng(5)
    f = lambda Z: Z[:, 0] + 2 * Z[:, 1] + Z[:, 0] * Z[:, 2]      # noqa: E731, uniform inputs
    A, B = rng.random((2 ** 17, 3)), rng.random((2 ** 17, 3))
    res = an.pick_freeze(f, A, B, [[0], [1], [2], [0, 2]])
    v0, v1, v2, v02 = 2.25 / 12, 4 / 12, 0.25 / 12, 1 / 144
    var = v0 + v1 + v2 + v02
    S = res[:, 0] / var
    T = res[:, 2] / var
    assert np.allclose(S, [v0 / var, v1 / var, v2 / var, (v0 + v2 + v02) / var], atol=0.015)
    assert np.allclose(T, [(v0 + v02) / var, v1 / var, (v2 + v02) / var, (v0 + v2 + v02) / var], atol=0.015)
    # an offset changes nothing (the values are centred before the products)
    g = lambda Z: 100 + f(Z)                                     # noqa: E731
    res_g = an.pick_freeze(g, A, B, [[0], [1], [2], [0, 2]])
    assert np.allclose(res_g[:, 0], res[:, 0], atol=1e-8) and np.allclose(res_g[:, 2], res[:, 2])
    assert (res_g[:, 1] < 0.01).all()                            # standard errors stay small

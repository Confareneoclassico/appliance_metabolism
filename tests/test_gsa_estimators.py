"""Estimators of the sensitivity analysis on test functions with known
indices: the Ishigami function and a linear model with Gaussian inputs."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsa import estimators as est  # noqa: E402

# Ishigami, a = 7, b = 0.1, inputs uniform on [-pi, pi]
A, B = 7.0, 0.1
V1 = 0.5 * (1 + B * math.pi ** 4 / 5) ** 2
V2 = A ** 2 / 8
V13 = B ** 2 * math.pi ** 8 * (1 / 18 - 1 / 50)
V = V1 + V2 + V13
S_ISHIGAMI = np.array([V1, V2, 0.0]) / V
SH_ISHIGAMI = np.array([V1 + V13 / 2, V2, V13 / 2]) / V


def ishigami(u):
    x = -math.pi + 2 * math.pi * u
    return np.sin(x[:, 0]) + A * np.sin(x[:, 1]) ** 2 + B * x[:, 2] ** 4 * np.sin(x[:, 0])


def test_ishigami_reference_values():
    assert np.allclose(S_ISHIGAMI, [0.3139, 0.4424, 0.0], atol=1e-4)
    assert np.allclose(SH_ISHIGAMI, [0.4358, 0.4424, 0.1218], atol=1e-4)


def test_first_order_and_ot_on_ishigami():
    rng = np.random.default_rng(1)
    u = rng.random((8192, 3))
    y = ishigami(u)
    s, o = est.indices_all(u, y)
    assert np.allclose(s, S_ISHIGAMI, atol=0.02)
    assert np.all(o >= s / 2)


def test_linear_gaussian_ot_index():
    # Y = sum b_i X_i with X_i ~ N(0, 1): Y | X_i is Gaussian with variance
    # Var Y - b_i^2, so iota_i = S_i / 2 + (1 - sqrt(1 - S_i))^2 / 2
    rng = np.random.default_rng(2)
    b = np.array([1.0, 2.0, 0.5])
    x = rng.standard_normal((20000, 3))
    y = x @ b
    s_true = b ** 2 / np.sum(b ** 2)
    iota_true = s_true / 2 + (1 - np.sqrt(1 - s_true)) ** 2 / 2
    s, o = est.indices_all(x, y)
    assert np.allclose(s, s_true, atol=0.02)
    assert np.allclose(o, iota_true, atol=0.02)


def test_w2sq_exact():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(40), rng.standard_normal(40) + 1
    assert math.isclose(est.w2sq_exact(a, b), np.mean((np.sort(a) - np.sort(b)) ** 2), rel_tol=1e-12)
    # a sample against itself repeated three times is at distance zero
    assert est.w2sq_exact(a, np.repeat(a, 3)) == 0.0
    # sizes 7 and 5: against the quantile functions on a fine grid
    a, b = rng.standard_normal(7), rng.standard_normal(5)
    u = (np.arange(350_000) + 0.5) / 350_000
    qa = np.sort(a)[np.minimum((u * 7).astype(int), 6)]
    qb = np.sort(b)[np.minimum((u * 5).astype(int), 4)]
    assert math.isclose(est.w2sq_exact(a, b), np.mean((qa - qb) ** 2), rel_tol=1e-6)
    # W2 is at least the difference of the means
    assert est.w2sq_exact(a, b) >= (a.mean() - b.mean()) ** 2


def test_classes():
    x = np.repeat([1.0, 2.0, 3.0], 10)
    assert np.array_equal(est.classes(x), np.repeat([0, 1, 2], 10))
    z = np.arange(100.0)
    c = est.classes(z)
    assert c.max() == 9 and np.all(np.bincount(c) == 10)      # 10 classes of 10


def test_shapley_on_ishigami():
    rng = np.random.default_rng(4)
    u = rng.random((1024, 3))
    sh = est.shapley_knn(u, ishigami(u), [[0], [1], [2]], k=3)
    assert math.isclose(sh.sum(), 1.0, rel_tol=1e-12)
    assert np.allclose(sh, SH_ISHIGAMI, atol=0.05)


def test_shapley_groups_and_dummy():
    # two factors as one player give the effect of the pair; a dummy gets ~0
    rng = np.random.default_rng(5)
    u = rng.random((1024, 4))
    y = ishigami(u[:, :3])
    sh = est.shapley_knn(u, y, [[0, 2], [1], [3]], k=3)
    assert math.isclose(sh[0], SH_ISHIGAMI[0] + SH_ISHIGAMI[2], abs_tol=0.05)
    assert abs(sh[2]) < 0.03


def test_shapley_weights_sum():
    M = est.shapley_weights(4)
    # an additive cost gives each player its own share
    W = np.array([sum(v for j, v in enumerate([0.1, 0.2, 0.3, 0.4]) if m & (1 << j))
                  for m in range(16)])
    assert np.allclose(M @ W, [0.1, 0.2, 0.3, 0.4])


def test_convergence_statistics_on_ishigami():
    from gsa import convergence as conv
    from scipy.stats import qmc
    u = qmc.Sobol(d=5, scramble=True, seed=7).random_base2(10)   # 3 factors + 2 dummies
    X = u.copy()
    X[:, 4] = (u[:, 4] * 3).astype(int)                            # a 3-level dummy
    y = ishigami(u[:, :3])
    res = conv.nested(X, y, [256, 512, 1024], "S", factor_cols=[0, 1, 2], dummy_col=3, B=100)
    last = res[-1]
    assert list(last["nonzero"]) == [True, True, False]            # X3 has no first-order effect
    assert last["stat_ranking"] < 1
    assert last["lo"][1] < S_ISHIGAMI[1] < last["hi"][1] + 0.02
    assert "max_change" in last


def test_shapley_multi_matches_single_and_noise():
    rng = np.random.default_rng(6)
    u = rng.random((512, 4))
    y1 = ishigami(u[:, :3])
    y2 = y1 + 2.0 * rng.standard_normal(512)                       # a noisy output
    players = [[0], [1], [2]]
    sh, share = est.shapley_knn_multi(u, np.column_stack([y1, y1]), players, k=3, noise=False)
    assert np.allclose(sh[:, 0], est.shapley_knn(u, y1, players, k=3)) and np.allclose(share, 0)
    sh, share = est.shapley_knn_multi(u, np.column_stack([y1, y2]), players, k=3)
    assert np.allclose(sh.sum(axis=0) + share, 1.0)
    # on a smooth model the residual share is the noise: Y = X1 + 2 X2 + e
    lin = u[:, 0] + 2 * u[:, 1]
    y3 = lin + 0.5 * rng.standard_normal(512)
    sh3, share3 = est.shapley_knn_multi(u, np.column_stack([lin, y3]), players, k=3)
    assert share3[0] < 0.05
    assert share3[1] == pytest.approx(0.25 / y3.var(), abs=0.05)
    assert np.allclose(sh3[:, 0], [1 / 5, 4 / 5, 0], atol=0.06)
    counts = rng.multinomial(512, np.full(512, 1 / 512), size=20)
    sh2, share2, shb, shareb = est.shapley_knn_multi(u, np.column_stack([y1, y2]), players, k=3,
                                                     counts=counts)
    assert np.allclose(sh2, sh) and shb.shape == (20, 3, 2) and shareb.shape == (20, 2)
    assert np.allclose(shb.sum(axis=1) + shareb, 1.0)


def test_song_shapley_on_ishigami():
    f = lambda x: ishigami(x)                                     # noqa: E731
    sample = lambda r, g: g.random((r, 3))                        # noqa: E731
    sh, var_f = est.song_shapley(f, sample, [[0], [1], [2]], n_perm=20000, seed=1)
    assert var_f == pytest.approx(V, rel=0.05)
    assert np.allclose(sh, SH_ISHIGAMI, atol=0.03) and sh.sum() == pytest.approx(1.0)

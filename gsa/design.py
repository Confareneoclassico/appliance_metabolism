"""Factors and sample of the sensitivity analysis (design note of 19 November
2026, Table 3).

Every run is one point of a scrambled Sobol' sequence over the unit cube.
Each factor has its own coordinate, the two group factors (appliance powers
X1 to X6, country heterogeneity X7 and X8) included, and the transform below
turns the coordinates into factor values. Within an appliance factor every
activity is placed at the same quantile of its own range (comonotone), since
what is uncertain is the state of knowledge about a class of appliances. The
first 2^m points of the sequence are themselves balanced, so a sample of 1,024
runs contains the nested designs of 128, 256 and 512 runs, and extending it to
2,048 or 4,096 keeps the first 1,024 runs.

Matched dummies, a continuous one and one for each number of levels in the
design, get coordinates of their own and are analysed like the factors; they
do not enter the model.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

# The 18 activities at 5 W other than radio (831), in X6
FIVE_WATT = [31, 39, 329, 333, 351, 352, 353, 359, 363, 421, 422, 424, 513, 514, 529, 619, 719, 732]


@dataclass
class Factor:
    id: str
    name: str
    kind: str                 # 'power', 'uniform', 'levels', 'integer', 'seed'
    low: float = 0.0
    high: float = 1.0
    levels: tuple = ()        # level labels for kind 'levels'
    v8: object = None         # value in the run behind manuscript v8
    codes: dict = field(default_factory=dict)   # power factors: code -> (low, high, law)
    player: str = ""          # Shapley player (group) of the factor


# Power factors: one coordinate each, every code at the same quantile of its
# own range; law 'log' (log-uniform) unless the range starts at zero
POWER = {
    "X1": {311: (400, 1200, "log")},
    "X2": {331: (250, 800, "log"), 332: (1000, 2800, "log")},
    "X3": {321: (300, 1000, "log")},
    "X4": {821: (50, 200, "log"), 831: (5, 30, "log")},
    "X5": {**{c: (40, 250, "log") for c in (721, 722, 723, 729)}, 733: (40, 250, "log")},
    "X6": {371: (20, 150, "log"), **{c: (0, 20, "lin") for c in FIVE_WATT}},
}

FACTORS = [
    Factor("X1", "Cooking (311)", "power", codes=POWER["X1"], v8=810, player="P_appliances"),
    Factor("X2", "Laundry (331) and ironing (332)", "power", codes=POWER["X2"], v8=(530, 2800),
           player="P_appliances"),
    Factor("X3", "Cleaning (321)", "power", codes=POWER["X3"], v8=620, player="P_appliances"),
    Factor("X4", "Television (821) and radio (831)", "power", codes=POWER["X4"], v8=(71, 5),
           player="P_appliances"),
    Factor("X5", "Computing (721-729) and games (733)", "power", codes=POWER["X5"], v8=(201, 210),
           player="P_appliances"),
    Factor("X6", "Household management (371) and the 18 activities at 5 W", "power",
           codes=POWER["X6"], v8=(70, 5), player="P_appliances"),
    Factor("X7", "SD of log country multipliers", "uniform", 0.0, 0.3, v8=0.0, player="P_country"),
    Factor("X8", "Seed of the country multipliers", "seed", player="P_country"),
    Factor("X9", "Shared-appliance rule", "levels", levels=("union", "per_person", "tv_capped"),
           v8="union"),
    Factor("X10", "Ownership", "levels", levels=("ignored", "applied"), v8="ignored"),
    Factor("X11", "Weight of codes recorded only as secondary", "uniform", 0.0, 1.0, v8=1.0),
    Factor("X12", "Base load (W per household-day)", "uniform", 0.0, 100.0, v8=0.0),
    Factor("X13", "Diarist coverage", "levels", levels=("all", "first_listed"), v8="all"),
    Factor("X14", "Day type", "levels", levels=("ddv7_then_ddv1", "ddv1_only"), v8="ddv7_then_ddv1"),
    Factor("X15", "Survey weight", "levels", levels=("none", "first_listed", "mean"), v8="none"),
    Factor("X16", "K of the global solution", "integer", 6, 12, v8=9),
    Factor("X17", "K of the stratified solutions", "integer", 3, 8, v8="knee or 4"),
    Factor("X18", "Share of variance kept by PCA", "uniform", 0.80, 0.99, v8=0.95),
    Factor("X19", "Lighting (W per household while lit)", "uniform", 0.0, 150.0, v8=0.0),
    Factor("X20", "Laundry run-on (minutes)", "uniform", 0.0, 120.0, v8=0.0),
    Factor("X21", "Seed of K-means", "seed", v8=42),
]
for f in FACTORS:
    f.player = f.player or f.id


def level_counts():
    """Numbers of levels of the discrete factors (for the matched dummies)."""
    counts = set()
    for f in FACTORS:
        if f.kind == "levels":
            counts.add(len(f.levels))
        elif f.kind == "integer":
            counts.add(int(f.high - f.low + 1))
    return sorted(counts)


DUMMIES = ["D_cont"] + [f"D_{k}" for k in level_counts()]
COLUMNS = [f.id for f in FACTORS] + DUMMIES


def sample(m, seed=20261119, skip=0):
    """Coordinates of the points skip+1 to 2^m of the scrambled Sobol'
    sequence over all factors and dummies, as an array (2^m - skip, len(COLUMNS)).
    The same seed always gives the same sequence, so a larger m extends a
    smaller one."""
    sampler = qmc.Sobol(d=len(COLUMNS), scramble=True, seed=seed)
    return sampler.random_base2(m)[skip:]


def _power(u, low, high, law, uniform_in_watts=False):
    if law == "lin" or uniform_in_watts:
        return low + u * (high - low)
    return low * (high / low) ** u


def transform(U, uniform_in_watts=False):
    """Factor values of every run: a list of dicts, one per run. Power
    factors give a dict code -> W; levels give a label; integers an int;
    seeds an int in [0, 2^31). With uniform_in_watts, the power factors are
    uniform in W on their ranges (the robustness check of the design)."""
    U = np.asarray(U)
    runs = []
    for row in U:
        run = {}
        for j, f in enumerate(FACTORS):
            u = row[j]
            if f.kind == "power":
                run[f.id] = {c: float(_power(u, lo, hi, law, uniform_in_watts))
                             for c, (lo, hi, law) in f.codes.items()}
            elif f.kind == "uniform":
                run[f.id] = float(f.low + u * (f.high - f.low))
            elif f.kind == "levels":
                run[f.id] = f.levels[min(int(u * len(f.levels)), len(f.levels) - 1)]
            elif f.kind == "integer":
                k = int(f.high - f.low + 1)
                run[f.id] = int(f.low) + min(int(u * k), k - 1)
            elif f.kind == "seed":
                run[f.id] = int(u * 2 ** 31) % (2 ** 31)
        runs.append(run)
    return runs


def dummy_values(U):
    """Values of the dummies: the continuous one is its coordinate, the
    others are levels 0 .. k - 1."""
    U = np.asarray(U)
    out = {}
    for j, name in enumerate(DUMMIES):
        u = U[:, len(FACTORS) + j]
        out[name] = u if name == "D_cont" else np.minimum((u * int(name[2:])).astype(int),
                                                          int(name[2:]) - 1)
    return out


def analysis_matrix(U):
    """Values of the factors and dummies as numbers, for the estimators: the
    coordinate for power factors and seeds (monotone in the factor value),
    the value for uniform and integer factors, the level index for levels."""
    U = np.asarray(U)
    X = np.zeros_like(U)
    for j, f in enumerate(FACTORS):
        u = U[:, j]
        if f.kind == "levels":
            X[:, j] = np.minimum((u * len(f.levels)).astype(int), len(f.levels) - 1)
        elif f.kind == "integer":
            k = int(f.high - f.low + 1)
            X[:, j] = f.low + np.minimum((u * k).astype(int), k - 1)
        elif f.kind == "uniform":
            X[:, j] = f.low + u * (f.high - f.low)
        else:
            X[:, j] = u
    for j, (name, v) in enumerate(dummy_values(U).items()):
        X[:, len(FACTORS) + j] = v
    return X


def players(include_dummy=True):
    """Shapley players as lists of columns of the coordinate matrix: the
    appliance factors and the two country factors as groups, the others one
    by one, and the continuous dummy last."""
    groups = {}
    for j, f in enumerate(FACTORS):
        groups.setdefault(f.player, []).append(j)
    out = list(groups.items())
    if include_dummy:
        out.append(("D_cont", [len(FACTORS)]))
    return [name for name, _ in out], [cols for _, cols in out]

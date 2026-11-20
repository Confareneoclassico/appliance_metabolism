"""One run of the model of the sensitivity analysis: factor values X1 to X21
in, outputs Y1 to Y11 and the mean profiles out.

The model rebuilds the power of every household-day and slot from the bundle
(gsa.bundle) with the run's appliance powers and rules, then repeats the steps
of the pipeline that the outputs rest on: the mean profiles by country and day
type and the comparison of the day types (Stage 06), standardisation, PCA and
K-means of the global solution (Stage 05) and of the 47 stratified ones (Stage
06), the peak-time statistics of the published groups of the global solution
(Stage 08) and the association tests (Stage 07), with the publication rules of
disclosure.py. At the values of BASELINE, with the K of both levels chosen by
the knee as in the pipeline and n_init = 20, it reproduces Stages 03 and 05 to
08 (gsa.check compares the two).

Power in slot t of household-day h, in W:

    p = m_country * (sum over the codes c in use at home of own_c * w_c * q_c
                     + own_331 * w_331 * r * v)
        + X19 * darkness(country, year, month, t) * [a diarist at home is in an
                                                    activity that needs light]
        + X12

where q_c depends on the shared-appliance rule (X9), the weight of codes
recorded only as secondary (X11) and the diarists covered (X13):

    union (v8)       q = 1 if a diarist at home has c as main activity, else X11
    per_person       q = (diarists with c as main activity)
                         + X11 (diarists with c as secondary activity only)
    tv_once          per_person, with television (821) counted as in union,
                     since the files only tell whether a household has a set
    first_listed     q of the first-listed diarist alone (1, X11 or 0), any X9

The run-on of laundry (X20) adds r = clip(X20 / 10 - (k - 1), 0, 1) in the k-th
slot after laundry stops (k = 1 to 12), with v = 1 if laundry was the main
activity in its last slot and X11 if it was only a secondary one. The country
multipliers are m = exp(X7 z), with z drawn from a standard normal with seed X8
for each country and centred over the countries of the bundle, so that they
change the differences between countries and not the overall level. Ownership
(X10 applied) sets own_c = 0 where the household lacks the appliance of code c
(gsa.design.OWNERSHIP); households not asked are taken to own it.

Survey weights (X15) enter every mean and share, the contingency tables of
the tests (scaled to the number of observations) and K-means (sample_weight);
standardisation and PCA are unweighted, as in the pipeline. The base load
shifts every slot equally, so it moves no peak: peak times are taken from the
power without it, and a household-day without other demand has no peak, as in
the pipeline.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from countries import COUNTRIES  # noqa: E402
from peaks import THETA  # noqa: E402

from gsa.design import POWER, OWNERSHIP  # noqa: E402
from gsa import benchmarks  # noqa: E402
from gsa.bundle import load_group, RUNON_SLOTS, LAUNDRY  # noqa: E402

SLOTS = 144
HOURS_PER_SLOT = 1 / 6
MIN_N_UNFLAGGED = 50
NOT_APPLICABLE, MISSING = -98, -99
SINGLE_DIARIST = ("HU", "NL", "NO")     # one diary per household (docs/PIPELINE.md)
DAY_TYPES = ("working", "non_working", "all")
SD_THRESHOLD_H = 2.0
N_INIT_ELBOW = 10                       # as in Stages 05 and 06
GLOBAL_K_RANGE, GLOBAL_K_FALLBACK = range(2, 16), 8            # Stage 05

# The values of the canonical run (version 1.1.0). X16 and X17 are "knee" for
# the K the pipeline chooses; the sample draws them as integers.
BASELINE = {
    "X1": {311: 810.0}, "X2": {331: 530.0, 332: 2800.0}, "X3": {321: 620.0},
    "X4": {821: 71.0, 831: 5.0},
    "X5": {721: 201.0, 722: 201.0, 723: 201.0, 729: 201.0, 733: 210.0},
    "X6": {371: 70.0, **{c: 5.0 for c in POWER["X6"] if c != 371}},
    "X7": 0.0, "X8": 0, "X9": "union", "X10": "ignored", "X11": 1.0, "X12": 0.0, "X13": "all",
    "X14": "ddv7_then_ddv1", "X15": "none", "X16": "knee", "X17": "knee", "X18": 0.95,
    "X19": 0.0, "X20": 0.0, "X21": 42,
}
OUTPUTS = ["Y1", "Y2", "Y2b", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10", "Y11"]


def bias_corrected_v(chi2, n, r, k):
    """Cramer's V with the bias correction of Bergsma (2013)."""
    if not np.isfinite(chi2) or n <= 1 or r < 2 or k < 2:
        return np.nan
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else np.nan


def chi2_table(table):
    """Pearson's chi-squared statistic (no continuity correction) of a table
    with its empty rows and columns dropped, as pd.crosstab and
    chi2_contingency(correction=False) give it, with its total and shape."""
    t = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n = float(t.sum())
    if t.shape[0] < 2 or t.shape[1] < 2 or n <= 0:
        return np.nan, n, t.shape
    e = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    return float(((t - e) ** 2 / e).sum()), n, t.shape


def circular_sd_hours(slot_w, n_days):
    """Circular SD (sqrt(-2 ln R), in hours) of the peak time of each row of
    slot weights (groups x 144); NaN for groups with fewer than two
    household-days with demand (peaks.circular_peak_stats)."""
    c, s = slot_w @ np.cos(THETA), slot_w @ np.sin(THETA)
    tot = slot_w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.hypot(c, s) / tot
        sd = np.sqrt(-2.0 * np.log(np.clip(r, 1e-300, 1.0))) * 24 / (2 * np.pi)
    sd[(n_days < 2) | ~(tot > 0)] = np.nan
    return sd


def group_peak_weights(p, base, groups, n_groups, weight, chunk=20000):
    """Slot weights of the daily peaks summed over the members of each group
    (groups x 144) and the number of members with demand: the weight of each
    household-day is shared equally among the slots at its daily maximum, and
    household-days without demand above the base load have no peak
    (peaks.peak_weights, with the base load taken out)."""
    slot_w = np.zeros((n_groups, SLOTS))
    n_days = np.zeros(n_groups)
    for a in range(0, len(p), chunk):
        e = p[a:a + chunk]
        mx = e.max(axis=1)
        tie = (e == mx[:, None]) & (mx > base)[:, None]
        m = tie.sum(axis=1)
        has = m > 0
        w = np.zeros(len(e))
        w[has] = weight[a:a + chunk][has] / m[has]
        onehot = np.zeros((len(e), n_groups))
        onehot[np.arange(len(e)), groups[a:a + chunk]] = w
        slot_w += onehot.T @ tie
        n_days += np.bincount(groups[a:a + chunk], weights=has, minlength=n_groups)
    return slot_w, n_days


def option_a(labels):
    """Published groups of a solution (disclosure.published_groups): every
    cluster of fewer than MIN_N_UNFLAGGED household-days is reported with the
    largest, ties for the largest going to the lowest label."""
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    small = np.flatnonzero((sizes > 0) & (sizes < MIN_N_UNFLAGGED)
                           & (np.arange(len(sizes)) != largest))
    groups = labels.copy()
    groups[np.isin(labels, small)] = largest
    return groups, len(small)


def indicator(groups, n_groups, weight):
    """Sparse matrix (n x n_groups) of the weights of the members of each
    group; each row of `groups` gives one group index per household-day (-1
    for none), so that a household-day can belong to several groups."""
    g = np.atleast_2d(groups)
    rows = np.repeat(np.arange(g.shape[1])[None, :], g.shape[0], axis=0).ravel()
    cols = g.ravel()
    keep = cols >= 0
    w = np.broadcast_to(np.asarray(weight, dtype=float), g.shape).ravel()
    return sparse.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=(g.shape[1], n_groups))


class Model:
    def __init__(self, folder, n_init=20, mmap=True):
        folder = Path(folder)
        self.meta = json.loads((folder / "meta.json").read_text())
        e = load_group(folder, "entries", mmap)
        self.row, self.day, self.code = e["row"], e["day"], e["code"]
        self.cm, self.cs, self.fm, self.fs = e["cm"], e["cs"], e["fm"], e["fs"]
        self.codes = np.asarray(e["codes"])
        h = load_group(folder, "households", mmap)
        self.n = n = len(h["country"])
        self.country = np.asarray(h["country"]).astype(np.int64)
        self.day_type = {"ddv7_then_ddv1": np.asarray(h["day_type_ddv7"]),
                         "ddv1_only": np.asarray(h["day_type_ddv1"])}
        self.weight = {"none": None, "first_listed": np.asarray(h["weight_first"], dtype=float),
                       "mean": np.asarray(h["weight_mean"], dtype=float)}
        self.features = np.asarray(h["features"]).astype(np.int64)
        self.feature_names = list(self.meta["feature_names"])
        self.reference = np.asarray(h["reference"]).astype(np.int64)
        self.dark_row = np.asarray(h["dark_row"])
        s = load_group(folder, "slots", mmap)
        self.lit = {"all": s["lit_all"], "first_listed": s["lit_first"]}
        self.runon = {"all": (s["runon_all"], s["runon_all_main"]),
                      "first_listed": (s["runon_first"], s["runon_first_main"])}
        self.dark_table = np.asarray(load_group(folder, "darkness", mmap)["table"], dtype=np.float64)
        self.countries = sorted(set(self.country.tolist()))
        self.n_init = n_init
        self.code_pos = {int(c): i for i, c in enumerate(self.codes)}
        self.tv = self.code_pos.get(821, -1)
        self.laundry = self.code_pos.get(LAUNDRY, -1)
        # ownership: entries of a code whose appliance the household lacks,
        # and household-days without a washing machine (for the run-on)
        self.unowned = np.zeros(len(self.row), bool)
        self.no_washer = np.zeros(n, bool)
        for item, spec in OWNERSHIP.items():
            lacks = np.asarray(h[item.lower()]) == spec["lacks"]
            pos = [self.code_pos[c] for c in spec["codes"] if c in self.code_pos]
            self.unowned |= np.isin(self.code, pos) & lacks[self.day]
            if LAUNDRY in spec["codes"]:
                self.no_washer |= lacks
        # categories of the background variables
        self.cats = []
        for j in range(self.features.shape[1]):
            v = self.features[:, j]
            valid = (v != NOT_APPLICABLE) & (v != MISSING)
            _, inv = np.unique(v, return_inverse=True)
            self.cats.append((valid, inv.astype(np.int64).ravel(), int(inv.max()) + 1))
        self.nonc = [j for j, f in enumerate(self.feature_names) if f != "COUNTRY"]
        self._eligible = {}

    # ------------------------------------------------------------------ power
    def powers(self, x):
        """Power of each of the bundle's codes, in W."""
        w = np.zeros(len(self.codes))
        for fid in ("X1", "X2", "X3", "X4", "X5", "X6"):
            for c, val in x[fid].items():
                if int(c) in self.code_pos:
                    w[self.code_pos[int(c)]] = float(val)
        return w

    def multipliers(self, sigma, seed):
        """Multiplier of every country code (centred log normal)."""
        z = np.random.default_rng(int(seed)).standard_normal(len(COUNTRIES))
        z = z - z[np.array(self.countries)].mean()
        return np.exp(float(sigma) * z)

    def profiles(self, x):
        """Household power (n, 144) in W for the factor values x. The
        temporary arrays are modified in place to keep memory low."""
        w = self.powers(x)
        x11 = float(x["X11"])
        if x["X13"] == "first_listed":
            cover = "first_listed"
            q = self.fm.astype(np.float64)
            q += x11 * self.fs
        else:
            cover = "all"
            cm = self.cm.astype(np.float64)
            if x["X9"] == "union":
                q = np.where(cm > 0, 1.0, x11)
            else:
                q = cm + x11 * self.cs
                if x["X9"] == "tv_once":
                    tv = self.code == self.tv
                    q[tv] = np.where(cm[tv] > 0, 1.0, x11)
            del cm
        q *= w[self.code]
        if x["X10"] == "applied":
            q[self.unowned] = 0.0
        p = np.bincount(self.row, weights=q, minlength=self.n * SLOTS).reshape(self.n, SLOTS)
        del q
        if float(x["X20"]) > 0 and self.laundry >= 0:
            k, was_main = self.runon[cover]
            lut = np.zeros(RUNON_SLOTS + 1)
            lut[1:] = np.clip(float(x["X20"]) / 10.0 - np.arange(RUNON_SLOTS), 0.0, 1.0)
            extra = lut[k]
            if x11 != 1.0:
                np.multiply(extra, x11, out=extra, where=~np.asarray(was_main))
            scale = np.full(self.n, w[self.laundry])
            if x["X10"] == "applied":
                scale[self.no_washer] = 0.0
            extra *= scale[:, None]
            p += extra
            del extra
        if float(x["X7"]) > 0:
            p *= self.multipliers(x["X7"], x["X8"])[self.country][:, None]
        if float(x["X19"]) > 0:
            light = self.dark_table[self.dark_row]
            light *= self.lit[cover]
            light *= float(x["X19"])
            p += light
            del light
        if float(x["X12"]) > 0:
            p += float(x["X12"])
        return p

    # ------------------------------------------------------------- clusterings
    def _kmeans(self, z, k, seed, weight, n_init, last=False):
        """K-means as in the pipeline. On the last use of z it is centred in
        place (copy_x=False), which gives the same result with less memory."""
        km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=n_init, copy_x=not last)
        labels = km.fit_predict(z, sample_weight=weight)
        return labels, km.inertia_

    def cluster(self, p, k, variance, seed, weight, k_range, fallback):
        """Labels of one solution: standardised slots, PCA keeping a share
        `variance` of the variance, and K-means with K = k, or with the knee of
        the WCSS curve over k_range (fallback where none is found) for
        k = 'knee', as Stages 05 and 06 choose it. Returns the labels, K,
        whether K is a fallback and the number of components."""
        z = StandardScaler().fit_transform(p)
        # PCA centres its input in place (copy=False), which gives the same
        # result as the pipeline's PCA with one copy fewer
        pca = PCA(n_components=float(variance), copy=False)
        z = pca.fit_transform(z)
        is_fallback = False
        if k == "knee":
            wcss = [self._kmeans(z, j, seed, weight, N_INIT_ELBOW)[1] for j in k_range]
            if len(wcss) > 1:
                from kneed import KneeLocator
                kl = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
                k, is_fallback = (kl.elbow, False) if kl.elbow else (fallback, True)
            else:
                k, is_fallback = 2, True
        labels, _ = self._kmeans(z, k, seed, weight, self.n_init, last=True)
        return labels, int(k), is_fallback, int(pca.n_components_)

    def solutions(self, day_type):
        """The stratified solutions of Stage 06 with the names of its files:
        the two day types, the countries (all their household-days) and every
        country and day type with at least MIN_N_UNFLAGGED household-days."""
        out = {}
        for d, name in ((0, "Working_Days"), (1, "Non_Working_Days")):
            idx = np.flatnonzero(day_type == d)
            if len(idx) >= MIN_N_UNFLAGGED:
                out[f"{name}_clustering_results.csv"] = idx
        for c in self.countries:
            idx = np.flatnonzero(self.country == c)
            if len(idx) >= MIN_N_UNFLAGGED:
                out[f"{COUNTRIES[c]}_clustering_results.csv"] = idx
        for c in self.countries:
            for d, name in ((0, "working_days"), (1, "non_working_days")):
                idx = np.flatnonzero((self.country == c) & (day_type == d))
                if len(idx) >= MIN_N_UNFLAGGED:
                    out[f"{COUNTRIES[c]}_{name}_clustering_results.csv"] = idx
        return out

    def eligible(self, key, idx):
        """Tests that Stage 07 publishes in a solution: at least
        MIN_N_UNFLAGGED valid answers and item non-response, among the
        household-days of the countries that collected the item and to which it
        applies, of at most 50%."""
        if key not in self._eligible:
            ok = []
            ctry = self.country[idx]
            for j, (valid, _, _) in enumerate(self.cats):
                v = valid[idx]
                n_valid = int(v.sum())
                if n_valid < MIN_N_UNFLAGGED:
                    ok.append(False)
                    continue
                collected = np.isin(ctry, np.unique(ctry[v]))
                applies = collected & (self.features[idx, j] != NOT_APPLICABLE)
                ok.append(100 * (applies.sum() - n_valid) / applies.sum() <= 50)
            self._eligible[key] = np.array(ok)
        return self._eligible[key]

    def tests(self, idx, labels, weight, eligible):
        """Plain and bias-corrected Cramer's V of every published test of a
        solution (NaN where Stage 07 would publish none)."""
        k = int(labels.max()) + 1
        plain = np.full(len(self.cats), np.nan)
        corrected = np.full(len(self.cats), np.nan)
        for j, (valid, inv, ncat) in enumerate(self.cats):
            if not eligible[j]:
                continue
            v = valid[idx]
            lab, cat = labels[v], inv[idx][v]
            w = None
            if weight is not None:
                w = weight[idx][v]
                w = w * (len(w) / w.sum())
            table = np.bincount(lab * ncat + cat, weights=w, minlength=k * ncat).reshape(k, ncat)
            chi2, n, shape = chi2_table(table)
            if not np.isfinite(chi2):
                continue
            plain[j] = np.sqrt(chi2 / (n * (min(shape) - 1)))
            corrected[j] = bias_corrected_v(chi2, n, shape[0], shape[1])
        return plain, corrected

    def strongest(self, v):
        """The strongest correlate of a solution, country aside."""
        vals = v[self.nonc]
        return self.feature_names[self.nonc[int(np.nanargmax(vals))]] if np.isfinite(vals).any() else None

    # ----------------------------------------------------------------- outputs
    def mean_profiles(self, p, day_type, weight):
        """Mean profiles (countries + pooled, x working, non-working, all days,
        x 144) over the household-days whose day type is known, as in Stage
        06, with their numbers of household-days."""
        nc = len(self.countries)
        ci = np.searchsorted(np.array(self.countries), self.country)
        known = day_type >= 0
        dt = np.where(known, day_type, 0)
        groups = np.vstack([np.where(known, ci * 3 + dt, -1), np.where(known, ci * 3 + 2, -1),
                            np.where(known, nc * 3 + dt, -1), np.where(known, nc * 3 + 2, -1)])
        ng = (nc + 1) * 3
        G = indicator(groups, ng, 1.0 if weight is None else weight)
        C = indicator(groups, ng, 1.0)
        sums = np.asarray(G.sum(axis=0)).ravel()
        counts = np.asarray(C.sum(axis=0)).ravel().round().astype(np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = np.asarray(G.T @ p) / sums[:, None]
        prof[counts == 0] = np.nan
        return prof.reshape(nc + 1, 3, SLOTS), counts.reshape(nc + 1, 3), G, sums

    def run(self, x, detail=False, progress=None):
        """Outputs, mean profiles and their numbers of household-days for the
        factor values x. With detail, also the labels and tests of every
        solution (for gsa.check); progress(name, K) is called after each
        solution."""
        weight = self.weight[x["X15"]]
        w1 = np.ones(self.n) if weight is None else weight
        day_type = self.day_type[x["X14"]]
        p = self.profiles(x)
        energy = p.sum(axis=1) * HOURS_PER_SLOT
        day_max = p.max(axis=1)
        out = {}

        # mean profiles and the comparison of the day types (Stage 06)
        prof, counts, G, sums = self.mean_profiles(p, day_type, weight)
        nc = len(self.countries)
        own = (np.asarray(G.T @ day_max) / np.where(sums > 0, sums, 1)).reshape(nc + 1, 3)
        coinc_change, own_change = [], []
        for i in range(nc):
            if counts[i, 0] >= MIN_N_UNFLAGGED and counts[i, 1] >= MIN_N_UNFLAGGED:
                wc, nwc = prof[i, 0].max(), prof[i, 1].max()
                coinc_change.append(100 * (nwc / wc - 1) if wc > 0 else 0.0)
                a, b = own[i, 0], own[i, 1]
                own_change.append(100 * (b - a) / a if a > 0 else 0.0)
        out["Y3"] = float(np.median(coinc_change)) if coinc_change else np.nan
        out["Y4"] = float(np.max(np.abs(own_change))) if own_change else np.nan
        single = np.isin(self.country, [COUNTRIES.index(c) for c in SINGLE_DIARIST])
        out["Y7"] = (float(np.average(energy[single], weights=w1[single])
                           / np.average(energy[~single], weights=w1[~single]))
                     if single.any() and (~single).any() else np.nan)
        pos = {COUNTRIES[c]: i for i, c in enumerate(self.countries)}
        be, fr = pos.get("BE"), pos.get("FR")
        ok = lambda i: i is not None and counts[i, 2] >= MIN_N_UNFLAGGED  # noqa: E731
        out["Y9"] = benchmarks.shape_correlation(prof[be, 2], benchmarks.SYNERGRID_S21_BE) if ok(be) else np.nan
        out["Y10"] = benchmarks.shape_correlation(prof[fr, 2], benchmarks.ENEDIS_RES1_FR) if ok(fr) else np.nan
        out["Y11"] = benchmarks.evening_to_midday(prof[fr, 2]) if ok(fr) else np.nan

        # global solution (Stages 05, 07 and 08)
        labels, k, k_fallback, n_comp = self.cluster(p, x["X16"], x["X18"], x["X21"], weight,
                                                     GLOBAL_K_RANGE, GLOBAL_K_FALLBACK)
        if progress:
            progress("clustering_results.csv", k)
        groups, merged = option_a(labels)
        ids, gi = np.unique(groups, return_inverse=True)
        gi = gi.ravel()
        Gg = indicator(gi, len(ids), w1)
        gw = np.asarray(Gg.sum(axis=0)).ravel()
        share = gw / gw.sum()
        mean_energy = np.asarray(Gg.T @ energy).ravel() / gw
        slot_w, n_days = group_peak_weights(p, float(x["X12"]), gi, len(ids), w1)
        sd = circular_sd_hours(slot_w, n_days)
        out["Y1"] = float(share[np.argmin(mean_energy)])
        out["Y2"] = float(share[sd <= SD_THRESHOLD_H].sum())
        fin = np.isfinite(sd)
        out["Y2b"] = float((share[fin] * sd[fin]).sum() / share[fin].sum()) if fin.any() else np.nan
        everyone = np.arange(self.n)
        plain, corrected = self.tests(everyone, labels, weight, self.eligible("global", everyone))
        vals = corrected[self.nonc]
        out["Y5"] = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
        out["Y5_variable"] = self.strongest(corrected)
        ref = self.reference >= 0
        out["Y8"] = float(adjusted_rand_score(self.reference[ref], labels[ref])) if ref.any() else np.nan
        strongest = [self.strongest(plain)]
        details = {"clustering_results.csv": {"idx": everyone, "labels": labels, "k": k,
                                              "fallback": k_fallback, "plain": plain,
                                              "corrected": corrected}}

        # stratified solutions (Stages 06 and 07)
        fallbacks = 0
        for name, idx in self.solutions(day_type).items():
            max_k = min(10, len(idx) // 10)
            k_s = x["X17"] if x["X17"] == "knee" else min(int(x["X17"]), max_k)
            wi = None if weight is None else weight[idx]
            lab, k_s, fb, _ = self.cluster(p[idx], k_s, x["X18"], x["X21"], wi,
                                           range(2, max_k + 1), min(4, max_k))
            fallbacks += fb
            if progress:
                progress(name, k_s)
            pl, co = self.tests(idx, lab, weight, self.eligible((x["X14"], name), idx))
            strongest.append(self.strongest(pl))
            if detail:
                details[name] = {"idx": idx, "labels": lab, "k": k_s, "fallback": fb, "plain": pl,
                                 "corrected": co}
        found = [s for s in strongest if s is not None]
        out["Y6"] = int(sum(s == "HHC5" for s in found))
        out["solutions"] = len(found)
        out["k_global"] = k
        out["k_fallbacks"] = int(fallbacks + k_fallback)
        out["pca_components"] = n_comp
        out["clusters_merged"] = int(merged)
        out["mean_daily_energy_wh"] = float(np.average(energy, weights=w1))
        if detail:
            out["detail"] = {"solutions": details, "groups": ids, "group_sd_h": sd,
                             "group_share": share, "group_mean_energy_wh": mean_energy,
                             "coincident_change": coinc_change, "own_change": own_change,
                             "countries": [COUNTRIES[c] for c in self.countries]}
        return out, prof, counts

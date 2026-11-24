"""LaTeX tables of the sensitivity analysis from the files of gsa.analyse.

    python3 -m gsa.tables --results FOLDER --out FOLDER

Writes one .tex file per table (booktabs, caption above the table):

  tab_outputs.tex      every output: canonical value, its percentile among
                       the runs, median and 5th to 95th percentile of the runs
  tab_dependence.tex   dependence between the columns of the sample at each
                       nested size: pairs above 0.1 and the largest pairs
  tab_indices.tex      for every output, the players read as non-zero: the
                       surrogate's Shapley effect with its interval and the
                       adjusted first-order and OT indices (longtable)
  tab_convergence.tex  statistics of Sarrazin et al. (2016) for the adjusted
                       indices at the largest size, and the largest change of
                       a Shapley effect of the surrogate between them
  tab_uniform.tex      the 256 runs with powers uniform in W against the same
                       points with log-uniform powers
  tab_knn.tex          nearest-neighbour against surrogate Shapley effects
  tab_adjustment.tex   the indices that the adjustment moves by more than 0.03 at
                       1,024 runs or more
  tab_stability.tex    screening at 0.05 and ranking from one nested size to the
                       next, for the last two steps
  tab_distributions.tex  the uncertainty analysis: moments and shape of every
                       output over the runs
  tab_ot_agreement.tex rankings of the OT and first-order indices compared
  tab_ot_excess.tex    the factors whose OT index exceeds that of a pure shift
                       of a Gaussian output, with the parts of the index
  tab_budget.tex       variance budget of every output (needs budget.csv)
  tab_brackets.tex     first-order index, Shapley effect and total index of
                       the surrogate, the appliance powers one by one and as a
                       group (longtable)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsa.figures import OUTPUT_LABEL, PLAYER_LABEL, ROW_ORDER, is_true  # noqa: E402

OUTPUTS = list(OUTPUT_LABEL)
UNITS = {"Y2b": "h", "Y3": "\\%", "Y4": "\\%"}
CRITERIA = {"ranking": 1.0, "screening": 0.05, "indices": 0.05, "max_change": 0.02}


def largest_n(results):
    """Number of runs of the largest nested design in the results."""
    return int(pd.read_csv(results / "indices.csv", usecols=["N"]).N.max())


def num(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:,.{d}f}".replace("-", "$-$") if x < 0 else f"{x:,.{d}f}"


def tex(text):
    return str(text).replace("_", "\\_")


def label(p):
    return PLAYER_LABEL.get(p, p).strip()


def wrap(body, caption, lab, cols, long=False, size="\\small", fit=False):
    """A table with its caption above; fit scales the tabular to the text width."""
    if long:
        return (f"{{{size}\n\\begin{{longtable}}{{{cols}}}\n\\caption{{{caption}}}\\label{{{lab}}}\\\\\n"
                f"{body}\\end{{longtable}}}}\n")
    tab = f"\\begin{{tabular}}{{{cols}}}\n{body}\\end{{tabular}}"
    if fit:
        tab = f"\\resizebox{{\\textwidth}}{{!}}{{%\n{tab}}}"
    return (f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\\label{{{lab}}}\n{size}\n"
            f"{tab}\n\\end{{table}}\n")


def table_outputs(results):
    s = pd.read_csv(results / "summary.csv").set_index("output")
    body = ("\\toprule\nOutput & Canonical & Percentile & Median & 5th to 95th \\\\\n"
            " &  & of canonical &  & percentile \\\\\n\\midrule\n")
    for y in OUTPUTS:
        r = s.loc[y]
        d = 2 if y in ("Y2b", "Y3", "Y4", "Y6") else 3
        unit = f" ({UNITS[y]})" if y in UNITS else ""
        pct = num(r.get("canonical_percentile", np.nan), 0)
        body += (f"{OUTPUT_LABEL[y]}{unit} & {num(r.canonical, d)} & {pct} & {num(r['median'], d)} & "
                 f"{num(r.p05, d)} to {num(r.p95, d)} \\\\\n")
    body += "\\bottomrule\n"
    return wrap(body, f"Outputs at the canonical settings and over the {largest_n(results):,} runs. The "
                "percentile of the canonical value is the share of runs below it (ties counted half).",
                "tab:outputs", "lrrrr", size="\\footnotesize")


def table_dependence(results):
    d = pd.read_csv(results / "design_dependence.csv")
    body = ("\\toprule\nRuns & Classes & Independent & Pairs & Largest pairs (classes of $\\to$ "
            "column: index) \\\\\n & & & above 0.1 & \\\\\n\\midrule\n")
    for N, g in d.groupby("N"):
        top = g.sort_values("index", ascending=False).head(3)
        pairs = "; ".join(f"{tex(r.classes_of)} $\\to$ {tex(r.column)}: {r['index']:.2f}"
                          for _, r in top.iterrows())
        K = int(round(g.independent.max() * (N - 1) + 1))
        body += (f"{N:,} & {K} & {g.independent.max():.3f} & {int((g['index'] > 0.1).sum())} of {len(g)} & "
                 f"{pairs} \\\\\n")
    body += "\\bottomrule\n"
    return wrap(body, "Dependence between the columns of the sample: the first-order index of each "
                "column on the quantile classes of each other column (factors and dummies, ordered "
                "pairs). Independent columns would give about (K $-$ 1)/(N $-$ 1) with K classes.",
                "tab:dependence", "rrrrp{6.2cm}", size="\\footnotesize")


def table_indices(results):
    ind = pd.read_csv(results / "indices.csv")
    sur = pd.read_csv(results / "surrogate.csv")
    N = ind.N.max()
    ind, sur = ind[ind.N == N], sur[sur.N == sur.N.max()]
    head = ("\\toprule\nOutput and player & Shapley & 95\\% interval & $S_i$ & $\\iota_i$ \\\\\n\\midrule\n"
            "\\endfirsthead\n\\toprule\nOutput and player & Shapley & 95\\% interval & $S_i$ & $\\iota_i$ "
            "\\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n")
    body = head
    for y in OUTPUTS:
        a = ind[ind.output == y]
        s = sur[(sur.output == y)]
        sh = s[s.shapley.notna()].set_index("player")
        first = s.iloc[0]
        body += (f"\\multicolumn{{5}}{{l}}{{{OUTPUT_LABEL[y]}}} \\\\\n"
                 f"\\multicolumn{{5}}{{l}}{{\\quad surrogate: cross-validated $R^2$ {first.cv_r2:.2f}, "
                 f"reproduces {first.explained:.2f} of the variance}} \\\\\n")
        nz_s = set(a[(a.family == "S") & a.nonzero.map(is_true)].factor)
        nz_o = set(a[(a.family == "OT") & a.nonzero.map(is_true)].factor)
        nz_sh = set(sh[sh.nonzero.map(is_true)].index)
        rows = [p for p in ROW_ORDER if p in nz_sh or p in nz_s or p in nz_o]
        if not rows:
            body += "\\quad none read as non-zero & & & & \\\\\n"
        for p in rows:
            def get(fam):
                r = a[(a.family == fam) & (a.factor == p)]
                if r.empty:
                    return "--"
                v = num(r.iloc[0].point)
                return v + ("" if p in (nz_s if fam == "S" else nz_o) else "$^{\\circ}$")
            if p in sh.index:
                r = sh.loc[p]
                shv = num(r.shapley) + ("" if p in nz_sh else "$^{\\circ}$")
                ci = f"{num(r.lo)} to {num(r.hi)}"
            else:
                shv, ci = "--", "--"
            body += f"\\quad {label(p)} & {shv} & {ci} & {get('S')} & {get('OT')} \\\\\n"
        body += "\\addlinespace\n"
    return wrap(body, f"Indices at {N:,} runs for the players read as non-zero in at least one family: "
                "Shapley effects of the surrogate as shares of Var $Y$ with the 2.5th and 97.5th "
                "percentiles of 50 bootstrap refits, and the adjusted first-order ($S_i$) and OT "
                "($\\iota_i$) indices. $^{\\circ}$: not read as non-zero in that family. The group of "
                "the appliance powers has a Shapley effect only, its factors first-order and OT "
                "indices only.", "tab:indices", "lrrrr", long=True)


def table_convergence(results):
    c = pd.read_csv(results / "convergence_indices.csv")
    sur = pd.read_csv(results / "surrogate.csv")
    sur = sur[sur.shapley.notna() & (sur.player != "D_cont")]
    sizes = sorted(sur.N.unique())
    N = c.N.max()
    prev_n = sorted(c.N.unique())[-2]
    body = (f"\\toprule\n & \\multicolumn{{4}}{{c}}{{$S_i$ at {N:,} runs}} & \\multicolumn{{4}}{{c}}{{$\\iota_i$ at "
            f"{N:,} runs}} & Shapley \\\\\n\\cmidrule(lr){{2-5}}\\cmidrule(lr){{6-9}}\\cmidrule(lr){{10-10}}\n"
            "Output & Rank. & Screen. & Indices & Change & Rank. & Screen. & Indices & Change & Change \\\\\n"
            "\\midrule\n")
    for y in OUTPUTS:
        cells = []
        for fam in ("S", "OT"):
            r = c[(c.output == y) & (c.family == fam) & (c.N == N)].iloc[0]
            for k in ("ranking", "screening", "indices", "max_change"):
                v = r[k]
                ok = np.isfinite(v) and v < CRITERIA[k]
                cells.append(num(v, 3) + ("" if ok else "$^{*}$"))
        a = sur[sur.output == y].pivot(index="player", columns="N", values="shapley")
        ch = float((a[sizes[-1]] - a[sizes[-2]]).abs().max()) if len(sizes) > 1 else np.nan
        cells.append(num(ch, 3) + ("" if ch < CRITERIA["max_change"] else "$^{*}$"))
        body += f"{y} & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n"
    return wrap(body, f"Convergence at {N:,} runs. Ranking is the 95th percentile of the adjusted "
                "rank difference over the bootstrap resamples (criterion below 1). Screening and "
                "indices are the widest 95\\% intervals among the factors read as zero and as "
                f"non-zero (below 0.05). Change is the largest change of an index from {prev_n:,} to "
                f"{N:,} runs (below 0.02), for the Shapley effects of the surrogate too. $^{{*}}$: "
                "criterion not met.",
                "tab:convergence", "lrrrrrrrrr", size="\\footnotesize")


def table_uniform(results):
    u = pd.read_csv(results / "uniform.csv")
    s = u[u.factor == "_summary"].set_index(["sample", "output"])
    sh = u[u.Shapley.notna()].set_index(["sample", "output", "factor"]).Shapley
    body = ("\\toprule\n & \\multicolumn{3}{c}{Log-uniform powers} & \\multicolumn{3}{c}{Powers uniform "
            "in W} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nOutput & Median & 5th to 95th & "
            "Sh, powers & Median & 5th to 95th & Sh, powers \\\\\n\\midrule\n")
    for y in OUTPUTS:
        d = 2 if y in ("Y2b", "Y3", "Y4", "Y6") else 3
        cells = []
        for name in ("log-uniform", "uniform"):
            r = s.loc[(name, y)]
            cells += [num(r["median"], d),
                      f"{num(r.p05, d)} to {num(r.p95, d)}",
                      num(sh.get((name, y, "P_appliances"), np.nan), 2)]
        body += f"{y} & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n"
    return wrap(body, "The 256 runs with powers uniform in W against the same 256 points with "
                "log-uniform powers: median and 5th to 95th percentile of each output and the "
                "Shapley effect of the appliance powers in the surrogate of each sample.",
                "tab:uniform", "lrrrrrr", size="\\footnotesize")


def table_knn(results):
    k = pd.read_csv(results / "knn_shapley.csv")
    sur = pd.read_csv(results / "surrogate.csv")
    N = k.N.max()
    k1 = k[(k.N == N) & (k.step == 1)].set_index(["output", "player"]).point
    k2 = k[(k.N == N) & (k.step == 2)].set_index(["output", "player"]).point
    # the surrogate at the same size as the nearest-neighbour estimates
    n_s = N if N in set(sur.N) else sur.N.max()
    s = sur[(sur.N == n_s) & sur.shapley.notna()].set_index(["output", "player"]).shapley
    players = ["P_appliances", "X13", "X19", "X16", "X15", "X12", "X10", "X17"]
    short = {"P_appliances": "Powers", "X13": "Coverage", "X19": "Lighting", "X16": "K, global",
             "X15": "Weight", "X12": "Base load", "X10": "Ownership", "X17": "K, strat."}
    body = ("\\toprule\nOutput & " + " & ".join(short[p] for p in players) + " \\\\\n & " +
            " & ".join(p if p != "P_appliances" else "X1-X6" for p in players) + " \\\\\n\\midrule\n")
    for y in OUTPUTS:
        cells = []
        for p in players:
            sv, kv = s.get((y, p), np.nan), k2.get((y, p), k1.get((y, p), np.nan))
            cells.append(f"{num(sv, 2)} / {num(kv, 2)}")
        body += f"{y} & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n"
    return wrap(body, f"Shapley effects at {N:,} runs: surrogate / nearest neighbours (step 2 where "
                "the player was retained, step 1 otherwise).", "tab:knn",
                "l" + "r" * len(players), size="\\small", fit=True)


def stability_counts(a, b, threshold=0.05):
    """Screening and ranking from one size (a) to the next (b), two Series of
    indices on the same factors: crossings of the threshold, whether the
    order of the factors above it at both sizes is kept, and whether the top
    two and top five keep their members and order."""
    a = a.reindex(b.index)
    up = int(((a < threshold) & (b >= threshold)).sum())
    down = int(((a >= threshold) & (b < threshold)).sum())
    both = [f for f in b.sort_values(ascending=False).index if a[f] >= threshold and b[f] >= threshold]
    kept = list(a.loc[both].sort_values(ascending=False).index) == both
    ra = list(a.sort_values(ascending=False).index)
    rb = list(b.sort_values(ascending=False).index)
    return up, down, kept, ra[:2] == rb[:2], ra[:5] == rb[:5]


def table_stability(results, threshold=0.05):
    """For the last two steps between nested sizes, by family: indices that
    cross the threshold either way, and the outputs (of twelve) whose order
    above the threshold, top two and top five are kept."""
    ind = pd.read_csv(results / "indices.csv")
    sur = pd.read_csv(results / "surrogate.csv")
    factors = [f"X{i}" for i in range(1, 22)]
    sizes = sorted(set(ind.N) & set(sur.N))
    steps = list(zip(sizes[:-1], sizes[1:]))[-2:]

    def get(fam, y, N):
        if fam == "Shapley":
            x = sur[(sur.N == N) & (sur.output == y) & sur.shapley.notna() & (sur.player != "D_cont")]
            return x.set_index("player").shapley
        x = ind[(ind.N == N) & (ind.family == fam) & (ind.output == y) & ind.factor.isin(factors)]
        return x.set_index("factor").point

    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n0:,} to {n1:,} runs}}" for n0, n1 in steps)
    rules = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(steps)))
    body = (f"\\toprule\n & {head} \\\\\n{rules}\n"
            " & " + " & ".join(["First-order & OT & Shapley"] * len(steps)) + " \\\\\n\\midrule\n")
    labels = [f"Below {threshold:.2f}, then at or above it", f"At or above {threshold:.2f}, then below it",
              f"Order above {threshold:.2f} kept (outputs)", "Top two kept (outputs)",
              "Top five kept (outputs)"]
    cells = {k: [] for k in range(5)}
    for n0, n1 in steps:
        for fam in ("S", "OT", "Shapley"):
            counts = [stability_counts(get(fam, y, n0), get(fam, y, n1), threshold) for y in OUTPUTS]
            cells[0].append(sum(c[0] for c in counts))
            cells[1].append(sum(c[1] for c in counts))
            for k in (2, 3, 4):
                cells[k].append(sum(bool(c[k]) for c in counts))
    for k, lab in enumerate(labels):
        body += lab + " & " + " & ".join(str(v) for v in cells[k]) + " \\\\\n"
    body += "\\bottomrule\n"
    players = sur[(sur.N == sizes[-1]) & sur.shapley.notna() & (sur.player != "D_cont")].player.nunique()
    return wrap(body, f"Screening at {threshold:.2f} and ranking from one nested size to the next. "
                f"First-order and OT indices on the 21 factors one by one, Shapley effects on the {players} "
                "players other than the dummy (the six powers as one). Counts of indices over the twelve "
                "outputs, then the number of outputs whose order is kept.", "tab:stability",
                "l" + "rrr" * len(steps), size="\\footnotesize")


def table_adjustment(results):
    """Indices that the adjustment moves by more than 0.03 at any size of at
    least 1,024 runs (the last three at most), raw and adjusted at each."""
    ind = pd.read_csv(results / "indices.csv")
    ind = ind[~ind.factor.str.startswith("D_")]
    sizes = [N for N in sorted(ind.N.unique()) if N >= 1024][-3:] or sorted(ind.N.unique())[-2:]
    keyed = {N: ind[ind.N == N].set_index(["factor", "output", "family"]) for N in sizes}
    moved = sorted({k for N in sizes for k, r in keyed[N].iterrows() if abs(r.raw - r.point) > 0.03})
    groups = " & ".join(f"\\multicolumn{{2}}{{c}}{{{N:,} runs}}" for N in sizes)
    rules = "".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(sizes)))
    head = " & ".join("Raw & Adjusted" for N in sizes)
    body = (f"\\toprule\n & & & {groups} \\\\\n{rules}\n"
            f"Factor & Output & Family & {head} \\\\\n\\midrule\n")
    for f, y, fam in moved:
        cells = " & ".join(f"{num(keyed[N].loc[(f, y, fam)].raw)} & {num(keyed[N].loc[(f, y, fam)].point)}"
                           for N in sizes)
        fam_tex = "$S_i$" if fam == "S" else "$\\iota_i$"
        body += f"{label(f)} & {y} & {fam_tex} & {cells} \\\\\n"
    if not moved:
        body += f"\\multicolumn{{{3 + 2 * len(sizes)}}}{{l}}{{none}} \\\\\n"
    body += "\\bottomrule\n"
    caption = ("Indices that the adjustment moves by more than 0.03 at "
               + (", ".join(f"{N:,}" for N in sizes[:-1]) + " or " if len(sizes) > 1 else "")
               + f"{sizes[-1]:,} runs, raw and adjusted at each size.")
    return wrap(body, caption, "tab:adjustment", "lll" + "rr" * len(sizes), size="\\footnotesize")


SHORT = {"Y1": "Lowest-energy share", "Y2": "Predictable-peak share", "Y2b": "Peak-time SD (h)",
         "Y3": "Coincident peak (\\%)", "Y4": "Own peak (\\%)", "Y5": "Strongest $V$",
         "Y6": "Solutions led by adults", "Y7": "Single/multi energy", "Y8": "Agreement (ARI)",
         "Y9": "Shape, Belgium", "Y10": "Shape, France", "Y11": "Evening/midday, France"}


def ordinal(k):
    k = int(round(k))
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def table_distributions(results):
    s = pd.read_csv(results / "summary.csv").set_index("output")
    body = ("\\toprule\nOutput & Canonical & Median & 5th to 95th & Mean & SD & Skew- & Excess & BC & Spike \\\\\n"
            " & (percentile) & & percentile & & & ness & kurtosis & & \\\\\n\\midrule\n")
    for y in OUTPUTS:
        r = s.loc[y]
        d = 0 if y == "Y6" else 2 if y in ("Y2b", "Y3", "Y4") else 3
        spike = f"{100 * r.mode_share:.0f}\\% at {r['mode']:g}" if r.mode_share > 0.2 else ""
        body += (f"{y} {SHORT[y]} & {num(r.canonical, d)} ({ordinal(r.canonical_percentile)}) & "
                 f"{num(r['median'], d)} & "
                 f"{num(r.p05, d)} to {num(r.p95, d)} & {num(r['mean'], d)} & {num(r.sd, d if d else 1)} & "
                 f"{num(r.skewness, 2)} & {num(r.excess_kurtosis, 2)} & {num(r.bimodality, 2)} & {spike} \\\\\n")
    body += "\\bottomrule\n"
    return wrap(body, "Uncertainty analysis: every output over the runs of the sample. Skewness and "
                "excess kurtosis bias-corrected. BC is the bimodality coefficient of Pfister et al. (2013), "
                "above 0.555 for a bimodal or strongly skewed distribution. Spike: share of the runs at one "
                "value, where it exceeds a fifth.", "tab:distributions", "lrrrrrrrrl", size="\\footnotesize",
                fit=True)


def table_ot_agreement(results):
    from gsa.analyse import top_down_correlation
    from scipy.stats import spearmanr
    d = pd.read_csv(results / "ot_parts.csv")
    body = ("\\toprule\nOutput & Spearman & Top-down & Leading by S & Leading by OT & Top four \\\\\n"
            " & correlation & correlation & & & in the same order \\\\\n\\midrule\n")
    for y in OUTPUTS:
        a = d[d.output == y]
        s, o = a.S.to_numpy(), a.OT.to_numpy()
        f = a.factor.to_numpy()
        top_s, top_o = f[np.argsort(-s)][:4], f[np.argsort(-o)][:4]
        body += (f"{y} {SHORT[y]} & {spearmanr(s, o)[0]:.2f} & {top_down_correlation(s, o):.2f} & "
                 f"{label(top_s[0])} & {label(top_o[0])} & {'yes' if list(top_s) == list(top_o) else 'no'} \\\\\n")
    body += "\\bottomrule\n"
    return wrap(body, "Rankings of the 21 factors by the adjusted first-order and OT indices at the largest "
                "size. The top-down correlation (Iman and Conover 1987) weighs the leading factors most.",
                "tab:ot_agreement", "lrrllc", size="\\footnotesize", fit=True)


def table_ot_excess(results, excess=0.04):
    d = pd.read_csv(results / "ot_parts.csv")
    d = d[d.OT - d.location_OT > excess].copy()
    d["o"] = d.output.map({y: i for i, y in enumerate(OUTPUTS)})
    d = d.sort_values(["o", "OT"], ascending=[True, False])
    body = ("\\toprule\nOutput & Factor & S & OT & OT of & \\multicolumn{3}{c}{Parts of the OT index} \\\\\n"
            "\\cmidrule(lr){6-8}\n & & & & a shift & Mean & Spread & Shape \\\\\n\\midrule\n")
    for _, r in d.iterrows():
        body += (f"{r.output} & {label(r.factor)} & {num(r.S)} & {num(r.OT)} & {num(r.location_OT)} & "
                 f"{num(r['mean'])} & {num(r['spread'])} & {num(r['shape'])} \\\\\n")
    body += "\\bottomrule\n"
    return wrap(body, f"Factors whose adjusted OT index exceeds by more than {excess:.2f} the OT index of a "
                "factor that only shifts a Gaussian output with the same first-order index, "
                "$(S + (1 - \\sqrt{1 - S})^2)/2$. The parts follow from "
                "$W_2^2 = (m_c - m)^2 + (s_c - s)^2 + 2 s_c s (1 - r_c)$ for the mean $m$, standard deviation "
                "$s$ and correlation $r_c$ of the quantile functions of a class and of the whole sample, each "
                "averaged over the classes and divided by $2\\,\\mathrm{Var}\\,Y$. The mean part is $S/2$.",
                "tab:ot_excess", "llrrrrrr", size="\\footnotesize")


def table_brackets(results, floor=0.05):
    b = pd.read_csv(results / "brackets.csv")
    ind = pd.read_csv(results / "indices.csv")
    N = ind.N.max()
    adj = ind[(ind.N == N) & (ind.family == "S")].set_index(["output", "factor"]).point
    head = ("\\toprule\nOutput & Player & S, given & S & Shapley & T \\\\\n"
            " & & data & & effect & \\\\\n\\midrule\n")
    body = head + "\\endfirsthead\n" + head + "\\endhead\n"
    for y in OUTPUTS:
        a = b[(b.output == y) & (b.game == "single") & (b.player != "D_cont")]
        g = b[(b.output == y) & (b.game == "grouped") & (b.player == "P_appliances")].iloc[0]
        rows = a[(a["T"] >= floor) | (a.shapley >= floor)].sort_values("shapley", ascending=False)
        first = True
        for _, r in rows.iterrows():
            body += (f"{y if first else ''} & {label(r.player)} & {num(adj[(y, r.player)])} & {num(r.S)} & "
                     f"{num(r.shapley)} & {num(r['T'])} \\\\\n")
            first = False
        pw = a[a.player.isin(["X1", "X2", "X3", "X4", "X5", "X6"])]
        sum_given = sum(adj[(y, f"X{i}")] for i in range(1, 7))
        body += (f"{y if first else ''} & Powers, sum of the six & {num(sum_given)} & {num(pw.S.sum())} & "
                 f"{num(pw.shapley.sum())} & {num(pw['T'].sum())} \\\\\n"
                 f" & Powers as one player & -- & {num(g.S)} & {num(g.shapley)} & {num(g['T'])} \\\\\n"
                 f" & Reproduced by the surrogate & & & {num(a.explained.iloc[0])} & \\\\\n")
        body += "\\midrule\n" if y != OUTPUTS[-1] else ""
    body += "\\bottomrule\n"
    return wrap(body, f"First-order index S, Shapley effect and total index T of the surrogate on independent "
                f"points, as shares of Var Y, for the players with an effect or total index of at least "
                f"{floor:.2f}, the appliance powers one by one, with the first-order index of the given-data "
                "estimator beside them. For the six powers together: the sums of their indices, and the "
                "closed indices and Shapley effect of the group as one player. For independent inputs "
                "$S \\le \\mathrm{Sh} \\le T$.", "tab:brackets", "llrrrr", long=True,
                size="\\footnotesize")


def table_budget(results):
    d = pd.read_csv(results / "budget.csv").set_index("output")
    split = "kmeans" in d.columns
    head = ("\\toprule\nOutput & First-order & Interactions & "
            + ("K-means & Country & " if split else "")
            + "Seeds & Not \\\\\n & effects & & " + ("seed & multipliers & " if split else "")
            + "together & reproduced \\\\\n\\midrule\n")
    body = head
    for y in [y for y in OUTPUTS if y in d.index]:
        r = d.loc[y]
        seeds = f"{num(r.seeds, 2)} ({num(r.seeds_lo, 2)} to {num(r.seeds_hi, 2)})"
        body += (f"{y} {SHORT[y]} & {num(r.first_order, 2)} & {num(r.interactions, 2)} & "
                 + (f"{num(r.kmeans, 2)} & {num(r.multipliers, 2)} & " if split else "")
                 + f"{seeds} & {num(r.error, 2)} \\\\\n")
    body += "\\bottomrule\n"
    return wrap(body, "Variance budget of every output, as shares of Var Y. First-order effects and "
                "interactions of the factors other than the seeds from the surrogate (the sum of its "
                "first-order indices, and its Shapley effects less that sum), the total index of the seeds "
                "on the model from the reruns with the seeds redrawn (Jansen's estimator, 95\\% bootstrap "
                "interval), and what the surrogate does not reproduce by difference.", "tab:budget",
                "lrr" + ("rr" if split else "") + "rr", size="\\footnotesize", fit=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    r = args.results
    for name, f in (("outputs", table_outputs), ("dependence", table_dependence),
                    ("indices", table_indices), ("convergence", table_convergence),
                    ("uniform", table_uniform), ("knn", table_knn), ("adjustment", table_adjustment),
                    ("stability", table_stability), ("distributions", table_distributions),
                    ("ot_agreement", table_ot_agreement), ("ot_excess", table_ot_excess),
                    ("brackets", table_brackets), ("budget", table_budget)):
        try:
            (args.out / f"tab_{name}.tex").write_text(f(r))
        except (FileNotFoundError, KeyError, AttributeError) as e:
            print(f"tab_{name}: {e}")
    print(f"Tables written to {args.out}")


if __name__ == "__main__":
    main()

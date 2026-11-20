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
                       indices at 512 and 1,024 runs, and the largest change of
                       a Shapley effect of the surrogate between them
  tab_uniform.tex      the 256 runs with powers uniform in W against the same
                       points with log-uniform powers
  tab_knn.tex          nearest-neighbour against surrogate Shapley effects
  tab_adjustment.tex   the indices that the adjustment moves by more than 0.03
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
    return wrap(body, "Outputs at the canonical settings and over the 1,024 runs. The percentile of "
                "the canonical value is the share of runs below it (ties counted half).",
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
    body = ("\\toprule\n & \\multicolumn{4}{c}{$S_i$ at 1,024 runs} & \\multicolumn{4}{c}{$\\iota_i$ at "
            "1,024 runs} & Shapley \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-10}\n"
            "Output & Rank. & Screen. & Indices & Change & Rank. & Screen. & Indices & Change & Change \\\\\n"
            "\\midrule\n")
    N = c.N.max()
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
    return wrap(body, "Convergence at 1,024 runs. Ranking is the 95th percentile of the adjusted "
                "rank difference over the bootstrap resamples (criterion below 1). Screening and "
                "indices are the widest 95\\% intervals among the factors read as zero and as "
                "non-zero (below 0.05). Change is the largest change of an index from 512 to 1,024 "
                "runs (below 0.02), for the Shapley effects of the surrogate too. $^{*}$: criterion "
                "not met.",
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
    s = sur[(sur.N == sur.N.max()) & sur.shapley.notna()].set_index(["output", "player"]).shapley
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
    return wrap(body, "Shapley effects at 1,024 runs: surrogate / nearest neighbours (step 2 where "
                "the player was retained, step 1 otherwise).", "tab:knn",
                "l" + "r" * len(players), size="\\small", fit=True)


def table_adjustment(results):
    ind = pd.read_csv(results / "indices.csv")
    a = ind[(ind.N == ind.N.max()) & ~ind.factor.str.startswith("D_")]
    a = a[(a.raw - a.point).abs() > 0.03].sort_values(["factor", "output", "family"])
    body = ("\\toprule\nFactor & Output & Family & Raw & Adjusted & Leading drivers & Imbalance left \\\\\n"
            "\\midrule\n")
    for _, r in a.iterrows():
        fam = "$S_i$" if r.family == "S" else "$\\iota_i$"
        body += (f"{label(r.factor)} & {r.output} & {fam} & {num(r.raw)} & {num(r.point)} & "
                 f"{r.drivers} & {num(r.imbalance, 2)} \\\\\n")
    if a.empty:
        body += "\\multicolumn{7}{l}{none} \\\\\n"
    body += "\\bottomrule\n"
    return wrap(body, "Indices at 1,024 runs that the adjustment moves by more than 0.03.",
                "tab:adjustment", "lllrrlr", size="\\footnotesize")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    r = args.results
    for name, f in (("outputs", table_outputs), ("dependence", table_dependence),
                    ("indices", table_indices), ("convergence", table_convergence),
                    ("uniform", table_uniform), ("knn", table_knn), ("adjustment", table_adjustment)):
        try:
            (args.out / f"tab_{name}.tex").write_text(f(r))
        except FileNotFoundError as e:
            print(f"tab_{name}: {e}")
    print(f"Tables written to {args.out}")


if __name__ == "__main__":
    main()

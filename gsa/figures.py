"""Figures of the sensitivity analysis from the files of gsa.analyse.

    python3 -m gsa.figures --results FOLDER --out FOLDER

Draws, as PDF:

  fig_effects.pdf      Shapley effects of the surrogate on every output at the
                       largest size, as shares of Var Y (blank where an effect
                       does not clear the dummy), and the share of Var Y the
                       surrogate does not reproduce
  fig_indices.pdf      first-order and OT indices and the surrogate's total
                       indices (Shapley effects without totals.csv) with their
                       95% intervals, one panel per output, for the factors
                       read as non-zero and of at least 0.02 in some family
  fig_adjustment.pdf   (only for adjusted indices) raw against adjusted
                       first-order and OT indices at 1,024 runs and at the
                       largest size
  fig_convergence.pdf  total indices of the surrogate (Shapley effects
                       without totals.csv) of every factor against the number
                       of runs, square-root scale, the factors of 0.05 or more
                       labelled
  fig_convergence_s.pdf, fig_convergence_ot.pdf  the same for the first-order
                       and OT indices of the runs (all 21 factors)
  fig_profiles.pdf     the pooled profile by day type and the hourly shapes of
                       Belgium and France against their measured benchmarks:
                       median and 5th to 95th percentile over the runs
  fig_hourly.pdf       OT index of each factor on the hourly values of
                       the pooled profile (blank below the dummy)
  fig_drivers.pdf      outputs of the runs against their leading factors, with
                       the canonical value (needs --share, the folder of
                       gsa_outputs.csv)
  fig_distributions.pdf  the uncertainty analysis: the distribution of every
                       output over the runs, split by diarist coverage, with
                       the canonical value (needs --share)
  fig_ot_s.pdf         OT index against first-order index of every
                       factor and output, with the bound S/2 and the curve of
                       a pure shift of a Gaussian output (needs ot_parts.csv)
  fig_budget.pdf       variance budget of every output from the model: the
                       factors' first-order effects (runs) and interactions
                       (by difference, the part the surrogate apportions
                       shown apart) and the seeds (needs budget.csv, from the
                       reruns of gsa.reseed)
  fig_totals.pdf       total index of every factor (surrogate) with the
                       first-order index below it, and the total index of the
                       seeds from the reruns (needs totals.csv; seeds.csv)
  fig_brackets.pdf     first-order index, Shapley effect and total index of
                       the surrogate for the leading players of every output,
                       the appliance powers one by one, and the share of Var Y that the
                       surrogate reproduces (needs brackets.csv)

Colours follow the validated reference palette (one blue ramp for
magnitudes; blue, orange and aqua as categorical slots, which pass the
colour-vision checks as a set of three, with a legend or a direct label for
every series).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from gsa import benchmarks  # noqa: E402

TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAMILY = {"T": BLUE, "S": ORANGE, "OT": AQUA, "Shapley": "#e87ba4"}
FAMILY_LABEL = {"T": "Total index (surrogate)", "S": "First-order index", "OT": "OT index",
                "Shapley": "Shapley effect (surrogate)"}
PLAYER_COLOUR = {"P_appliances": BLUE, "X13": ORANGE, "X19": AQUA}
RAMP = ["#ffffff", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

OUTPUT_LABEL = {
    "Y1": "Y1 Share in the lowest-energy group", "Y2": "Y2 Share in predictable-peak groups",
    "Y2b": "Y2b Mean SD of peak time", "Y3": "Y3 Change in coincident peak",
    "Y4": "Y4 Largest change in own peak", "Y5": "Y5 Strongest association (V)",
    "Y6": "Y6 Solutions led by adults", "Y7": "Y7 Single- to multi-diarist energy",
    "Y8": "Y8 Agreement with canonical partition", "Y9": "Y9 Shape correlation, Belgium",
    "Y10": "Y10 Shape correlation, France", "Y11": "Y11 Evening to midday, France"}
SHORT_OUTPUT = {
    "Y1": "Y1 Lowest-energy share", "Y2": "Y2 Predictable peaks", "Y2b": "Y2b Peak-time SD",
    "Y3": "Y3 Coincident peak", "Y4": "Y4 Own peak", "Y5": "Y5 Strongest V",
    "Y6": "Y6 Led by adults", "Y7": "Y7 Single/multi energy", "Y8": "Y8 Agreement (ARI)",
    "Y9": "Y9 Shape, Belgium", "Y10": "Y10 Shape, France", "Y11": "Y11 Evening/midday"}
PLAYER_LABEL = {
    "P_appliances": "Appliance powers (X1-X6)", "X1": "  Cooking (X1)", "X2": "  Laundry, ironing (X2)",
    "X3": "  Cleaning (X3)", "X4": "  TV, radio (X4)", "X5": "  Computing (X5)",
    "X6": "  Household mgmt, 5 W (X6)", "X7": "Country spread (X7)", "X9": "Shared appliances (X9)",
    "X10": "Ownership (X10)", "X11": "Secondary weight (X11)", "X12": "Base load (X12)",
    "X13": "Diarist coverage (X13)", "X14": "Day type (X14)", "X15": "Survey weight (X15)",
    "X16": "K, global (X16)", "X17": "K, stratified (X17)", "X18": "PCA share (X18)",
    "X19": "Lighting (X19)", "X20": "Laundry run-on (X20)", "X8": "Multiplier seed (X8)",
    "X21": "K-means seed (X21)"}
OUTPUTS = list(OUTPUT_LABEL)
APPLIANCES = ["X1", "X2", "X3", "X4", "X5", "X6"]
ROW_ORDER = ["P_appliances"] + APPLIANCES + ["X7", "X9", "X10", "X11", "X12", "X13", "X14", "X15",
                                            "X16", "X17", "X18", "X19", "X20"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 7.5,
        "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.labelcolor": TEXT2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": TEXT2, "ytick.labelcolor": TEXT2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 6.5, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "pdf.fonttype": 42})


def ramp():
    return LinearSegmentedColormap.from_list("blue", RAMP)


def largest(df):
    return df[df.N == df.N.max()]


def is_true(v):
    return v is True or v == "True" or (isinstance(v, (bool, np.bool_)) and bool(v))


def fig_effects(sur, out):
    s = largest(sur)
    s = s[s.shapley.notna() & (s.player != "D_cont")]
    players = [p for p in ROW_ORDER if p in set(s.player)]
    E = pd.DataFrame(np.nan, index=OUTPUTS, columns=players)
    for _, r in s.iterrows():
        if is_true(r.nonzero):
            E.loc[r.output, r.player] = r.shapley
    E = E[[p for p in players if E[p].notna().any()]]
    rest = 1 - s.groupby("output").explained.first().reindex(OUTPUTS)
    vals = E.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.imshow(np.where(np.isnan(vals), 0.0, np.clip(vals, 0, 1)), cmap=ramp(), vmin=0, vmax=1,
              aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if vals[i, j] > 0.55 else TEXT)
        ax.text(vals.shape[1] - 0.15, i, f"{rest.iloc[i]:.2f}", ha="left", va="center",
                fontsize=6.5, color=TEXT2)
    ax.text(vals.shape[1] - 0.15, -0.7, "Not\nreproduced", ha="left", va="bottom", fontsize=6.5,
            color=TEXT2, fontweight="bold")
    ax.set_xticks(range(vals.shape[1]))
    ax.set_xticklabels([PLAYER_LABEL[p].strip() for p in E.columns], rotation=35, ha="right")
    ax.set_yticks(range(vals.shape[0]))
    ax.set_yticklabels([OUTPUT_LABEL[o] for o in E.index])
    ax.set_xlim(-0.5, vals.shape[1] + 0.9)
    ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_effects.pdf")
    plt.close(fig)


def fig_totals(tot, out, seeds=None, floor=0.02):
    """Total index of every factor (rows: outputs, columns: factors whose
    total index reaches `floor` and reads as non-zero on some output), from
    the surrogate on the largest size, with the first-order index of the
    surrogate below it; and, with seeds (seeds.csv of the reruns), the total
    index of the K-means seed and of the draw of the country multipliers on
    the model, after a gap."""
    t = largest(tot)
    t = t[(t.player != "D_cont") & (t.player != "P_appliances")]
    keep = [p for p in ROW_ORDER if p != "P_appliances" and
            ((t.player == p) & (t["T"] >= floor) & t.nonzero.map(is_true)).any()]
    T = t.pivot(index="output", columns="player", values="T").reindex(index=OUTPUTS, columns=keep)
    S = t.pivot(index="output", columns="player", values="S").reindex(index=OUTPUTS, columns=keep)
    cols = [PLAYER_LABEL[p].strip() for p in keep]
    vals, svals = T.to_numpy(dtype=float), S.to_numpy(dtype=float)
    if seeds is not None:
        sd = seeds.pivot(index="output", columns="variant", values="T").reindex(OUTPUTS)
        km = sd["kmeans"].to_numpy(dtype=float)
        mult = (sd["seeds"] - sd["kmeans"]).to_numpy(dtype=float)
        vals = np.column_stack([vals, np.full(len(OUTPUTS), np.nan), km, mult])
        svals = np.column_stack([svals, np.full((len(OUTPUTS), 3), np.nan)])
        cols += ["", "K-means seed (X21)", "Draw of the multipliers (X8)"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    shown = np.where(np.isnan(vals), 0.0, np.clip(vals, 0, 1))
    ax.imshow(shown, cmap=ramp(), vmin=0, vmax=1, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isfinite(vals[i, j]) or vals[i, j] < 0.005:
                continue
            dark = vals[i, j] > 0.55
            ax.text(j, i - 0.12, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="#ffffff" if dark else TEXT)
            if np.isfinite(svals[i, j]):
                ax.text(j, i + 0.24, f"{max(svals[i, j], 0):.2f}", ha="center", va="center", fontsize=4.8,
                        color="#ffffff" if dark else TEXT2)
    ax.set_xticks(range(vals.shape[1]))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(vals.shape[0]))
    ax.set_yticklabels([OUTPUT_LABEL[o] for o in OUTPUTS], fontsize=6.5)
    ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_totals.pdf")
    plt.close(fig)


def fig_indices(ind, sur, out, tot=None):
    """First-order, OT and total (or Shapley) indices with their 95% intervals
    for the factors read as non-zero with an index of at least 0.02, one panel
    per output. With tot (totals.csv) the third family is the total index of
    the surrogate, with the powers one by one; otherwise the Shapley effect."""
    a = largest(ind)
    third = "T" if tot is not None else "Shapley"
    if tot is not None:
        s = largest(tot)
        s = s[s.player != "D_cont"].rename(columns={"T": "value"})
    else:
        s = largest(sur)
        s = s[s.shapley.notna()].rename(columns={"shapley": "value"})
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 9.4))
    for ax, y in zip(axes.ravel(), OUTPUTS):
        ay, sy = a[a.output == y], s[s.output == y]
        nz = (set(ay[ay.nonzero.map(is_true) & (ay.point >= 0.02)].factor)
              | set(sy[sy.nonzero.map(is_true) & (sy.value >= 0.02)].player))
        if third == "T":
            rows = [p for p in ROW_ORDER if p in nz and p != "P_appliances"]
        else:
            rows = [p for p in ROW_ORDER if p in nz or (p == "P_appliances" and nz & set(APPLIANCES))]
        ypos = np.arange(len(rows))[::-1]
        for fam, dy in (("S", 0.24), ("OT", 0.0), (third, -0.24)):
            for yp, p in zip(ypos, rows):
                if fam == third:
                    r = sy[sy.player == p]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    pt, lo, hi = r.value, r.lo, r.hi
                else:
                    r = ay[(ay.family == fam) & (ay.factor == p)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    pt, lo, hi = r.point, r.lo, r.hi
                ax.plot([lo, hi], [yp + dy, yp + dy], color=FAMILY[fam], linewidth=1.0)
                ax.plot(pt, yp + dy, "o", color=FAMILY[fam], markersize=3.2,
                        markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels([PLAYER_LABEL[p] for p in rows], fontsize=6)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(SHORT_OUTPUT[y], loc="left")
        ax.axvline(0, color=AXIS, linewidth=0.6)
        ax.grid(axis="x")
        ax.set_xlim(-0.05, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], color=FAMILY[f], marker="o", markersize=3.2, linewidth=1.0,
                          label=FAMILY_LABEL[f]) for f in ("S", "OT", third)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.022, 1, 1))
    fig.savefig(out / "fig_indices.pdf")
    plt.close(fig)


def fig_adjustment(ind, out):
    """Raw against adjusted indices at 1,024 runs and at the largest size (one
    row each), or at the two largest sizes when there are fewer runs."""
    all_sizes = sorted(ind.N.unique())
    sizes = [1024, all_sizes[-1]] if 1024 in all_sizes and all_sizes[-1] > 1024 else all_sizes[-2:]
    fig, axes = plt.subplots(len(sizes), 2, figsize=(7.0, 3.3 * len(sizes)), squeeze=False)
    for row, N in zip(axes, sizes):
        a = ind[(ind.N == N) & ~ind.factor.str.startswith("D_")]
        for ax, fam, title in ((row[0], "S", "First-order index"), (row[1], "OT", "OT index")):
            f = a[a.family == fam]
            moved = (f.raw - f.point).abs() > 0.03
            ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=1)
            ax.scatter(f.raw[~moved], f.point[~moved], s=9, color=BLUE, edgecolor=SURFACE,
                       linewidth=0.5, zorder=2, label="Changed by 0.03 or less")
            ax.scatter(f.raw[moved], f.point[moved], s=14, color=ORANGE, edgecolor=SURFACE,
                       linewidth=0.5, zorder=3, label="Changed by more than 0.03")
            lim = 0.3
            ax.set_xlim(-0.01, lim)
            ax.set_ylim(-0.01, lim)
            ax.set_xlabel("Raw (classes of the sample as drawn)")
            ax.set_ylabel("Adjusted (shares of the drivers restored)")
            ax.set_title(f"{title}, {N:,} runs", loc="left")
            ax.grid(True)
            ax.set_aspect("equal")
            ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_adjustment.pdf")
    plt.close(fig)


def convergence_frame(family, ind=None, sur=None, tot=None):
    """Long table (output, player, N, value, lo, hi) of one family of indices
    on the nested designs: "T" total indices of the surrogate (factors one by
    one), "S" and "OT" first-order and OT indices of the runs (all 21
    factors), "Shapley" Shapley effects of the surrogate."""
    if family == "T":
        s = tot[(tot.player != "D_cont") & (tot.player != "P_appliances")].rename(columns={"T": "value"})
    elif family == "Shapley":
        s = sur[sur.shapley.notna() & (sur.player != "D_cont")].rename(columns={"shapley": "value"})
    else:
        factors = [f"X{i}" for i in range(1, 22)]
        s = ind[(ind.family == family) & ind.factor.isin(factors)].rename(
            columns={"factor": "player", "point": "value"})
    return s[["output", "player", "N", "value", "lo", "hi"]]


def fig_convergence(s, out, family="T", name="fig_convergence.pdf", label_at=0.05):
    """Every factor of every output against the number of runs, with its 95%%
    interval at the largest size, on a square-root scale so that the indices
    near zero stay apart. The factors whose index reaches label_at at the
    largest size are coloured by family and labelled, the others are grey."""
    N = s.N.max()
    sizes = sorted(s.N.unique())
    fwd = lambda v: np.sqrt(np.clip(v, 0, None))  # noqa: E731
    inv = lambda v: np.square(v)                   # noqa: E731
    ticks = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 9.0), sharex=True)
    for ax, y in zip(axes.ravel(), OUTPUTS):
        a = s[s.output == y]
        last = a[a.N == N].set_index("player")
        order = last.value.sort_values().index.tolist()
        labelled = [p for p in order[::-1] if last.loc[p, "value"] >= label_at][:7]
        labels = []
        for p in order:                                # grey first, coloured on top
            r = a[a.player == p].sort_values("N")
            hot = p in labelled
            c = family_colour(p) if hot else MUTED
            if hot and c == MUTED:
                c = TEXT2
            ax.plot(r.N, r.value, color=c, linewidth=1.3 if hot else 0.7, alpha=1.0 if hot else 0.55,
                    marker="o" if hot else None, markersize=2.6, markeredgecolor=SURFACE,
                    markeredgewidth=0.5, zorder=3 if hot else 2)
            lo, hi = last.loc[p, "lo"], last.loc[p, "hi"]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([N, N], [max(lo, 0), max(hi, 0)], color=c, linewidth=0.9 if hot else 0.6,
                        alpha=1.0 if hot else 0.55, zorder=3 if hot else 2)
            if hot:
                labels.append([float(fwd(last.loc[p, "value"])), PLAYER_LABEL[p].strip().split(" (")[0]])
        # direct labels, pushed apart on the square-root scale
        labels.sort()
        for k in range(1, len(labels)):
            labels[k][0] = max(labels[k][0], labels[k - 1][0] + 0.056)
        for yl, text in labels:
            ax.annotate(text, (N, inv(yl)), xytext=(7, 0), textcoords="offset points", fontsize=5.3,
                        color=TEXT2, va="center")
        ax.set_xscale("log", base=2)
        ax.set_yscale("function", functions=(fwd, inv))
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0", "0.01", "0.05", "0.1", "0.25", "0.5", "1"], fontsize=6)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=6, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 3.4)
        ax.set_title(SHORT_OUTPUT[y], loc="left")
        ax.grid(axis="y")
    for ax in axes[-1]:
        ax.set_xlabel("Number of runs")
    for ax in axes[:, 0]:
        ax.set_ylabel(FAMILY_LABEL[family] + " (square-root scale)", fontsize=6.3)
    handles = [plt.Line2D([], [], color=c, linewidth=1.3, marker="o", markersize=2.6,
                          markeredgecolor=SURFACE) for c in (BLUE, ORANGE, AQUA, TEXT2)]
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.7, alpha=0.55))
    fig.legend(handles, ["Appliance powers", "Diarist coverage", "Lighting", "Other factor",
                         f"Below {label_at:.2f} at {N:,} runs"],
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / name)
    plt.close(fig)


def fig_profiles(bands_file, out):
    z = np.load(bands_file)
    bands, countries = z["bands"], [str(c) for c in z["countries"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    clock = 4 + (np.arange(144) + 0.5) / 6
    for ax, d, title in ((axes[0, 0], 0, "All countries, working days"),
                         (axes[0, 1], 1, "All countries, non-working days")):
        lo, med, hi = bands[:, -1, d, :]
        ax.fill_between(clock, lo, hi, color=BLUE, alpha=0.15, linewidth=0,
                        label="5th to 95th percentile of the runs")
        ax.plot(clock, med, color=BLUE, label="Median of the runs")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Mean household power (W)")
        ax.set_xticks([4, 8, 12, 16, 20, 24, 28])
        ax.set_xticklabels(["04", "08", "12", "16", "20", "00", "04"])
        ax.set_xlim(4, 28)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.legend(loc="upper left")
    for ax, c, bench, name in ((axes[1, 0], "BE", benchmarks.SYNERGRID_S21_BE, "Synergrid S21"),
                               (axes[1, 1], "FR", benchmarks.ENEDIS_RES1_FR, "Enedis RES1")):
        k = countries.index(c)
        if "shape_bands" in z.files:
            shapes = z["shape_bands"][:, k, :]
            band, mid = "Runs: 5th to 95th percentile of the shapes", "Runs: median shape"
        else:
            shapes = np.array([benchmarks.shape(benchmarks.hourly(bands[q, k, 2, :])) for q in range(3)])
            band, mid = "Runs: shapes of the 5th to 95th percentiles", "Runs: shape of the median profile"
        ax.fill_between(np.arange(24), shapes[0], shapes[2], color=BLUE, alpha=0.15, linewidth=0,
                        label=band)
        ax.plot(np.arange(24), shapes[1], color=BLUE, label=mid)
        ax.plot(np.arange(24), benchmarks.shape(bench), color=ORANGE, label=f"Measured: {name}")
        ax.set_title(f"{'Belgium' if c == 'BE' else 'France'}, all days: hourly shape", loc="left")
        ax.set_ylabel("Share of the daily energy above the minimum")
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        ax.set_xlim(0, 23)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.legend(loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Hour of the day")
    fig.tight_layout()
    fig.savefig(out / "fig_profiles.pdf")
    plt.close(fig)


def fig_hourly(hourly, out, day="all"):
    h = hourly[(hourly.day_type == day) & ~hourly.factor.str.startswith("D_")]
    piv = h.pivot_table(index="factor", columns="hour", values="ot")
    thr = h.pivot_table(index="factor", columns="hour", values="dummy_95")
    above = (piv > thr) & (piv >= 0.02)
    keep = [f for f in ROW_ORDER if f in piv.index and above.loc[f].any()]
    vals = piv.loc[keep].where(above.loc[keep]).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(keep) + 1.1))
    im = ax.imshow(np.nan_to_num(vals, nan=0.0), cmap=ramp(), vmin=0,
                   vmax=max(0.5, float(np.nanmax(vals))), aspect="auto")
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([PLAYER_LABEL[f].strip() for f in keep])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{x:02d}" for x in range(0, 24, 2)])
    ax.set_xlabel("Hour of the day")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    adj = ", adjusted" if "imbalance" in h.columns and h.imbalance.notna().any() else ""
    cb.set_label(f"OT index{adj} (blank below the dummy or 0.02)", color=TEXT2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_hourly.pdf")
    plt.close(fig)


def fig_drivers(table, summary, out):
    """Outputs of the runs against the factors that drive them, each run one
    point, with the canonical value as a line."""
    canon = summary.set_index("output").canonical
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    rng = np.random.default_rng(0)

    def canonical(ax, y):
        ax.axhline(canon[y], color=TEXT2, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.annotate("canonical", (1.0, canon[y]), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=5.5, color=TEXT2)

    def by_level(ax, y, x, levels, names, colour_by=None):
        for k, (lev, name) in enumerate(zip(levels, names)):
            sel = table[x] == lev
            jitter = k + rng.uniform(-0.28, 0.28, sel.sum())
            if colour_by is None:
                ax.scatter(jitter, table.loc[sel, y], s=3, color=BLUE, alpha=0.5, linewidth=0)
            else:
                for var, c in colour_by.items():
                    m = (table.loc[sel, "Y5_variable"] == var).to_numpy()
                    ax.scatter(jitter[m], table.loc[sel, y].to_numpy()[m], s=3, color=c, alpha=0.6,
                               linewidth=0, label=var if k == 0 or not m.any() else None)
            med = table.loc[sel, y].median()
            ax.plot([k - 0.35, k + 0.35], [med, med], color=TEXT, linewidth=1.2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.6, len(levels) - 0.4)

    ax = axes[0, 0]
    ax.scatter(table.X19, table.Y3, s=3, color=BLUE, alpha=0.5, linewidth=0)
    canonical(ax, "Y3")
    ax.axhline(0, color=AXIS, linewidth=0.6)
    ax.set_xlabel("Lighting X19 (W per household while lit)")
    ax.set_title("Y3 Coincident peak, change (%)", loc="left")
    ax = axes[0, 1]
    by_level(ax, "Y6", "X13", [0, 1], ["Every diarist", "First listed"])
    canonical(ax, "Y6")
    ax.set_xlabel("Diarist coverage X13")
    ax.set_title("Y6 Solutions led by adults", loc="left")
    ax = axes[0, 2]
    colours = {"HHC5": BLUE, "DDV6": ORANGE, "INC1": AQUA}
    by_level(ax, "Y5", "X13", [0, 1], ["Every diarist", "First listed"], colour_by=colours)
    canonical(ax, "Y5")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=3,
                          label={"HHC5": "Adults (HHC5)", "DDV6": "Employed or student (DDV6)",
                                 "INC1": "Sex (INC1)"}[v]) for v, c in colours.items()]
    ax.set_ylim(0.14, 0.52)
    ax.legend(handles=handles, loc="upper left", fontsize=5.5, title="Strongest variable",
              title_fontsize=5.5)
    ax.set_xlabel("Diarist coverage X13")
    ax.set_title("Y5 Strongest association (V)", loc="left")
    ax = axes[1, 0]
    by_level(ax, "Y7", "X13", [0, 1], ["Every diarist", "First listed"])
    canonical(ax, "Y7")
    ax.set_xlabel("Diarist coverage X13")
    ax.set_title("Y7 Single- to multi-diarist energy", loc="left")
    ax = axes[1, 1]
    ks = sorted(table.X16.unique())
    by_level(ax, "Y1", "X16", ks, [str(int(k)) for k in ks])
    canonical(ax, "Y1")
    ax.set_xlabel("K of the global solution X16")
    ax.set_title("Y1 Share in the lowest-energy group", loc="left")
    ax = axes[1, 2]
    ax.scatter(table.X19, table.Y9, s=3, color=BLUE, alpha=0.5, linewidth=0, label="Belgium (Y9)")
    ax.scatter(table.X19, table.Y10, s=3, color=ORANGE, alpha=0.5, linewidth=0, label="France (Y10)")
    ax.set_xlabel("Lighting X19 (W per household while lit)")
    ax.set_title("Y9, Y10 Shape correlation", loc="left")
    ax.legend(loc="lower right", fontsize=5.5, markerscale=2)
    for a in axes.ravel():
        a.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig_drivers.pdf")
    plt.close(fig)


UNITS = {"Y1": "Share of household-days", "Y2": "Share of household-days", "Y2b": "Hours",
         "Y3": "Change (%)", "Y4": "Change (%)", "Y5": "Cram\u00e9r's V (bias-corrected)",
         "Y6": "Solutions (of 48)", "Y7": "Ratio of mean daily energy",
         "Y8": "Adjusted Rand index", "Y9": "Correlation of hourly shapes",
         "Y10": "Correlation of hourly shapes", "Y11": "Ratio of hourly maxima"}


def ordinal(k):
    k = int(round(k))
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def family_colour(player):
    if player in APPLIANCES or player == "P_appliances":
        return BLUE
    return {"X13": ORANGE, "X19": AQUA}.get(player, MUTED)


def fig_distributions(table, summary, out):
    """Histogram of every output over the runs, stacked by diarist coverage
    (the factor that splits the bimodal outputs), with the canonical value.
    A spike holding more than a fifth of the runs is cut and labelled."""
    canon = summary.set_index("output")
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 8.4))
    groups = [(0, "Every diarist", BLUE), (1, "First listed", ORANGE)]
    for k, (ax, y) in enumerate(zip(axes.ravel(), OUTPUTS)):
        v = table[y].to_numpy(dtype=float)
        c = float(canon.loc[y, "canonical"])
        lo, hi = min(v.min(), c), max(v.max(), c)
        if y == "Y6":
            bins = np.arange(-0.5, 40.5, 1.0)
        else:
            pad = 0.02 * (hi - lo)
            bins = np.linspace(lo - pad, hi + pad, 41)
        parts = [table.loc[table.X13 == lev, y].to_numpy(dtype=float) for lev, _, _ in groups]
        counts, _, _ = ax.hist(parts, bins=bins, stacked=True, color=[g[2] for g in groups],
                               edgecolor=SURFACE, linewidth=0.3, label=[g[1] for g in groups])
        total = counts[-1]
        top = np.sort(total)
        spike = top[-1] > 0.2 * len(v) and len(top) > 1
        height = top[-2] if spike else top[-1]
        ax.set_ylim(0, 1.32 * height)               # headroom for the labels
        if spike:
            i = int(np.argmax(total))
            ax.annotate(f"{int(total[i]):,} runs at {canon.loc[y, 'mode']:g}, cut",
                        ((bins[i] + bins[i + 1]) / 2, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(5, -2), textcoords="offset points", ha="left", va="top",
                        fontsize=5.5, color=TEXT2)
        ax.axvline(c, color=TEXT, linewidth=0.9, linestyle=(0, (3, 2)))
        pct = canon.loc[y, "canonical_percentile"]
        x0, x1 = bins[0], bins[-1]
        left = (c - x0) / (x1 - x0) > 0.72 or (spike and (c - x0) / (x1 - x0) < 0.3)
        ax.annotate(f"canonical, {ordinal(pct)} percentile", (c, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-3 if left else 3, -2 if not spike else -10), textcoords="offset points",
                    ha="right" if left else "left", va="top", fontsize=5.5, color=TEXT2)
        ax.set_xlim(x0, x1)
        ax.set_title(SHORT_OUTPUT[y], loc="left")
        ax.set_xlabel(UNITS[y], fontsize=6.5)
        if k % 3 == 0:
            ax.set_ylabel("Runs")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=g[2]) for g in groups]
    fig.legend(handles, [f"Diarist coverage: {g[1].lower()}" for g in groups], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "fig_distributions.pdf")
    plt.close(fig)


def fig_ot_s(otp, out, excess=0.04):
    """OT index against first-order index of every factor and
    output, with the lower bound S/2 and the OT index of a factor that only
    shifts a Gaussian output. Points more than `excess` above that curve, the
    factors that change the spread or shape of the output, are named."""
    from gsa.analyse import location_ot
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    grid = np.linspace(0, 1, 201)
    d = otp.copy()
    d["excess"] = d.OT - d.location_OT
    for ax, (xmax, ymax), named in zip(axes, ((1.0, 1.0), (0.5, 0.42)), (False, True)):
        ax.plot(grid, grid / 2, color=AXIS, linewidth=0.8, label="Lower bound S/2")
        ax.plot(grid, location_ot(grid), color=TEXT2, linewidth=0.9, linestyle=(0, (3, 2)),
                label="Shift of a Gaussian output")
        for colour in (MUTED, BLUE, AQUA, ORANGE):
            sel = d[[family_colour(f) == colour for f in d.factor]]
            ax.scatter(sel.S, sel.OT, s=10, color=colour, edgecolor=SURFACE, linewidth=0.4, zorder=3)
        if named:
            lab = d[(d.excess > excess) & (d.S <= xmax) & (d.OT <= ymax)].sort_values("OT")
            placed = []                              # (x, y) of the labels already set
            for _, r in lab.iterrows():
                x, y = r.S + 0.008 * xmax, r.OT
                for px, py in sorted(placed, key=lambda t: t[1]):
                    if abs(px - x) < 0.2 * xmax and abs(py - y) < 0.05 * ymax:
                        y = py + 0.05 * ymax
                placed.append((x, y))
                ax.annotate(f"{r.factor} on {r.output}", (r.S, r.OT), xytext=(x, y), textcoords="data",
                            fontsize=5.5, color=TEXT2, va="center",
                            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.5,
                                            shrinkA=0, shrinkB=2) if abs(y - r.OT) > 1e-9 else None)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("First-order index S")
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("OT index")
    axes[0].set_title("All factors and outputs", loc="left")
    axes[1].set_title("Detail", loc="left")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=4)
               for c in (BLUE, ORANGE, AQUA, MUTED)]
    names = ["Appliance powers", "Diarist coverage", "Lighting", "Other factors"]
    lines = axes[0].get_legend_handles_labels()
    axes[0].legend(handles + lines[0], names + lines[1], loc="upper left", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(out / "fig_ot_s.pdf")
    plt.close(fig)


def fig_brackets(br, out, floor=0.02, top=7):
    """For every output, the players of the surrogate (the appliance powers
    one by one) whose total index reaches `floor`: a line from the
    first-order to the total index, and the Shapley effect as a dot, all as
    shares of Var Y. For independent inputs S <= Sh <= T."""
    b = br[(br.game == "single") & (br.player != "D_cont")]
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 8.6))
    for ax, y in zip(axes.ravel(), OUTPUTS):
        a = b[(b.output == y) & (b["T"] >= floor)].sort_values("shapley", ascending=False).head(top)
        a = a.iloc[::-1]
        for i, (_, r) in enumerate(a.iterrows()):
            colour = family_colour(r.player)
            ax.plot([r.S, r["T"]], [i, i], color=colour, linewidth=1.6, solid_capstyle="round")
            ax.plot(r.shapley, i, "o", color=colour, markersize=4, markeredgecolor=SURFACE,
                    markeredgewidth=0.6, zorder=3)
        ax.set_yticks(range(len(a)))
        ax.set_yticklabels([PLAYER_LABEL[p].strip() for p in a.player], fontsize=5.5)
        ax.set_ylim(-0.7, max(len(a), 1) - 0.3)
        hi = float(max(a["T"].max(), a.shapley.max())) if len(a) else 0.1
        ax.set_xlim(0, min(1.06, max(0.1, 1.12 * hi)))
        explained = float(b.loc[b.output == y, "explained"].iloc[0])
        ax.set_title(SHORT_OUTPUT[y], loc="left")
        ax.annotate(f"reproduced {explained:.0%}", (1.0, 0.0), xycoords="axes fraction", xytext=(-2, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=5.5, color=TEXT2)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel("Share of Var Y")
    handles = [plt.Line2D([], [], color=TEXT2, linewidth=1.6),
               plt.Line2D([], [], color=TEXT2, marker="o", linestyle="", markersize=4)]
    fig.legend(handles, ["First-order to total index", "Shapley effect"], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "fig_brackets.pdf")
    plt.close(fig)


def fig_budget(budget, out):
    """Variance budget of every output as a stacked bar, from the model
    alone: first-order effects of the factors (runs), their interactions (by
    difference) split into the part the surrogate apportions and the rest,
    and the K-means seed and the draw of the country multipliers (reruns).
    The parts add up to one."""
    YELLOW, MAGENTA = "#eda100", "#e87ba4"
    parts = [("first_order_runs", "First-order effects", BLUE),
             ("surrogate_interactions", "Interactions apportioned by the surrogate", AQUA),
             ("interactions_not_apportioned", "Interactions not apportioned", MUTED)]
    if "kmeans" in budget.columns:
        parts += [("kmeans", "K-means seed", YELLOW), ("multipliers", "Draw of the multipliers (X8)", MAGENTA)]
    else:
        parts += [("seeds", "Seeds", YELLOW)]
    b = budget.set_index("output").reindex([y for y in OUTPUTS if y in set(budget.output)])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ypos = np.arange(len(b))[::-1]
    left = np.zeros(len(b))
    for col, name, colour in parts:
        v = np.clip(b[col].to_numpy(dtype=float), 0, None)
        ax.barh(ypos, v, left=left, color=colour, edgecolor=SURFACE, linewidth=1.0, height=0.7, label=name)
        left += v
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT_OUTPUT[y] for y in b.index])
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Share of Var Y")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.17), ncol=3, fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_budget.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--share", type=Path, default=None, help="folder of gsa_outputs.csv")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    r = args.results
    if args.share is not None and (r / "summary.csv").is_file():
        table = pd.read_csv(args.share / "gsa_outputs.csv")
        summ = pd.read_csv(r / "summary.csv")
        fig_drivers(table, summ, args.out)
        if "mode" in summ.columns:
            fig_distributions(table, summ, args.out)
    if (r / "ot_parts.csv").is_file():
        fig_ot_s(pd.read_csv(r / "ot_parts.csv"), args.out)
    if (r / "brackets.csv").is_file():
        fig_brackets(pd.read_csv(r / "brackets.csv"), args.out)
    if (r / "budget.csv").is_file():
        fig_budget(pd.read_csv(r / "budget.csv"), args.out)
    ind = pd.read_csv(r / "indices.csv")
    sur = pd.read_csv(r / "surrogate.csv") if (r / "surrogate.csv").is_file() else None
    tot = pd.read_csv(r / "totals.csv") if (r / "totals.csv").is_file() else None
    seeds = pd.read_csv(r / "seeds.csv") if (r / "seeds.csv").is_file() else None
    if sur is not None:
        fig_effects(sur, args.out)
    if tot is not None:
        fig_totals(tot, args.out, seeds)
    fig_indices(ind, sur, args.out, tot)
    if ind.imbalance.notna().any():
        fig_adjustment(ind, args.out)
    if tot is not None:
        fig_convergence(convergence_frame("T", tot=tot), args.out, "T")
    elif sur is not None:
        fig_convergence(convergence_frame("Shapley", sur=sur), args.out, "Shapley")
    for fam in ("S", "OT"):
        fig_convergence(convergence_frame(fam, ind=ind), args.out, fam,
                        name=f"fig_convergence_{fam.lower()}.pdf")
    if (r / "profile_bands.npz").is_file():
        fig_profiles(r / "profile_bands.npz", args.out)
    if (r / "hourly.csv").is_file():
        fig_hourly(pd.read_csv(r / "hourly.csv"), args.out)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()

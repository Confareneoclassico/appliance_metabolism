"""Figures of the sensitivity analysis from the files of gsa.analyse.

    python3 -m gsa.figures --results FOLDER --out FOLDER

Draws, as PDF:

  fig_effects.pdf      Shapley effects of the surrogate on every output at the
                       largest size, as shares of Var Y (blank where an effect
                       does not clear the dummy), and the share of Var Y the
                       surrogate does not reproduce
  fig_indices.pdf      adjusted first-order and OT indices and the surrogate's
                       Shapley effects with their 95% intervals, one panel per
                       output, for the factors read as non-zero and of at least
                       0.02 in some family
  fig_adjustment.pdf   raw against adjusted first-order and OT indices of
                       every factor and output at the two largest sizes: the
                       raking moves only the indices that took on another
                       factor's effect
  fig_convergence.pdf  Shapley effects of the surrogate of the leading
                       players against the number of runs (128 to 1,024)
  fig_profiles.pdf     the pooled profile by day type and the hourly shapes of
                       Belgium and France against their measured benchmarks:
                       median and 5th to 95th percentile over the runs
  fig_hourly.pdf       adjusted OT index of each factor on the hourly values of
                       the pooled profile (blank below the dummy)
  fig_drivers.pdf      outputs of the runs against their leading factors, with
                       the canonical value (needs --share, the folder of
                       gsa_outputs.csv)

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
FAMILY = {"Shapley": BLUE, "S": ORANGE, "OT": AQUA}
FAMILY_LABEL = {"Shapley": "Shapley effect (surrogate)", "S": "First-order index (adjusted)",
                "OT": "OT index (adjusted)"}
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


def fig_indices(ind, sur, out):
    a = largest(ind)
    s = largest(sur)
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 9.4))
    for ax, y in zip(axes.ravel(), OUTPUTS):
        ay, sy = a[a.output == y], s[(s.output == y) & s.shapley.notna()]
        nz = (set(ay[ay.nonzero.map(is_true) & (ay.point >= 0.02)].factor)
              | set(sy[sy.nonzero.map(is_true) & (sy.shapley >= 0.02)].player))
        rows = [p for p in ROW_ORDER if p in nz or (p == "P_appliances" and nz & set(APPLIANCES))]
        ypos = np.arange(len(rows))[::-1]
        for fam, dy in (("S", 0.24), ("OT", 0.0), ("Shapley", -0.24)):
            for yp, p in zip(ypos, rows):
                if fam == "Shapley":
                    r = sy[sy.player == p]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    pt, lo, hi = r.shapley, r.lo, r.hi
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
                          label=FAMILY_LABEL[f]) for f in ("S", "OT", "Shapley")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.022, 1, 1))
    fig.savefig(out / "fig_indices.pdf")
    plt.close(fig)


def fig_adjustment(ind, out):
    """Raw against adjusted indices at the two largest sizes (one row each)."""
    sizes = sorted(ind.N.unique())[-2:]
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


def fig_convergence(sur, out):
    s = sur[sur.shapley.notna() & (sur.player != "D_cont")]
    N = s.N.max()
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 8.2), sharex=True)
    for ax, y in zip(axes.ravel(), OUTPUTS):
        a = s[s.output == y]
        top = a[a.N == N].sort_values("shapley", ascending=False).player.head(3).tolist()
        labels = []
        for p in top:
            r = a[a.player == p].sort_values("N")
            c = PLAYER_COLOUR.get(p, MUTED)
            ax.plot(r.N, r.shapley, color=c, marker="o", markersize=3, markeredgecolor=SURFACE,
                    markeredgewidth=0.6)
            last = r[r.N == N].iloc[0]
            if np.isfinite(last.lo):
                ax.plot([N, N], [last.lo, last.hi], color=c, linewidth=1.0)
            labels.append([last.shapley, PLAYER_LABEL[p].strip().split(" (")[0]])
        # direct labels, pushed apart where the effects are close
        labels.sort()
        for i in range(1, len(labels)):
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.08)
        for yl, text in labels:
            ax.annotate(text, (N, yl), xytext=(6, 0), textcoords="offset points", fontsize=5.5,
                        color=TEXT2, va="center")
        ax.set_xscale("log", base=2)
        sizes = sorted(s.N.unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=6)
        ax.minorticks_off()
        ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 3.2)
        ax.set_ylim(min(0, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1]))
        ax.set_title(SHORT_OUTPUT[y], loc="left")
        ax.grid(axis="y")
    for ax in axes[-1]:
        ax.set_xlabel("Number of runs")
    fig.tight_layout()
    fig.savefig(out / "fig_convergence.pdf")
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
    cb.set_label("OT index, adjusted (blank below the dummy or 0.02)", color=TEXT2)
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
        fig_drivers(pd.read_csv(args.share / "gsa_outputs.csv"), pd.read_csv(r / "summary.csv"), args.out)
    ind = pd.read_csv(r / "indices.csv")
    sur = pd.read_csv(r / "surrogate.csv")
    fig_effects(sur, args.out)
    fig_indices(ind, sur, args.out)
    fig_adjustment(ind, args.out)
    fig_convergence(sur, args.out)
    if (r / "profile_bands.npz").is_file():
        fig_profiles(r / "profile_bands.npz", args.out)
    if (r / "hourly.csv").is_file():
        fig_hourly(pd.read_csv(r / "hourly.csv"), args.out)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()

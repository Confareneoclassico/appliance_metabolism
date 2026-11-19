"""Supplementary figure: mean profiles of the country-level clusters.

Reads the export written by Stage 06 (processed/stratified_cluster_mean_profiles.csv)
and draws one panel per country with the mean daily profile of every cluster
found in that country. Clusters are coloured by their size rank within the
country (the largest first), so that the same colour marks the same rank in
every panel, not the same behaviour. Each panel has its own vertical scale.

The export follows Eurostat's publication rules for HETUS (disclosure.py).
From version 1.1.0 every cluster of fewer than 50 household-days is reported
together with the largest cluster of its country, so each published group has
at least 50 household-days and is drawn with a solid line; the panel title
gives the number of clusters K of the country's solution and how many small
clusters are included in the largest. Rows that an older export flags (20 to
49 household-days) are drawn dashed and withheld rows are not drawn.

The first eight size ranks have their own colour, in a fixed order; ranks from
the ninth on are drawn in grey, so no colour is reused (audit of November 2026,
L8: the lists used to have eight entries while K can reach 10).

The script needs only the export file, not the HETUS microdata, so the figure
can be redrawn and checked by anyone who has the aggregate outputs.

    python pipeline/plot_country_cluster_profiles.py [path/to/export.csv]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import PROCESSED_DIR, FIGURES_DIR

DIARY_START_HOUR = 4  # HETUS diary day runs from 04:00 to 04:00
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOUR = "#9a998f"   # ranks beyond the eighth
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def ordinal(k):
    """1st, 2nd, 3rd, 4th, ..., 11th, 12th, 13th, ..., 21st, ..."""
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def rank_colour(rank):
    """Colour of a size rank counted from 0 (the largest)."""
    return SERIES_COLOURS[rank] if rank < len(SERIES_COLOURS) else OTHER_COLOUR


def legend_entries(max_rank):
    """Colours and labels of the ranks drawn, from 1 to `max_rank`, with the
    ranks beyond the eighth in one grey entry."""
    n_named = min(max_rank, len(SERIES_COLOURS))
    colours = SERIES_COLOURS[:n_named]
    labels = [ordinal(r + 1) + (" (largest)" if r == 0 else "") for r in range(n_named)]
    if max_rank > len(SERIES_COLOURS):
        colours.append(OTHER_COLOUR)
        labels.append(f"{ordinal(len(SERIES_COLOURS) + 1)} and smaller")
    return colours, labels


def _as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1"])


def main(path=None):
    path = path or PROCESSED_DIR / "stratified_cluster_mean_profiles.csv"
    data = pd.read_csv(path)
    # Country strata are named by their two-letter code; the pooled day-type
    # strata and the country x day-type strata have longer names
    data = data[data["stratum"].str.fullmatch(r"[A-Z]{2}")].copy()
    pcols = [f"P_{t:03d}" for t in range(144)]
    if "withheld" in data:
        withheld = _as_bool(data["withheld"])
    elif "profile_suppressed" in data:  # exports written before October 2026
        withheld = _as_bool(data["profile_suppressed"])
    else:
        withheld = pd.Series(False, index=data.index)
    data["withheld"] = withheld | data[pcols].isna().all(axis=1)
    data["flagged"] = (data["n_class"].fillna("").astype(str) != "") & ~data["withheld"] \
        if "n_class" in data else False
    if "size_rank" not in data:
        data["size_rank"] = data.groupby("stratum")["n"].rank(ascending=False, method="first")
    if "n_small_clusters_merged" not in data:
        data["n_small_clusters_merged"] = 0
    hours = np.arange(144) / 6  # hours since 04:00
    countries = sorted(data["stratum"].unique())
    n_cols = 3
    n_rows = int(np.ceil(len(countries) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.25 * n_rows + 0.9),
                             sharex=True, squeeze=False)
    max_rank_shown, any_flagged = 0, False
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        if idx >= len(countries):
            ax.set_visible(False)
            continue
        country = countries[idx]
        sub = data[data["stratum"] == country].sort_values("size_rank")
        merged = int(pd.to_numeric(sub["n_small_clusters_merged"]).fillna(0).sum())
        k_total = len(sub) + merged
        n_total = int(sub["stratum_n"].iloc[0]) if "stratum_n" in sub else int(sub["n"].sum())
        shown = 0
        for _, row in sub.iterrows():
            if row["withheld"]:
                continue
            rank = int(row["size_rank"]) - 1
            y = row[pcols].values.astype(float)
            colour = rank_colour(rank)
            style = "--" if row["flagged"] else "-"
            ax.plot(hours, y, color=colour, linewidth=1.5, linestyle=style)
            k = int(np.nanargmax(y))
            ax.plot(hours[k], y[k], "o", color=colour, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            shown += 1
            any_flagged |= bool(row["flagged"])
            max_rank_shown = max(max_rank_shown, rank + 1)
        note = f"K = {k_total}"
        if merged:
            note += f" ({merged} in the largest)"
        elif shown < k_total:
            note += f" ({shown} shown)"
        ax.set_title(f"{country}  {note},  n = {n_total:,}", fontsize=10.5,
                     color=INK, loc="left")
        ax.grid(True, axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.set_xlim(0, 24)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xticklabels([f"{(DIARY_START_HOUR + t) % 24:02d}:00" for t in range(0, 25, 6)])
    colours, labels = legend_entries(max_rank_shown)
    handles = [plt.Line2D([0], [0], color=c, linewidth=2.2) for c in colours]
    if any_flagged:
        handles.append(plt.Line2D([0], [0], color=INK_2, linewidth=1.6, linestyle="--"))
        labels.append("20 to 49 household-days")
    n_cols_legend = len(labels) if len(labels) <= 6 else int(np.ceil(len(labels) / 2))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=n_cols_legend, frameon=False, fontsize=10.5, labelcolor=INK,
               title="Cluster size rank within the country (clusters of fewer than 50 "
                     "household-days are included in the largest)", title_fontsize=11)
    fig.supxlabel("Time of day", fontsize=13, color=INK)
    fig.supylabel("Mean power demand (W)", fontsize=13, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.945 if n_cols_legend == len(labels) else 0.915])
    out = FIGURES_DIR / "country_cluster_profiles_small_multiples.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Country cluster profiles saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

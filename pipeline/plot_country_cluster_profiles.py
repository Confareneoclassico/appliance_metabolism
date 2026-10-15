"""Supplementary figure: mean profiles of the country-level clusters.

Reads the export written by Stage 06 (processed/stratified_cluster_mean_profiles.csv)
and draws one panel per country with the mean daily profile of every cluster
found in that country. Clusters are coloured by their size rank within the
country (the largest first), so that the same colour marks the same rank in
every panel, not the same behaviour. Each panel has its own vertical scale.

The export follows Eurostat's publication rules for HETUS (disclosure.py):
clusters of fewer than 20 household-days are withheld and are not drawn (their
household-days are included in the largest cluster of their country), and
clusters of 20 to 49 household-days are flagged and drawn with a dashed line.
The panel title gives how many of the country's clusters are drawn.

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
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
ORDINALS = ["1st (largest)", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


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
        k_total = len(sub)
        n_total = int(sub["stratum_n"].iloc[0]) if "stratum_n" in sub else int(sub["n"].sum())
        shown = 0
        for _, row in sub.iterrows():
            if row["withheld"]:
                continue
            rank = int(row["size_rank"]) - 1
            y = row[pcols].values.astype(float)
            colour = SERIES_COLOURS[rank % len(SERIES_COLOURS)]
            style = "--" if row["flagged"] else "-"
            ax.plot(hours, y, color=colour, linewidth=1.5, linestyle=style)
            k = int(np.nanargmax(y))
            ax.plot(hours[k], y[k], "o", color=colour, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            shown += 1
            any_flagged |= bool(row["flagged"])
            max_rank_shown = max(max_rank_shown, rank + 1)
        note = f"K = {k_total}" + (f" ({shown} shown)" if shown < k_total else "")
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
    handles = [plt.Line2D([0], [0], color=SERIES_COLOURS[r], linewidth=2.2)
               for r in range(max_rank_shown)]
    labels = ORDINALS[:max_rank_shown]
    if any_flagged:
        handles.append(plt.Line2D([0], [0], color=INK_2, linewidth=1.6, linestyle="--"))
        labels.append("20 to 49 household-days")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False, fontsize=10.5, labelcolor=INK,
               title="Cluster size rank within the country", title_fontsize=11)
    fig.supxlabel("Time of day", fontsize=13, color=INK)
    fig.supylabel("Mean power demand (W)", fontsize=13, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = FIGURES_DIR / "country_cluster_profiles_small_multiples.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Country cluster profiles saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

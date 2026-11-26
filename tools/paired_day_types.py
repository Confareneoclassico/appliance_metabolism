"""Working and non-working days of the same households (Reviewer 2, comment 5).

Stage 06 compares the household-days of each day type within a country, and
the two samples hold different households wherever a household kept diaries
on one day type only. This script repeats the comparison on the households
that kept diaries on both, so that the day type changes and the households
do not:

    python3 tools/paired_day_types.py [--processed FOLDER | --bundle BUNDLE]
                                      [--out FOLDER] [--day-type ddv7|ddv1]
                                      [--boot 1000]

It reads household_interval_timeseries.csv of Stage 03 (identifiers, DDV1,
DDV7 and the power of every slot) from the processed folder of a run, so it
runs on the machine that holds the microdata, after Stages 01 to 03. The day
type is that of Stage 06 (DDV7, then DDV1 where DDV7 is missing; --day-type
ddv1 for the weekday rule of every country). A household is paired when it has
at least one working and one non-working household-day; where it has several
of one type (the Netherlands, seven diary days) its profiles of that type are
averaged, so that every paired household counts once on each side.

For every country with at least MIN_N_UNFLAGGED (50) paired households, and
for all of them together, it writes paired_daytype_comparison.csv:

  the change from working to non-working days of the coincident peak per
  household (maximum of the mean profile) among all household-days, as in
  Stage 06, and among the paired households, with a 95% interval from
  resampling the paired households;
  the same for the mean daily peak of the households (own peak) and for the
  mean daily energy, with the p-value of Wilcoxon's signed-rank test on the
  paired differences;
  the clock time of the coincident peak on each day type, paired;
  the number of paired households rounded to ten, and the share of each day
  type's household-days that belong to them (withheld where either part holds
  fewer than MIN_N_PUBLISH household-days, so that no small group can be
  derived from the sizes Stage 06 publishes).

Countries with fewer paired households are listed with their size class only.
Every value is an aggregate over at least 50 households, as the publication
rules of pipeline/disclosure.py require; no profile is written. A figure of
the two changes by country goes to the figures folder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from disclosure import MIN_N_PUBLISH, MIN_N_UNFLAGGED, round_size, size_class  # noqa: E402
from run_folder import check, country_names, describe, resolve  # noqa: E402

SLOTS = 144
ENERGY = [f"Household_Energy_W_{t}" for t in range(SLOTS)]
INTERVAL_HOURS = 10 / 60
WORKING, NON_WORKING = 0, 1


def by_ddv1(ddv1):
    """Stage 06's weekday rule: DDV1 2 to 6 working, 1 or 7 non-working."""
    ddv1 = np.asarray(ddv1, dtype=float)
    return np.where(np.isin(ddv1, [2, 3, 4, 5, 6]), WORKING,
                    np.where(np.isin(ddv1, [1, 7]), NON_WORKING, -1)).astype(np.int8)


def day_type(ddv7, ddv1, rule="ddv7"):
    """Stage 06's determine_day_type, vectorised: DDV7 1 or 2 working, 3 to 6
    non-working, the weekday where DDV7 is missing or negative, -1 where
    undetermined. rule="ddv1" applies the weekday rule everywhere."""
    if rule == "ddv1":
        return by_ddv1(ddv1)
    ddv7 = np.asarray(ddv7, dtype=float)
    out = np.full(len(ddv7), -1, dtype=np.int8)
    missing = ~np.isfinite(ddv7) | (ddv7 < 0)
    out[missing] = by_ddv1(ddv1)[missing]
    out[np.isin(ddv7, [1, 2])] = WORKING
    out[np.isin(ddv7, [3, 4, 5, 6])] = NON_WORKING
    return out


def slot_clock(k):
    """Clock time HH:MM of the start of slot k (the diary day starts at 04:00)."""
    m = (4 * 60 + 10 * int(k)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def load(processed, rule="ddv7"):
    """Identifiers, day types and power profiles of every household-day."""
    path = Path(processed) / "household_interval_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run Stages 01 to 03 first, with the same "
                                "HETUS_PROCESSED_DIR")
    ids, power = [], []
    for ch in pd.read_csv(path, usecols=["HID", "DIARY", "COUNTRY", "DDV1", "DDV7"] + ENERGY,
                          chunksize=20000, dtype={c: np.float32 for c in ENERGY}):
        ids.append(ch[["HID", "DIARY", "COUNTRY", "DDV1", "DDV7"]])
        power.append(ch[ENERGY].to_numpy(dtype=np.float32))
    ids = pd.concat(ids, ignore_index=True)
    ids["COUNTRY"] = country_names(ids.COUNTRY)          # codes of Stage 01 -> names of Stage 06
    ids["day_type"] = day_type(pd.to_numeric(ids.DDV7, errors="coerce"),
                               pd.to_numeric(ids.DDV1, errors="coerce"), rule)
    return ids, np.vstack(power)


def household_means(ids, power):
    """For every household with both day types: its mean profile on working
    and on non-working days, its mean daily peak and mean daily energy on
    each. Returns the country of each paired household and the arrays."""
    ok = ids.day_type.to_numpy() >= 0
    key = ids.COUNTRY.astype(str) + "|" + ids.HID.astype(str)
    hh, inv = np.unique(key[ok].to_numpy(), return_inverse=True)
    dt = ids.day_type.to_numpy()[ok]
    p = power[ok].astype(np.float64)
    peak = p.max(axis=1)
    energy = p.sum(axis=1) * INTERVAL_HOURS / 1000          # kWh
    n = len(hh)
    out = {}
    for t, name in ((WORKING, "w"), (NON_WORKING, "n")):
        m = dt == t
        cnt = np.bincount(inv[m], minlength=n).astype(float)
        prof = np.zeros((n, SLOTS))
        np.add.at(prof, inv[m], p[m])
        out[name] = {"count": cnt, "profile": prof,
                     "peak": np.bincount(inv[m], weights=peak[m], minlength=n),
                     "energy": np.bincount(inv[m], weights=energy[m], minlength=n)}
    paired = (out["w"]["count"] > 0) & (out["n"]["count"] > 0)
    res = {"country": np.array([h.split("|")[0] for h in hh[paired]])}
    for name in ("w", "n"):
        c = out[name]["count"][paired]
        res[name] = {"days": c,
                     "profile": out[name]["profile"][paired] / c[:, None],
                     "peak": out[name]["peak"][paired] / c,
                     "energy": out[name]["energy"][paired] / c}
    return res


def change(new, old):
    return 100 * (new / old - 1)


def boot_changes(pw, pn, kw, kn, ew, en, B, rng, batch=100):
    """Percentage changes of the coincident peak, the mean daily peak and the
    mean daily energy over B resamples of the paired households."""
    n = len(kw)
    out = []
    for b0 in range(0, B, batch):
        w = rng.multinomial(n, np.full(n, 1 / n), size=min(batch, B - b0)).astype(np.float64)
        coinc = change((w @ pn).max(axis=1), (w @ pw).max(axis=1))
        out.append(np.column_stack([coinc, change(w @ kn, w @ kw), change(w @ en, w @ ew)]))
    return np.vstack(out)


COLUMNS = ["country", "n_paired_households", "N_Class",
           "all_coincident_w", "all_coincident_nw", "all_own_peak_w", "all_own_peak_nw",
           "all_energy_w", "all_energy_nw", "all_coincident_change_pct", "all_own_peak_change_pct",
           "all_energy_change_pct", "share_paired_w_pct", "share_paired_nw_pct",
           "paired_coincident_w", "paired_coincident_nw", "paired_coincident_time_w",
           "paired_coincident_time_nw", "paired_own_peak_w", "paired_own_peak_nw",
           "paired_energy_w", "paired_energy_nw", "paired_coincident_change_pct",
           "paired_own_peak_change_pct", "paired_energy_change_pct",
           "paired_coincident_change_lo", "paired_coincident_change_hi",
           "paired_own_peak_change_lo", "paired_own_peak_change_hi",
           "paired_energy_change_lo", "paired_energy_change_hi",
           "own_peak_wilcoxon_p", "energy_wilcoxon_p"]


def compare(ids, power, B=1000, seed=20261126):
    """One row per country (and one for all countries together) with the
    changes among all household-days and among the paired households."""
    pairs = household_means(ids, power)
    rng = np.random.default_rng(seed)
    rows = []
    countries = sorted(ids.COUNTRY.astype(str).unique())
    for c in countries + ["ALL"]:
        sel_days = ids.COUNTRY.astype(str).to_numpy() == c if c != "ALL" else \
            np.isin(ids.COUNTRY.astype(str).to_numpy(), [k for k in countries
                                                         if (pairs["country"] == k).sum() >= MIN_N_UNFLAGGED])
        sel_pairs = pairs["country"] == c if c != "ALL" else \
            np.isin(pairs["country"], [k for k in countries if (pairs["country"] == k).sum() >= MIN_N_UNFLAGGED])
        n_pairs = int(sel_pairs.sum())
        row = {"country": c, "n_paired_households": np.nan,
               "N_Class": "none" if n_pairs == 0 else size_class(n_pairs)}
        if n_pairs < MIN_N_UNFLAGGED:
            if c != "ALL":
                rows.append(row)
            continue
        dt = ids.day_type.to_numpy()
        days_w, days_n = sel_days & (dt == WORKING), sel_days & (dt == NON_WORKING)
        # all household-days, as Stage 06
        all_w, all_n = power[days_w].astype(np.float64), power[days_n].astype(np.float64)
        row.update({
            "n_paired_households": float(round_size(n_pairs)), "N_Class": "",
            "all_coincident_w": all_w.mean(axis=0).max(), "all_coincident_nw": all_n.mean(axis=0).max(),
            "all_own_peak_w": all_w.max(axis=1).mean(), "all_own_peak_nw": all_n.max(axis=1).mean(),
            "all_energy_w": all_w.sum(axis=1).mean() * INTERVAL_HOURS / 1000,
            "all_energy_nw": all_n.sum(axis=1).mean() * INTERVAL_HOURS / 1000})
        row["all_coincident_change_pct"] = change(row["all_coincident_nw"], row["all_coincident_w"])
        row["all_own_peak_change_pct"] = change(row["all_own_peak_nw"], row["all_own_peak_w"])
        row["all_energy_change_pct"] = change(row["all_energy_nw"], row["all_energy_w"])
        # the share of each day type's household-days held by paired households
        for t, name, days in ((WORKING, "w", days_w), (NON_WORKING, "nw", days_n)):
            held = int(pairs["w" if t == WORKING else "n"]["days"][sel_pairs].sum())
            rest = int(days.sum()) - held
            ok = held >= MIN_N_PUBLISH and (rest == 0 or rest >= MIN_N_PUBLISH)
            row[f"share_paired_{name}_pct"] = 100 * held / days.sum() if ok else np.nan
        pw, pn = pairs["w"]["profile"][sel_pairs], pairs["n"]["profile"][sel_pairs]
        kw, kn = pairs["w"]["peak"][sel_pairs], pairs["n"]["peak"][sel_pairs]
        ew, en = pairs["w"]["energy"][sel_pairs], pairs["n"]["energy"][sel_pairs]
        mw, mn = pw.mean(axis=0), pn.mean(axis=0)
        row.update({
            "paired_coincident_w": mw.max(), "paired_coincident_nw": mn.max(),
            "paired_coincident_time_w": slot_clock(mw.argmax()),
            "paired_coincident_time_nw": slot_clock(mn.argmax()),
            "paired_own_peak_w": kw.mean(), "paired_own_peak_nw": kn.mean(),
            "paired_energy_w": ew.mean(), "paired_energy_nw": en.mean()})
        row["paired_coincident_change_pct"] = change(mn.max(), mw.max())
        row["paired_own_peak_change_pct"] = change(kn.mean(), kw.mean())
        row["paired_energy_change_pct"] = change(en.mean(), ew.mean())
        bs = boot_changes(pw, pn, kw, kn, ew, en, B, rng)
        for j, name in enumerate(("coincident", "own_peak", "energy")):
            row[f"paired_{name}_change_lo"], row[f"paired_{name}_change_hi"] = np.percentile(bs[:, j], [2.5, 97.5])
        row["own_peak_wilcoxon_p"] = float(wilcoxon(kn - kw).pvalue) if np.any(kn != kw) else 1.0
        row["energy_wilcoxon_p"] = float(wilcoxon(en - ew).pvalue) if np.any(en != ew) else 1.0
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def figure(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[table.n_paired_households.notna()].copy()
    pooled = t[t.country == "ALL"]
    t = t[t.country != "ALL"]
    t = pd.concat([pooled, t.iloc[np.argsort(t.all_coincident_change_pct.to_numpy())]])
    t["country"] = t.country.replace({"ALL": "All countries"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.3 * len(t) + 1.5), sharey=True)
    yy = np.arange(len(t))
    for ax, name, title in ((axes[0], "coincident", "Coincident peak per household"),
                            (axes[1], "own_peak", "Mean daily peak of the households")):
        ax.axvline(0, color="#c3c2b7", linewidth=0.8)
        ax.plot(t[f"all_{name}_change_pct"], yy + 0.12, "o", color="#898781", markersize=4,
                label="All household-days (Stage 06)")
        ax.errorbar(t[f"paired_{name}_change_pct"], yy - 0.12,
                    xerr=[t[f"paired_{name}_change_pct"] - t[f"paired_{name}_change_lo"],
                          t[f"paired_{name}_change_hi"] - t[f"paired_{name}_change_pct"]],
                    fmt="o", color="#2a78d6", markersize=4, elinewidth=1, capsize=0,
                    label="Same households, 95% interval")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel("Change from working to non-working days (%)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(t.country)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processed", type=Path, default=None,
                        help="processed folder of the run (default: that of pipeline/paths.py)")
    parser.add_argument("--bundle", type=Path, default=None,
                        help="bundle of the sensitivity analysis whose run to use (its meta.json)")
    parser.add_argument("--out", type=Path, default=None,
                        help="folder for the table (default: Country_DayType_Analysis of the run)")
    parser.add_argument("--figures", type=Path, default=None, help="folder for the figure")
    parser.add_argument("--day-type", choices=["ddv7", "ddv1"], default="ddv7")
    parser.add_argument("--boot", type=int, default=1000, help="resamples of the paired households")
    args = parser.parse_args()
    from paths import FIGURES_DIR
    processed = resolve(args.processed, args.bundle)
    check(processed, solutions=False)
    print(describe(processed))
    out = args.out or Path(processed) / "Country_DayType_Analysis"
    figs = args.figures or FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    ids, power = load(processed, args.day_type)
    table = compare(ids, power, B=args.boot)
    suffix = "" if args.day_type == "ddv7" else "_ddv1"
    name = f"paired_daytype_comparison{suffix}.csv"
    table.to_csv(out / name, index=False)
    if table.n_paired_households.notna().any():
        figure(table, figs / f"paired_daytype_comparison{suffix}.png")
    pd.set_option("display.width", 200)
    show = table[["country", "n_paired_households", "N_Class", "all_coincident_change_pct",
                  "paired_coincident_change_pct", "paired_coincident_change_lo",
                  "paired_coincident_change_hi", "all_own_peak_change_pct",
                  "paired_own_peak_change_pct"]].round(1)
    print(show.to_string(index=False))
    if not table.n_paired_households.notna().any():
        print(f"\nNo country has {MIN_N_UNFLAGGED} or more households with both day types.")
    print(f"\nWritten to {out / name}. Every value is an aggregate over at least "
          f"{MIN_N_UNFLAGGED} households; countries with fewer paired households show their size "
          "class only.")


if __name__ == "__main__":
    main()

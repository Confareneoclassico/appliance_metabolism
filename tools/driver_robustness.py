"""Robustness of the associations between clusters and household
characteristics (Reviewer 1, comment 5).

Stage 07 tests every characteristic against the clusters of each solution
with Pearson's chi-squared test and Cramer's V, treating household-days as
independent, although most households contribute two or more of them, and a
profile sums the activities of every diarist of the household, so that the
number of adults may stand for the number of diaries. This script repeats the
analysis of Stage 07 with four changes:

  Cramer's V with the bias correction of Bergsma (2013), as in the sensitivity
  analysis, beside the plain V of Stage 07;
  a 95% interval of the corrected V from resampling households, not
  household-days, so that the days of a household stay together;
  the corrected V on one household-day per household, drawn at random (the
  mean over the draws), which removes the repeated observations altogether;
  the corrected V with missing answers as a category of their own, instead of
  left out;
  the number of adults (HHC5) within each number of diarists of the
  household-day (1, 2, 3 or more), with the pooled within-stratum V, and the
  number of diarists as a characteristic of its own.

    python3 tools/driver_robustness.py [--processed FOLDER | --bundle BUNDLE]
                                       [--boot 200] [--draws 50]
                                       [--files global|countries|all]

It reads the solutions of Stages 05 and 06 (*_clustering_results.csv),
household_interval_timeseries.csv (household identifiers, in the order of the
solutions) and person_daily_metabolic_totals.csv (the diarists of each
household-day) from the processed folder of a run, so it runs on the machine
that holds the microdata. The rows of each solution are matched to the
household-days by their order and checked against their daily energy. It
writes driver_robustness.csv next to consolidated_chi2_stats.csv. Results on
fewer than 20 household-days are withheld and those on 20 to 49 flagged, item
non-response above 50% withholds a result and above 20% flags it, and strata
of the number of diarists are published only from 50 household-days, with
their size rounded to ten, as in pipeline/disclosure.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

from disclosure import (MIN_N_PUBLISH, MIN_N_UNFLAGGED, item_class, round_size,  # noqa: E402
                        size_class)
from run_folder import check, country_names, describe, resolve, solution_files, stop  # noqa: E402

FEATURES = ["DDV1", "DDV3", "DDV6", "COUNTRY", "INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
            "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"]
ENERGY = [f"Household_Energy_W_{t}" for t in range(144)]
NOT_APPLICABLE, MISSING = -98, -99
DIARIST_STRATA = {1: "1", 2: "2", 3: "3+"}


def bias_corrected_v(chi2, n, r, k):
    """Cramer's V with the bias correction of Bergsma (2013), as gsa.model."""
    if not np.isfinite(chi2) or n <= 1 or r < 2 or k < 2:
        return np.nan
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else np.nan


def table_stats(table):
    """Pearson's chi-squared (no continuity correction), size and shape of a
    contingency table with its empty rows and columns dropped."""
    t = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n = float(t.sum())
    if t.shape[0] < 2 or t.shape[1] < 2 or n <= 0:
        return np.nan, n, t.shape[0], t.shape[1]
    e = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / n
    return float(((t - e) ** 2 / e).sum()), n, t.shape[0], t.shape[1]


def plain_v(chi2, n, r, k):
    m = min(r - 1, k - 1)
    return float(np.sqrt(chi2 / (n * m))) if n * m > 0 and np.isfinite(chi2) else np.nan


def v_of(cluster, category, weights=None):
    """Plain and corrected V of two integer-coded arrays (weights: counts of
    each row, as a resample gives them)."""
    r, k = cluster.max() + 1, category.max() + 1
    t = np.bincount(cluster * k + category, weights=weights, minlength=r * k).reshape(r, k)
    chi2, n, rr, kk = table_stats(t)
    return plain_v(chi2, n, rr, kk), bias_corrected_v(chi2, n, rr, kk), n


def codes(x):
    return np.unique(np.asarray(x), return_inverse=True)[1]


def household_boot(cluster, category, household, B, rng):
    """Corrected V over B resamples of the households (their days together)."""
    hh = codes(household)
    H = hh.max() + 1
    out = np.empty(B)
    for b in range(B):
        w = rng.multinomial(H, np.full(H, 1 / H))[hh].astype(float)
        out[b] = v_of(cluster, category, w)[1]
    return out


def one_day_per_household(cluster, category, household, draws, rng):
    """Mean corrected V over random draws of one household-day per household."""
    hh = codes(household)
    order = np.argsort(hh, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(hh[order])) + 1]
    sizes = np.diff(np.r_[starts, len(hh)])
    vals = []
    for _ in range(draws):
        pick = order[starts + (rng.random(len(starts)) * sizes).astype(int)]
        vals.append(v_of(cluster[pick], category[pick])[1])
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else np.nan


def interval(boot):
    """95% percentile interval of the finite resampled values (NaN, NaN if none)."""
    boot = boot[np.isfinite(boot)]
    return tuple(np.percentile(boot, [2.5, 97.5])) if len(boot) else (np.nan, np.nan)


def values(column):
    """A characteristic as numbers, with MISSING for an empty cell; text
    categories (COUNTRY) are numbered."""
    x = pd.to_numeric(column, errors="coerce")
    if x.notna().sum() < column.notna().sum():          # text categories
        raw = column.astype(str)
        num = pd.Series(codes(raw), index=column.index).astype(float)
        num[raw.isin([str(MISSING), str(NOT_APPLICABLE), "nan"])] = np.nan
        x = num.where(~raw.isin([str(NOT_APPLICABLE)]), NOT_APPLICABLE)
    x = x.to_numpy(dtype=float)
    return np.where(np.isfinite(x), x, MISSING)


def publication(n, nonresponse_pct):
    cls = item_class(int(n), nonresponse_pct)
    return "withheld" if n < MIN_N_PUBLISH else ("flagged" if n < MIN_N_UNFLAGGED else cls)


def analyse(sol, household, n_diarists, B=200, draws=50, seed=20261126, name=""):
    """Rows of driver_robustness.csv for one solution: `sol` has the column
    cluster and the characteristics, `household` and `n_diarists` one value
    per row of `sol`."""
    rng = np.random.default_rng(seed)
    cl = codes(sol["cluster"])
    hh = np.asarray(household)
    rows = []
    feats = [f for f in FEATURES if f in sol.columns and sol[f].nunique() > 1]
    nd = np.minimum(np.asarray(n_diarists, dtype=int), 3)
    extra = {"N_DIARISTS": nd}
    for f in feats + list(extra):
        x = values(sol[f]) if f in sol.columns else extra[f]
        keep = ~np.isin(x, [MISSING, NOT_APPLICABLE])
        applies = x != NOT_APPLICABLE
        n = int(keep.sum())
        nonresp = 100 * (applies.sum() - n) / applies.sum() if applies.sum() else np.nan
        row = {"file": name, "Feature": f, "N_Observations": n if n >= MIN_N_UNFLAGGED else np.nan,
               "N_Class": size_class(n), "Item_Nonresponse_Pct": nonresp,
               "Publication_Class": publication(n, nonresp)}
        if row["Publication_Class"] != "withheld" and len(np.unique(x[keep])) > 1:
            c, v = cl[keep], codes(x[keep])
            row["V_plain"], row["V_corrected"], _ = v_of(c, v)
            boot = household_boot(c, v, hh[keep], B, rng)
            row["V_corrected_lo"], row["V_corrected_hi"] = interval(boot)
            row["V_one_day_per_household"] = one_day_per_household(c, v, hh[keep], draws, rng)
            miss = x != NOT_APPLICABLE
            if (x[miss] == MISSING).any():
                row["V_missing_as_category"] = v_of(cl[miss], codes(x[miss]))[1]
        rows.append(row)
    # the number of adults within each number of diarists
    if "HHC5" in feats:
        x = pd.to_numeric(sol["HHC5"], errors="coerce").to_numpy()
        ok = ~np.isin(x, [MISSING, NOT_APPLICABLE]) & np.isfinite(x)
        pooled, weight = 0.0, 0
        for s, label in DIARIST_STRATA.items():
            m = ok & (nd == s)
            n = int(m.sum())
            row = {"file": name, "Feature": f"HHC5 | diarists {label}",
                   "N_Observations": float(round_size(n)) if n >= MIN_N_UNFLAGGED else np.nan,
                   "N_Class": "" if n >= MIN_N_UNFLAGGED else size_class(n)}
            if n >= MIN_N_UNFLAGGED and len(np.unique(x[m])) > 1:
                c, v = cl[m], codes(x[m])
                row["V_plain"], row["V_corrected"], _ = v_of(c, v)
                boot = household_boot(c, v, hh[m], B, rng)
                row["V_corrected_lo"], row["V_corrected_hi"] = interval(boot)
                row["V_one_day_per_household"] = one_day_per_household(c, v, hh[m], draws, rng)
                if np.isfinite(row["V_corrected"]):
                    pooled += n * row["V_corrected"]
                    weight += n
                row["Publication_Class"] = ""
            else:
                row["Publication_Class"] = "one category" if n >= MIN_N_UNFLAGGED else \
                    ("withheld" if n < MIN_N_PUBLISH else "flagged")
            rows.append(row)
        rows.append({"file": name, "Feature": "HHC5 | diarists, pooled",
                     "V_corrected": pooled / weight if weight else np.nan, "Publication_Class": ""})
    out = pd.DataFrame(rows)
    # the strongest characteristic of the solution, country and diarists aside
    plain = out[out.Feature.isin([f for f in FEATURES if f != "COUNTRY"]) & out.V_corrected.notna()]
    out["rank_corrected"] = np.nan
    out.loc[plain.index, "rank_corrected"] = plain.V_corrected.rank(ascending=False, method="min")
    return out


def load_households(processed):
    """Household identifier, country, day type (Stage 06's rule), daily
    energy and number of diarists of every household-day, in the order of
    household_interval_timeseries.csv."""
    from paired_day_types import day_type
    p = Path(processed)
    parts = []
    for ch in pd.read_csv(p / "household_interval_timeseries.csv",
                          usecols=["HID", "DIARY", "COUNTRY", "DDV1", "DDV7"] + ENERGY, chunksize=20000):
        e = ch[ENERGY].to_numpy(dtype=float).sum(axis=1)
        parts.append(ch[["HID", "DIARY", "COUNTRY", "DDV1", "DDV7"]].assign(energy=e))
    ids = pd.concat(parts, ignore_index=True)
    ids["COUNTRY"] = country_names(ids.COUNTRY)          # codes of Stage 01 -> names of Stage 06
    ids["day_type"] = day_type(pd.to_numeric(ids.DDV7, errors="coerce"),
                               pd.to_numeric(ids.DDV1, errors="coerce"))
    rec = pd.read_csv(p / "person_daily_metabolic_totals.csv", usecols=["HID", "PID", "DIARY"])
    nd = rec.groupby(["HID", "DIARY"]).PID.nunique().rename("n_diarists").reset_index()
    ids = ids.merge(nd, on=["HID", "DIARY"], how="left", sort=False)
    if ids.n_diarists.isna().any():
        raise ValueError("household-days without diarists in person_daily_metabolic_totals.csv")
    return ids


def match(sol, ids, country, dtype):
    """The household-days of a solution, in its order, checked against their
    daily energy. Stage 06 leaves out the household-days without a day type,
    so a solution may hold all the household-days of its country or only
    those with a day type."""
    base = np.ones(len(ids), bool)
    if country is not None:
        base &= ids.COUNTRY.astype(str).to_numpy() == country
    dt = ids.day_type.to_numpy()
    candidates = [base & (dt == dtype)] if dtype is not None else [base, base & (dt >= 0)]
    e = sol[ENERGY].to_numpy(dtype=float).sum(axis=1)
    for m in candidates:
        sub = ids[m]
        if len(sub) == len(sol) and np.allclose(e, sub.energy.to_numpy(), rtol=1e-6, atol=1e-3):
            return sub
    raise ValueError(f"the {len(sol)} rows do not match the household-days of "
                     "household_interval_timeseries.csv in order")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processed", type=Path, default=None,
                        help="processed folder of the run (default: that of pipeline/paths.py)")
    parser.add_argument("--bundle", type=Path, default=None,
                        help="bundle of the sensitivity analysis whose run to use (its meta.json)")
    parser.add_argument("--boot", type=int, default=200, help="resamples of households")
    parser.add_argument("--draws", type=int, default=50, help="draws of one day per household")
    parser.add_argument("--files", choices=["global", "countries", "all"], default="countries")
    args = parser.parse_args()
    processed = resolve(args.processed, args.bundle)
    check(processed)
    print(describe(processed))
    ids = load_households(processed)
    out, skipped = [], []
    files = solution_files(processed, args.files)
    for path, country, dtype in files:
        sol = pd.read_csv(path)
        try:
            sub = match(sol, ids, country, dtype)
        except ValueError as e:
            skipped.append(f"{path.name}: {e}")
            print(f"{path.name}: SKIPPED, {e}")
            continue
        res = analyse(sol, sub.HID.to_numpy(), sub.n_diarists.to_numpy(), B=args.boot,
                      draws=args.draws, name=path.name)
        out.append(res)
        top = res[res.rank_corrected == 1]
        hhc5 = res[res.Feature == "HHC5"]
        print(f"{path.name}: strongest {top.Feature.iloc[0] if len(top) else '-'}"
              + (f", HHC5 corrected V {hhc5.V_corrected.iloc[0]:.3f}" if len(hhc5) and
                 np.isfinite(hhc5.V_corrected.iloc[0]) else ""))
    if not out:
        stop("No solution could be matched to the household-days of "
             "household_interval_timeseries.csv, so nothing was written:\n  " + "\n  ".join(skipped))
    table = pd.concat(out, ignore_index=True)
    dest = Path(processed) / "driver_robustness.csv"
    table.to_csv(dest, index=False)
    print(f"\n{len(out)} of {len(files)} solutions analysed. Written to {dest}")
    if skipped:
        print(f"{len(skipped)} solutions skipped (their rows do not follow the household-days of "
              "household_interval_timeseries.csv); send this message with the results.")
        sys.exit(1)


if __name__ == "__main__":
    main()

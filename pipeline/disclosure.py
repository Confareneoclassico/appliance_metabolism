"""Disclosure control for results that leave the machine holding the microdata.

Eurostat's guidelines for publications based on its scientific use files
(Access to confidential data for scientific purposes, Guidelines for
publication, section 7.7: HETUS follows the EU-SILC rules of section 7.2)
require that

  * results based on fewer than 20 observations, or on an item for which
    non-response exceeds 50%, are not published,
  * results based on 20 to 49 observations, or on an item for which
    non-response exceeds 20% but not 50%, are published only if flagged,
  * sample sizes in these two ranges are reported only as "less than 20" and
    "20 to 49", not as the actual number, and
  * a sample size below 20 cannot be derived from other published figures,
    such as row or column totals.

Every export file meant to be shared or published passes through these rules
before it is written: the cluster summaries and mean profiles of Stages 05 and
06 and the circular statistics of Stage 08 through `protect`, the largest
cluster of each country and day type (Stage 06) through `round_size`, and the
association tests of Stage 07 through `item_class`. The size thresholds can be
changed with HETUS_MIN_N_EXPORT (default 20), HETUS_FLAG_N (default 50) and
HETUS_ROUND_N (default 10), but the defaults are the ones Eurostat requires for
HETUS.

The audit of November 2026 found that these rules did not hold in the files of
version 1.0.0 (docs/PIPELINE.md Section 3.14): slot powers are whole watts, so a
published mean profile is a vector of integers divided by the size of its
group, which reveals the size, and least squares on the stratum profiles
recovers sizes even after rounding. From version 1.1.0 every cluster of fewer
than MIN_N_UNFLAGGED household-days is reported together with the largest
cluster of its solution in every export (`published_groups`), so that no
published figure refers to a group below that size or to the largest cluster
alone, and every export with group means passes `check_export` before it is
written.
"""
import os

import numpy as np
import pandas as pd

MIN_N_PUBLISH = int(os.environ.get("HETUS_MIN_N_EXPORT", "20"))
MIN_N_UNFLAGGED = int(os.environ.get("HETUS_FLAG_N", "50"))
ROUND_N = int(os.environ.get("HETUS_ROUND_N", "10"))
MAX_NONRESPONSE_PUBLISH = 50.0    # percent of the units the item applies to
MAX_NONRESPONSE_UNFLAGGED = 20.0  # percent

WITHHELD = f"<{MIN_N_PUBLISH}"
FLAGGED = f"{MIN_N_PUBLISH}-{MIN_N_UNFLAGGED - 1}"


def size_class(n):
    """'<20' for a withheld group, '20-49' for a flagged one, '' otherwise."""
    if pd.isna(n) or n < MIN_N_PUBLISH:
        return WITHHELD
    if n < MIN_N_UNFLAGGED:
        return FLAGGED
    return ""


def round_size(n):
    """Round a size, or an array of sizes, to the nearest ROUND_N."""
    return np.floor(np.asarray(n, dtype="float64") / ROUND_N + 0.5) * ROUND_N


def item_class(n_obs, nonresponse_pct=None):
    """Publication class of a result on one item: 'withheld', 'flagged' or ''.

    A result is withheld if it rests on fewer than MIN_N_PUBLISH observations
    or if more than 50% of the units the item applies to did not answer it,
    and flagged if it rests on fewer than MIN_N_UNFLAGGED observations or if
    more than 20% did not answer. `nonresponse_pct` is None for an item whose
    non-response cannot be told apart from answers that do not apply.
    """
    nr = None if nonresponse_pct is None or pd.isna(nonresponse_pct) else float(nonresponse_pct)
    if n_obs < MIN_N_PUBLISH or (nr is not None and nr > MAX_NONRESPONSE_PUBLISH):
        return "withheld"
    if n_obs < MIN_N_UNFLAGGED or (nr is not None and nr > MAX_NONRESPONSE_UNFLAGGED):
        return "flagged"
    return ""


def protect(df, n_col, stat_cols=(), group_col=None, share_col=None,
            profile_prefix="P_", count_cols=(), total_col=None, flagged_results=True):
    """Return a copy of `df` that can be shared or published.

    Adds `n_class` (see `size_class`) and `withheld` (True below
    MIN_N_PUBLISH). For a withheld group the size, the share, the columns in
    `stat_cols` and every profile column (names starting with
    `profile_prefix`) are left empty. A flagged group keeps its results,
    flagged by `n_class`, but its size and share are left empty, since only
    the size class may be reported. Wherever a set of groups (rows sharing
    `group_col`, or the whole table) contains a withheld or flagged group, the
    sizes of its other groups are rounded to the nearest ROUND_N and their
    shares recomputed from the rounded sizes in whole percentage points, so
    that the size of the withheld or flagged group cannot be recovered by
    subtracting the others from a published total. The shares are computed
    from the total in `total_col` (for instance the size of the stratum) or,
    without it, from the sum of the sizes in the set. Counts in `count_cols`
    (numbers of household-days with some property) are left empty when they
    lie between 1 and MIN_N_UNFLAGGED - 1.

    With `flagged_results=False` the results of flagged groups are left empty
    as well. Profile exports use it, since a mean of whole numbers over a group
    of 20 to 49 reveals the size that may be given only as a class.
    """
    out = df.copy()
    n = pd.to_numeric(out[n_col], errors="coerce")
    withheld = n.isna() | (n < MIN_N_PUBLISH)
    small = withheld | (n < MIN_N_UNFLAGGED)  # withheld or flagged
    out["n_class"] = [size_class(v) for v in n]
    out["withheld"] = withheld.values

    blank = [c for c in out.columns
             if c in stat_cols or (profile_prefix and str(c).startswith(profile_prefix))]
    for col in blank + ([share_col] if share_col is not None else []):
        if out[col].dtype.kind in "iub":
            out[col] = out[col].astype("float64")
    for col in blank:
        out.loc[withheld if flagged_results else small, col] = np.nan
    if share_col is not None:
        out.loc[small, share_col] = np.nan

    sets = out.groupby(group_col, sort=False).groups.values() if group_col else [out.index]
    sizes = n.astype("float64").copy()
    for idx in sets:
        idx = pd.Index(idx)
        if small.loc[idx].any():
            keep = idx[~small.loc[idx].values]
            sizes.loc[keep] = round_size(sizes.loc[keep])
            if share_col is not None:
                total = (pd.to_numeric(out.loc[idx, total_col]).iloc[0] if total_col
                         else n.loc[idx].sum())
                out.loc[keep, share_col] = (100 * sizes.loc[keep] / total).round(0)
    sizes[small] = np.nan
    out[n_col] = sizes.round().astype("Int64")
    for col in count_cols:
        counts = pd.to_numeric(out[col], errors="coerce")
        out[col] = counts.where(~((counts > 0) & (counts < MIN_N_UNFLAGGED)))
    return out


def check_strata(counts, min_n=None):
    """Raise ValueError if a stratum has between 1 and `min_n` - 1
    household-days (default MIN_N_UNFLAGGED).

    `counts` gives the household-days of every stratum (a Series indexed by
    stratum, for instance by country and day type). Stage 06 publishes the
    household-days of every country, of every country and day type and of
    each day type over all countries, so a small stratum could be derived from
    the others even if its own rows were withheld. Rather than suppress further
    figures to prevent that, the exports require every stratum to have at
    least `min_n` household-days or none; in HETUS 2010 the smallest country
    and day type has 946.
    """
    min_n = MIN_N_UNFLAGGED if min_n is None else min_n
    c = pd.Series(counts)
    bad = c[(c > 0) & (c < min_n)]
    if len(bad):
        cells = ", ".join(f"{k}: {int(v)}" for k, v in bad.items())
        raise ValueError(
            f"strata of fewer than {min_n} household-days ({cells}): their size could be "
            "derived from the published totals. Leave the country out of the run (remove "
            "its file from the raw data folder and run again from Stage 01).")


def assert_sizes(table, cols, min_n=None, name="export"):
    """Raise ValueError if a size column of a table about to be written
    holds a value between 1 and `min_n` - 1 (default MIN_N_UNFLAGGED)."""
    min_n = MIN_N_UNFLAGGED if min_n is None else min_n
    for col in cols:
        v = pd.to_numeric(table[col], errors="coerce")
        if ((v > 0) & (v < min_n)).any():
            raise ValueError(f"{name}: column {col} holds a size below {min_n}")


def published_groups(labels, min_n=None):
    """Groups under which the household-days of a clustering solution are
    published (Option A of the audit of November 2026).

    Every cluster of fewer than `min_n` (default MIN_N_UNFLAGGED) household-days
    is reported together with the largest cluster of the solution; ties for the
    largest go to the lowest label. Returns the published group of every
    household-day (the label of its cluster, or of the largest cluster for the
    members of a small one), the sorted list of the small clusters merged and
    the label of the largest cluster.
    """
    min_n = MIN_N_UNFLAGGED if min_n is None else min_n
    lab = pd.Series(np.asarray(labels))
    sizes = lab.value_counts().sort_index()
    largest = sizes.idxmax()
    small = sorted(c for c in sizes.index if sizes[c] < min_n and c != largest)
    groups = lab.where(~lab.isin(small), largest).to_numpy()
    return groups, small, largest


def recovered_size(values, n_max, tol=1e-6):
    """The smallest n up to `n_max` for which every value times n is a whole
    number, which is what an intruder reads off a mean of whole numbers over n
    units; None if there is none or the values are not all finite."""
    v = np.asarray(values, dtype="float64")
    if v.size == 0 or not np.isfinite(v).all():
        return None
    for n in range(1, int(n_max) + 1):
        x = v * n
        if np.all(np.abs(x - np.round(x)) < tol * max(1.0, n / 100.0)):
            return n
    return None


def check_export(table, n_col, profile_prefix="P_", stratum_col=None,
                 stratum_total_col=None, min_n=None):
    """Raise ValueError if a table of group results would let a size below
    `min_n` (default MIN_N_UNFLAGGED) be derived.

    Two tests are run. A row whose profile columns are means of whole numbers
    must not reveal a size below `min_n`. Within a stratum, the published or
    recovered sizes must add up to the stratum total, or leave a remainder of
    at least `min_n`, so that no small group can be obtained by subtraction.
    Rows with a withheld size are tested on the size their profile reveals.
    """
    min_n = MIN_N_UNFLAGGED if min_n is None else min_n
    prof = [c for c in table.columns if profile_prefix and str(c).startswith(profile_prefix)]
    n = pd.to_numeric(table[n_col], errors="coerce")
    problems = []
    seen = n.copy()
    if prof:
        cap = (pd.to_numeric(table[stratum_total_col], errors="coerce").max()
               if stratum_total_col else max(int(np.nanmax(n.to_numpy())) if n.notna().any() else 0, min_n))
        for i, row in table.iterrows():
            r = recovered_size(row[prof].to_numpy(), cap)
            if r is not None:
                if r < min_n:
                    problems.append(f"row {i}: its means reveal a group of {r}")
                if pd.isna(seen.loc[i]):
                    seen.loc[i] = r
    if (n.dropna() < min_n).any():
        problems.append(f"{int((n.dropna() < min_n).sum())} published size(s) below {min_n}")
    if stratum_total_col:
        totals = pd.to_numeric(table[stratum_total_col], errors="coerce")
        if (totals < min_n).any():
            problems.append(f"a stratum of fewer than {min_n} household-days")
    if stratum_col and stratum_total_col:
        for key, g in table.groupby(stratum_col, sort=False):
            total = pd.to_numeric(g[stratum_total_col], errors="coerce").iloc[0]
            got = seen.loc[g.index]
            if got.isna().any():
                continue
            rest = total - got.sum()
            if 0 < rest < min_n:
                problems.append(f"stratum {key}: the published sizes leave {int(rest)} household-days")
    if problems:
        raise ValueError("export would disclose small groups: " + "; ".join(problems[:10]))

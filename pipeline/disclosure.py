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
            profile_prefix="P_", count_cols=(), total_col=None):
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
        out.loc[withheld, col] = np.nan
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

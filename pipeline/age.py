"""Age of the respondent (INC2) in the 15 groups of the scientific use files.

Up to the release of November 2020, the scientific use files of HETUS 2010
give INC2 in 15 groups: 1 = 10-17, 2 = 18-19, then five-year groups from
3 = 20-24 to 14 = 75-79, and 15 = 80 and over (anonymisation guide of
23 January 2020, Annex 2). The release of 5 November 2026 gives INC2 in
completed years instead, with 10 for ages 10 to 17 and 80 for 80 and over
(release notes of 5 November 2026). The age in completed years determines the
group, so Stage 01 maps the new coding onto the 15 groups and every later
stage sees the coding of the earlier release: Stages 01 and 02 select the
metabolic rate by group (1 teenager, 2 to 11 adult, 12 to 15 senior), Stages
05 to 07, Stage 11 and the sensitivity analysis treat the groups as
categories, and no output gives age in single years. Without the mapping,
the earlier code would give the adult rate to diarists aged 10 to 17 and the
senior rate to every diarist aged 18 or over, and would test age with up to
64 categories instead of 15.

HETUS_INC2_CODING gives the coding of the raw files: "auto" (the default)
decides from the values of each national file, "groups" or "years" sets it.
Negative codes (not asked, refusal, not available) and empty cells are left
as they are. A value that fits neither coding stops the run.
"""
import os

import numpy as np
import pandas as pd

N_GROUPS = 15
GROUP_LABELS = {1: "10-17", 2: "18-19", **{3 + i: f"{20 + 5 * i}-{24 + 5 * i}" for i in range(12)},
                15: "80+"}
# Completed years that can occur: the files hold ages 10 to 85 before
# top-coding (11 to 85 in France, 10 to 74 in Luxembourg, 10 to 80 in Norway,
# 15 and over in Serbia); the release of November 2026 codes 10-17 as 10 and
# 80 and over as 80, but any age in the range maps to its group.
MIN_YEARS, MAX_YEARS = 10, 85
CODINGS = ("groups", "years")
DESCRIPTION = {"groups": "the 15 age groups (release of November 2020)",
               "years": "completed years (release of 5 November 2026), mapped to the 15 age groups"}


def coding_from_environment():
    """The coding set by HETUS_INC2_CODING (None for "auto")."""
    value = os.environ.get("HETUS_INC2_CODING", "auto").strip().lower()
    if value in ("", "auto"):
        return None
    if value not in CODINGS:
        raise ValueError(f"HETUS_INC2_CODING={value!r}: use auto, groups or years")
    return value


def group_of_age(years):
    """Group code (1 to 15) of ages in completed years (10 or more)."""
    y = np.asarray(years, dtype=float)
    return np.select([y < 18, y < 20, y < 80], [1, 2, 3 + (y - 20) // 5], default=N_GROUPS).astype(int)


def _valid(values):
    """The numeric values and the mask of those that are answers (>= 0)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    return v, v.notna() & (v >= 0)


def detect_coding(values):
    """'years' if some answer is above 15, the largest group code, else
    'groups'."""
    v, valid = _valid(values)
    return "years" if valid.any() and v[valid].max() > N_GROUPS else "groups"


def harmonise_inc2(values, coding=None, where="INC2"):
    """INC2 in the 15 groups, and the coding the values were read in.

    `coding` is "groups", "years" or None (decide from the values). Answers
    that fit neither (for instance 0, a non-integer, a year outside 10 to 85,
    or group codes mixed with years) raise ValueError."""
    if coding is None:
        coding = detect_coding(values)
    if coding not in CODINGS:
        raise ValueError(f"{where}: unknown coding {coding!r}")
    s = pd.Series(values)
    v, valid = _valid(s)
    whole = valid & (v == np.floor(v))
    if coding == "groups":
        ok = whole & v.between(1, N_GROUPS)
    else:
        ok = whole & v.between(MIN_YEARS, MAX_YEARS)
    bad = valid & ~ok
    if bad.any():
        examples = ", ".join(str(x) for x in sorted(v[bad].unique())[:5])
        raise ValueError(f"{where}: {int(bad.sum())} value(s) that are not {DESCRIPTION[coding]} "
                         f"(for instance {examples}); check the file or set HETUS_INC2_CODING")
    if coding == "groups":
        return s, coding
    out = v.copy()
    out[valid] = group_of_age(v[valid])
    if out.notna().all():
        out = out.astype(np.int64)
    return out, coding


def check_groups(values, where):
    """Stop if INC2 holds answers outside the 15 groups, as a processed file
    written from the release of November 2026 by a version of Stage 01
    without the mapping would."""
    v, valid = _valid(values)
    bad = valid & ~((v == np.floor(v)) & v.between(1, N_GROUPS))
    if bad.any():
        raise ValueError(f"{where}: INC2 has {int(bad.sum())} value(s) outside the 15 age groups "
                         f"(for instance {sorted(v[bad].unique())[:5]}); rerun Stage 01, which maps "
                         "ages in completed years to the groups")

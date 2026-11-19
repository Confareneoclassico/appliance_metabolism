"""Stage 09: how often the members of a household share an activity at home.

The statistic shows how often the shared-appliance rule of Stage 03 is
exercised. For every household-day with two or more diarists, a slot is
comparable when at least two diarists are at home (Wherep = 1) with a valid
activity code, and shared when at least two of those diarists record the same
code, as main or secondary activity. Code 999 (other unspecified time use,
which includes "no activity recorded") is not a valid code: two diarists
whose activities are both unspecified are not known to share one. Each diarist counts once per code, so a
code recorded by one diarist as both main and secondary activity is not a
second diarist. The share of a household-day is its shared slots over its
comparable slots.

Versions up to 1.0.1 counted all 144 slots in the denominator, whether or not
two diarists had a valid code in them, never read the location, so members at
their own workplaces with the same work code shared an activity, and counted a
code recorded as both main and secondary activity twice (audit of November
2026, D3). The figures quoted in the manuscript came from a vectorised version
that was not in the repository.

Outputs (working files, not to be shared: one row per household-day):
    same_activity_household_summary.csv
Aggregate outputs:
    shared_activities_frequency.csv   shared (slot, code) pairs and the
                                      household-days they come from, by
                                      activity code; codes shared in fewer
                                      than MIN_N_UNFLAGGED household-days are
                                      pooled as "other"
    activity_combination_counts.csv   main for every diarist, secondary for
                                      every diarist, or mixed
    same_activity_by_diarists.csv     mean and median share by number of
                                      diarists, under the publication rules
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import PROCESSED_DIR, FIGURES_DIR
from disclosure import size_class, round_size, MIN_N_PUBLISH, MIN_N_UNFLAGGED

SLOTS = 144
MACT = [f"Mact{i}" for i in range(1, SLOTS + 1)]
SACT = [f"Sactn{i}" for i in range(1, SLOTS + 1)]
WHERE = [f"Wherep{i}" for i in range(1, SLOTS + 1)]
NA16 = np.int16(-32768)   # an empty cell in the int16 arrays
CODE_BASE = 10000         # activity codes are below this
MAX_DIARISTS = 64         # diarist records per household-day are below this
UNSPECIFIED = 999         # other unspecified time use: not a valid code here
INPUT = PROCESSED_DIR / "lu_data_at_home_with_dominant_activity.csv"


def to_int16(frame):
    a = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    return np.where(np.isnan(a), NA16, a).astype(np.int16)


def read_records(path=INPUT):
    """Identifiers and the 144-slot main, secondary and location codes of
    every diarist record, as compact arrays."""
    ids, M, S, W = [], [], [], []
    header = pd.read_csv(path, nrows=0).columns
    for c in SACT:
        if c not in header:
            raise ValueError(f"{path} has no column {c}")
    for ch in pd.read_csv(path, usecols=["HID", "PID", "DIARY"] + MACT + SACT + WHERE,
                          chunksize=20000):
        ids.append(ch[["HID", "PID", "DIARY"]].copy())
        M.append(to_int16(ch[MACT]))
        S.append(to_int16(ch[SACT]))
        W.append(to_int16(ch[WHERE]))
    return pd.concat(ids, ignore_index=True), np.vstack(M), np.vstack(S), np.vstack(W)


def shared_slots(g, d, t, codes, flags):
    """For entries (group g, diarist d, slot t, code, flag 0 = main or
    1 = secondary), the slot ids (g * 144 + t) in which at least two distinct
    diarists of the group have the same code, and for each shared
    (slot, code) its code, whether it is main for every diarist (0),
    secondary for every diarist (1) or mixed (2), and its group."""
    key = (g.astype(np.int64) * SLOTS + t) * CODE_BASE + codes
    pair = np.unique((key * MAX_DIARISTS + d) * 2 + flags)   # main sorts first
    kd = pair // 2
    first = np.ones(len(kd), bool)
    first[1:] = kd[1:] != kd[:-1]                             # one entry per (key, diarist)
    kd, fl = kd[first], (pair % 2)[first]
    k = kd // MAX_DIARISTS
    uk, start, cnt = np.unique(k, return_index=True, return_counts=True)
    sh = cnt >= 2
    nsec = np.add.reduceat(fl, start) if len(start) else np.array([], int)
    kind = np.where(nsec == 0, 0, np.where(nsec == cnt, 1, 2))[sh]
    sk = uk[sh]
    return np.unique(sk // CODE_BASE), sk % CODE_BASE, kind, sk // CODE_BASE // SLOTS


def compute_sharing(ids, M, S, W, batch=20000):
    """One row per multi-diarist household-day (HID, DIARY): number of
    diarists, comparable slots, shared slots and the share in %, with the
    counts of shared (slot, code) pairs by code and by kind, and the
    distinct (household-day, code) pairs with a shared slot, as an array of
    (row of the summary, code)."""
    hid = ids["HID"].astype(str).to_numpy()
    pid = ids["PID"].astype(str).to_numpy()
    key = pd.Series(hid) + "|" + ids["DIARY"].astype(str).values
    g_all, uniques = pd.factorize(key)
    order = np.argsort(g_all, kind="stable")
    g_all = g_all[order]
    Mo, So, Wo = M[order], S[order], W[order]
    hid_o, pid_o = hid[order], pid[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g_all)) + 1]
    sizes = np.diff(np.r_[starts, len(g_all)])
    # number of distinct persons per household-day (repeated records count once)
    persons = np.array([len(set(pid_o[a:a + n])) for a, n in zip(starts, sizes)])
    multi = np.flatnonzero(persons >= 2)
    rows = {"key": [], "n_diarists": [], "comparable_slots": [], "shared_slots": []}
    code_counts = np.zeros(CODE_BASE)
    kinds = np.zeros(3)
    day_codes = []
    for b in range(0, len(multi), batch):
        grp = multi[b:b + batch]
        idx = np.concatenate([np.arange(starts[i], starts[i] + sizes[i]) for i in grp])
        gl = np.repeat(np.arange(len(grp)), sizes[grp])
        # diarist index within the household-day: one per person
        dl = np.concatenate([pd.factorize(pid_o[starts[i]:starts[i] + sizes[i]])[0] for i in grp])
        m, s, w = Mo[idx], So[idx], Wo[idx]
        home = w == 1
        a = (m > 0) & (m != UNSPECIFIED) & home
        c = (s > 0) & (s != UNSPECIFIED) & home
        T = np.broadcast_to(np.arange(SLOTS), m.shape)
        G = np.broadcast_to(gl[:, None], m.shape)
        D = np.broadcast_to(dl[:, None], m.shape)
        sl, codes, kind, pair_group = shared_slots(
            np.r_[G[a], G[c]], np.r_[D[a], D[c]], np.r_[T[a], T[c]],
            np.r_[m[a], s[c]].astype(np.int64),
            np.r_[np.zeros(a.sum(), np.int64), np.ones(c.sum(), np.int64)])
        ng = len(grp)
        shared = np.bincount(sl // SLOTS, minlength=ng)
        # comparable: two or more distinct diarists at home with a valid code
        has = a | c
        gtd = np.unique((G[has].astype(np.int64) * SLOTS + T[has]) * MAX_DIARISTS + D[has])
        gt, cnt = np.unique(gtd // MAX_DIARISTS, return_counts=True)
        comparable = np.bincount(gt[cnt >= 2] // SLOTS, minlength=ng)
        code_counts += np.bincount(codes, minlength=CODE_BASE)[:CODE_BASE]
        kinds += np.bincount(kind, minlength=3)[:3]
        dc = np.unique(pair_group * CODE_BASE + codes)
        day_codes.append(np.c_[b + dc // CODE_BASE, dc % CODE_BASE])
        rows["key"].extend(uniques[g_all[starts[grp]]])
        rows["n_diarists"].extend(persons[grp])
        rows["comparable_slots"].extend(comparable)
        rows["shared_slots"].extend(shared)
    out = pd.DataFrame(rows)
    out[["HID", "DIARY"]] = out["key"].str.split("|", n=1, expand=True)
    out = out.drop(columns="key")
    with np.errstate(invalid="ignore", divide="ignore"):
        out["same_activity_percentage"] = np.where(
            out.comparable_slots > 0, 100 * out.shared_slots / out.comparable_slots, np.nan)
    day_codes = np.vstack(day_codes) if day_codes else np.zeros((0, 2), dtype=np.int64)
    return (out[["HID", "DIARY", "n_diarists", "comparable_slots", "shared_slots",
                 "same_activity_percentage"]], code_counts, kinds, day_codes)


def by_diarists(summary):
    """Mean and median share by number of diarists (7 and more pooled), with
    sizes and results under the publication rules: groups of fewer than
    MIN_N_UNFLAGGED household-days are given only as a size class and without
    results, and the other sizes are rounded wherever such a group exists."""
    s = summary.dropna(subset=["same_activity_percentage"]).copy()
    s["diarists"] = np.where(s.n_diarists >= 7, "7+", s.n_diarists.astype(str))
    t = s.groupby("diarists").same_activity_percentage.agg(["size", "mean", "median"])
    t = t.rename(columns={"size": "household_days"})
    small = t.household_days < MIN_N_UNFLAGGED
    t["n_class"] = [size_class(n) for n in t.household_days]
    # Results of groups below MIN_N_UNFLAGGED are withheld, not only flagged:
    # a mean over a small group can reveal its size
    t.loc[small, ["mean", "median"]] = np.nan
    if small.any():
        t.loc[~small, "household_days"] = round_size(t.loc[~small, "household_days"])
    t.loc[small, "household_days"] = np.nan
    return t.reset_index()


def code_table(code_counts, day_codes, min_n=MIN_N_UNFLAGGED):
    """Shared (slot, code) pairs and the number of household-days they come
    from, by activity code, most frequent first. Codes shared in fewer than
    `min_n` household-days are pooled as "other", whose household-days are
    left empty when they are fewer than `min_n`, so that no count rests on
    fewer than `min_n` household-days."""
    pairs = pd.Series(code_counts).astype(int)
    pairs = pairs[pairs > 0]
    days = pd.Series(day_codes[:, 1]).value_counts() if len(day_codes) else pd.Series(dtype=int)
    days = days.reindex(pairs.index).fillna(0).astype(int)
    big = days[days >= min_n].index
    table = pd.DataFrame({"activity": big.astype(str), "frequency": pairs[big].values,
                          "household_days": days[big].values})
    table = table.sort_values("frequency", ascending=False)
    small = pairs.index.difference(big)
    if len(small):
        other_days = len(np.unique(day_codes[np.isin(day_codes[:, 1], small), 0]))
        table.loc[len(table)] = {"activity": "other", "frequency": int(pairs[small].sum()),
                                 "household_days": other_days if other_days >= min_n else np.nan}
    return table.reset_index(drop=True)


def make_figure(summary, table, path):
    s = summary.dropna(subset=["same_activity_percentage"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(s.same_activity_percentage, bins=20, color="#2a78d6", edgecolor="white")
    ax[0].set_xlabel("Shared share of comparable slots at home (%)")
    ax[0].set_ylabel("Household-days")
    shown = table[table["n_class"] == ""]
    ax[1].bar(shown.diarists, shown["mean"], color="#2a78d6")
    ax[1].set_xlabel("Number of diarists")
    ax[1].set_ylabel("Mean share (%)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Reading {INPUT}")
    ids, M, S, W = read_records()
    summary, code_counts, kinds, day_codes = compute_sharing(ids, M, S, W)
    summary.to_csv(PROCESSED_DIR / "same_activity_household_summary.csv", index=False)
    table = by_diarists(summary)
    table.to_csv(PROCESSED_DIR / "same_activity_by_diarists.csv", index=False)

    tot = kinds.sum()
    kinds_s = pd.Series(kinds, index=["main for every diarist", "secondary for every diarist",
                                      "mixed"], name="count")
    kinds_s.to_csv(PROCESSED_DIR / "activity_combination_counts.csv", header=["count"])
    freq = code_table(code_counts, day_codes)
    freq.to_csv(PROCESSED_DIR / "shared_activities_frequency.csv", index=False)

    s = summary.same_activity_percentage.dropna()
    print(f"Multi-diarist household-days: {len(summary)}; with at least one comparable slot: {len(s)}")
    print(f"Share of comparable slots at home with a shared activity: mean {s.mean():.1f}%, "
          f"median {s.median():.1f}%")
    print(table.to_string(index=False))
    if tot:
        print("Shared (slot, code) pairs: " + ", ".join(
            f"{k} {100 * v / tot:.1f}%" for k, v in kinds_s.items()))
        print("Most frequently shared codes: " + ", ".join(
            f"{r.activity} ({100 * r.frequency / tot:.1f}%)"
            for r in freq[freq.activity != "other"].head(5).itertuples()))
    make_figure(summary, table, FIGURES_DIR / "same_activity_analysis.png")


if __name__ == "__main__":
    main()

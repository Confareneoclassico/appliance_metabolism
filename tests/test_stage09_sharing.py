"""Stage 09: shared activities at home among the diarists of a household-day
(D3), with diarists counted as distinct persons per household-day (L14)."""
import importlib

import numpy as np
import pandas as pd

stage09 = importlib.import_module("09_same_diarists_coincidence")
SLOTS = 144


def build(records):
    """records: (HID, PID, DIARY, main, secondary, where), each of the last
    three a scalar or a 144-vector."""
    ids, M, S, W = [], [], [], []
    for hid, pid, diary, m, s, w in records:
        ids.append({"HID": hid, "PID": pid, "DIARY": diary})
        M.append(np.broadcast_to(m, SLOTS))
        S.append(np.broadcast_to(s, SLOTS))
        W.append(np.broadcast_to(w, SLOTS))
    as16 = lambda a: np.array(a, dtype=np.int16)
    return pd.DataFrame(ids), as16(M), as16(S), as16(W)


def half(a, b):
    return np.r_[np.full(72, a), np.full(72, b)]


RECORDS = [
    # A: both at home watching TV all day: every slot comparable and shared
    ("A", 1, 1, 821, 0, 1), ("A", 2, 1, 821, 0, 1),
    # B: asleep at home in the first half, at work with the same code in the
    # second half: only the slots at home are comparable
    ("B", 1, 1, half(11, 1110), 0, half(1, 2)), ("B", 2, 1, half(11, 1110), 0, half(1, 2)),
    # C: TV as main and secondary activity of one diarist is not shared
    ("C", 1, 1, 821, 821, 1), ("C", 2, 1, 11, 0, 1),
    # D: TV as main activity of one and secondary of the other is shared
    ("D", 1, 1, 821, 0, 1), ("D", 2, 1, 21, 821, 1),
    # E: three diarists, only one at home in the first half
    ("E", 1, 1, 21, 0, 1), ("E", 2, 1, 11, 0, half(2, 1)), ("E", 3, 1, 311, 0, half(2, 1)),
    # F: one diarist: not a multi-diarist household-day
    ("F", 1, 1, 821, 0, 1),
    # G: one of the two has no valid code (0): no slot is comparable
    ("G", 1, 1, 0, 0, 1), ("G", 2, 1, 821, 0, 1),
    # H: person 1 recorded twice, person 2 asleep: the repeated record is
    # not a second diarist
    ("H", 1, 1, 821, 0, 1), ("H", 1, 1, 821, 0, 1), ("H", 2, 1, 11, 0, 1),
    # I: the same two persons on two diary days are two household-days
    ("I", 1, 1, 821, 0, 1), ("I", 2, 1, 821, 0, 1),
    ("I", 1, 2, 821, 0, 1), ("I", 2, 2, 11, 0, 1),
]


def result():
    ids, M, S, W = build(RECORDS)
    summary, codes, kinds, day_codes = stage09.compute_sharing(ids, M, S, W)
    return summary.set_index(["HID", "DIARY"]), codes, kinds, day_codes


def test_shares_by_household_day():
    s, _, _, _ = result()
    expect = {  # (HID, DIARY): (n_diarists, comparable, shared)
        ("A", "1"): (2, 144, 144), ("B", "1"): (2, 72, 72), ("C", "1"): (2, 144, 0),
        ("D", "1"): (2, 144, 144), ("E", "1"): (3, 72, 0), ("G", "1"): (2, 0, 0),
        ("H", "1"): (2, 144, 0), ("I", "1"): (2, 144, 144), ("I", "2"): (2, 144, 0),
    }
    assert set(s.index) == set(expect)        # F has one diarist
    for key, (nd, comp, sh) in expect.items():
        row = s.loc[key]
        assert (row.n_diarists, row.comparable_slots, row.shared_slots) == (nd, comp, sh), key
    assert np.isnan(s.loc[("G", "1"), "same_activity_percentage"])
    assert s.loc[("B", "1"), "same_activity_percentage"] == 100.0


def test_kinds_and_codes_of_shared_pairs():
    s, codes, kinds, day_codes = result()
    # shared (slot, code) pairs: A 144, B 72, D 144, I-1 144 (all TV or sleep)
    assert kinds.tolist() == [144 + 72 + 144, 0, 144]   # D is mixed
    assert codes[821] == 144 * 3 and codes[11] == 72
    # household-days in which each code is shared: TV in A, D and I-1, sleep in B
    keys = s.reset_index()[["HID", "DIARY"]].apply(tuple, axis=1).to_numpy()
    shared_in = {(keys[r], c) for r, c in day_codes}
    assert shared_in == {(("A", "1"), 821), (("D", "1"), 821), (("I", "1"), 821),
                         (("B", "1"), 11)}


def test_rare_codes_are_pooled():
    counts = np.zeros(stage09.CODE_BASE)
    counts[[11, 821, 311, 321]] = [9000, 3000, 40, 15]
    day_codes = np.r_[np.c_[np.arange(300), np.full(300, 11)],     # sleep in 300 days
                      np.c_[np.arange(120), np.full(120, 821)],    # TV in 120
                      np.c_[np.arange(30), np.full(30, 311)],      # cooking in 30
                      np.c_[np.arange(25, 35), np.full(10, 321)]]  # cleaning in 10
    t = stage09.code_table(counts, day_codes)
    assert t.activity.tolist() == ["11", "821", "other"]
    assert t.household_days.tolist()[:2] == [300, 120]
    other = t.iloc[-1]
    assert other.frequency == 55 and np.isnan(other.household_days)  # 35 days < 50


def test_table_by_diarists_hides_small_groups():
    s, _, _, _ = result()
    table = stage09.by_diarists(s.reset_index())
    assert table["mean"].isna().all()              # every group has fewer than 50
    assert table["household_days"].isna().all()
    assert set(table["n_class"]) == {"<20"}


def test_unspecified_time_use_is_not_a_shared_activity():
    # J: both at home with main code 999 (other unspecified time use, which
    # includes "no activity recorded") in the first half and TV in the
    # second: only the second half is comparable, and all of it is shared.
    # K: secondary 999 for both diarists, main sleep for one and eating for
    # the other: comparable, nothing shared.
    records = [("J", 1, 1, half(999, 821), 0, 1), ("J", 2, 1, half(999, 821), 0, 1),
               ("K", 1, 1, 11, 999, 1), ("K", 2, 1, 21, 999, 1)]
    ids, M, S, W = build(records)
    s, codes, _, _ = stage09.compute_sharing(ids, M, S, W)
    s = s.set_index(["HID", "DIARY"])
    assert (s.loc[("J", "1"), "comparable_slots"], s.loc[("J", "1"), "shared_slots"]) == (72, 72)
    assert (s.loc[("K", "1"), "comparable_slots"], s.loc[("K", "1"), "shared_slots"]) == (144, 0)
    assert codes[999] == 0

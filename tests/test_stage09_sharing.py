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
    summary, codes, kinds = stage09.compute_sharing(ids, M, S, W)
    return summary.set_index(["HID", "DIARY"]), codes, kinds


def test_shares_by_household_day():
    s, _, _ = result()
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
    _, codes, kinds = result()
    # shared (slot, code) pairs: A 144, B 72, D 144, I-1 144 (all TV or sleep)
    assert kinds.tolist() == [144 + 72 + 144, 0, 144]   # D is mixed
    assert codes[821] == 144 * 3 and codes[11] == 72


def test_table_by_diarists_hides_small_groups():
    s, _, _ = result()
    table = stage09.by_diarists(s.reset_index())
    assert table["mean"].isna().all()              # every group has fewer than 50
    assert table["household_days"].isna().all()
    assert set(table["n_class"]) == {"<20"}

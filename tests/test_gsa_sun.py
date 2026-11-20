"""Sunrise, sunset and darkness of the lighting factor X19 (gsa.sun), against
published times of sunrise and sunset in local clock time."""
import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsa import sun  # noqa: E402

# Published times (timeanddate.com, rounded to the minute) at the capitals
KNOWN = [
    ("UK", dt.date(2010, 6, 21), "04:43", "21:21"),
    ("UK", dt.date(2010, 12, 21), "08:04", "15:53"),
    ("FI", dt.date(2010, 12, 21), "09:24", "15:12"),
    ("BE", dt.date(2010, 6, 21), "05:29", "22:00"),
    ("NO", dt.date(2010, 6, 21), "03:54", "22:44"),
    ("EL", dt.date(2010, 12, 21), "07:37", "17:10"),
]


def hours(text):
    h, m = text.split(":")
    return int(h) + int(m) / 60


@pytest.mark.parametrize("country,date,rise,set_", KNOWN)
def test_sun_times_within_two_minutes(country, date, rise, set_):
    r, s = sun.sun_times_local(country, date)
    assert abs(r - hours(rise)) < 2 / 60 and abs(s - hours(set_)) < 2 / 60


def test_summer_time_follows_the_european_rule():
    assert not sun.summer_time(dt.date(2010, 3, 27)) and sun.summer_time(dt.date(2010, 3, 28))
    assert sun.summer_time(dt.date(2010, 10, 30)) and not sun.summer_time(dt.date(2010, 10, 31))


def test_darkness_by_slot():
    june, december = sun.darkness("BE", 2010, 6), sun.darkness("BE", 2010, 12)
    slot = lambda h, m=0: ((h - 4) % 24) * 6 + m // 10  # noqa: E731  (diary day from 04:00)
    assert june.shape == (144,) and ((june >= 0) & (june <= 1)).all()
    assert june[slot(12)] == 0 and june[slot(1)] == 1          # midday light, 01:00 dark
    assert december[slot(17, 30)] == 1 and december[slot(8)] == 1
    assert june.sum() < december.sum()
    march = sun.darkness("BE", 2010, 3)                          # sunrise moves through the month
    assert ((march > 0) & (march < 1)).any()

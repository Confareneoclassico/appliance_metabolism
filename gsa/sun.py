"""Darkness by country, month and 10-minute slot, for the lighting factor X19.

Sunrise and sunset are those of the capital, from the NOAA approximation of
the solar position (accurate to a minute or two at these latitudes), with the
standard upper-limb horizon of 90.833 degrees. Clock time follows the
country's standard time and the European summer-time rule (from the last
Sunday of March to the last Sunday of October). The day of the month of a
diary is not in the scientific use files, so darkness is the share of the days
of the diary month on which the middle of the slot falls before sunrise or
after sunset. The diary day starts at 04:00; the slots after midnight are
compared with the same day's sunrise.
"""
import calendar
import datetime as dt

import numpy as np

# Latitude and longitude (degrees, east positive) of the capitals
CAPITALS = {
    "BE": (50.85, 4.35), "DE": (52.52, 13.40), "EE": (59.44, 24.75), "EL": (37.98, 23.73),
    "ES": (40.42, -3.70), "FI": (60.17, 24.94), "FR": (48.86, 2.35), "HU": (47.50, 19.04),
    "IT": (41.90, 12.50), "LU": (49.61, 6.13), "NL": (52.37, 4.90), "NO": (59.91, 10.75),
    "PL": (52.23, 21.01), "RO": (44.43, 26.10), "RS": (44.79, 20.45), "UK": (51.51, -0.13),
}
# Standard time, hours east of UTC
STANDARD_OFFSET = {
    "BE": 1, "DE": 1, "ES": 1, "FR": 1, "HU": 1, "IT": 1, "LU": 1, "NL": 1, "NO": 1,
    "PL": 1, "RS": 1, "EE": 2, "EL": 2, "FI": 2, "RO": 2, "UK": 0,
}
SLOTS = 144
# Clock time at the middle of each slot, in hours from 00:00 (04:05, 04:15, ...)
SLOT_MID = (4 + (np.arange(SLOTS) + 0.5) / 6) % 24


def _last_sunday(year, month):
    last = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last)
    return d - dt.timedelta(days=(d.weekday() - 6) % 7)


def summer_time(date):
    """European summer time on this date (from the last Sunday of March to the
    day before the last Sunday of October)."""
    return _last_sunday(date.year, 3) <= date < _last_sunday(date.year, 10)


def sun_times_utc(lat, lon, date):
    """Sunrise and sunset in hours UTC (NOAA approximation). NaN if the sun
    does not rise or set that day."""
    n_days = 366 if calendar.isleap(date.year) else 365
    g = 2 * np.pi / n_days * (date.timetuple().tm_yday - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    phi = np.radians(lat)
    x = np.cos(np.radians(90.833)) / (np.cos(phi) * np.cos(decl)) - np.tan(phi) * np.tan(decl)
    if abs(x) > 1:
        return np.nan, np.nan
    ha = np.degrees(np.arccos(x))
    rise = (720 - 4 * (lon + ha) - eqtime) / 60
    set_ = (720 - 4 * (lon - ha) - eqtime) / 60
    return rise, set_


def sun_times_local(country, date):
    """Sunrise and sunset at the capital, in local clock hours."""
    lat, lon = CAPITALS[country]
    rise, set_ = sun_times_utc(lat, lon, date)
    off = STANDARD_OFFSET[country] + (1 if summer_time(date) else 0)
    return rise + off, set_ + off


def darkness(country, year, month):
    """Share of the days of the month on which the middle of each of the 144
    slots of the diary day is before sunrise or after sunset, at the capital
    of the country. Germany's months come grouped by quarter in the files,
    and any month is accepted here."""
    days = calendar.monthrange(int(year), int(month))[1]
    dark = np.zeros(SLOTS)
    for d in range(1, days + 1):
        rise, set_ = sun_times_local(country, dt.date(int(year), int(month), d))
        dark += (SLOT_MID < rise) | (SLOT_MID > set_)
    return dark / days

"""Daily peak times of household-day power profiles and their circular
statistics.

Activity-based power is piecewise constant, so the daily maximum of a
household-day is often reached in several slots, contiguous or not. Versions
up to 1.0.1 took the first of them (idxmax, argmax), which moved peak times
towards the start of each plateau and towards the morning (audit of November
2026, defect D4). Here the unit weight of each household-day is shared equally
among the slots at its maximum, and household-days without any demand, which
have no peak, get no weight (they used to enter at 04:00).

Slot 0 covers 04:00-04:09, since the HETUS diary day starts at 04:00.
"""
import numpy as np

DIARY_START_HOUR = 4
SLOTS = 144
SLOT_HOURS = 24 / SLOTS
CLOCK_HOURS = (DIARY_START_HOUR + np.arange(SLOTS) * SLOT_HOURS) % 24
THETA = 2 * np.pi * CLOCK_HOURS / 24


def peak_weights(energy):
    """Weights of shape (n, 144) that share the unit weight of each
    household-day equally among the slots at its daily maximum. Rows without
    demand are all zero. Returns the weights and a mask of the rows with
    demand."""
    e = np.asarray(energy, dtype=float)
    mx = e.max(axis=1)
    tie = (e == mx[:, None]) & (mx[:, None] > 0)
    m = tie.sum(axis=1)
    has = m > 0
    w = np.zeros_like(e)
    w[has] = tie[has] / m[has, None]
    return w, has


def circular_peak_stats(energy, min_days=2):
    """Circular mean (clock hours, 0 to 24) and circular standard deviation
    (radians, sqrt(-2 ln R) as in scipy.stats.circstd) of the daily peak time
    of a set of household-days, with ties shared and days without demand left
    out, and the number of days without demand. The mean and SD are NaN when
    fewer than `min_days` household-days have demand."""
    w, has = peak_weights(energy)
    n_zero = int((~has).sum())
    if has.sum() < min_days:
        return np.nan, np.nan, n_zero
    slot_w = w.sum(axis=0)
    c, s = float(slot_w @ np.cos(THETA)), float(slot_w @ np.sin(THETA))
    r = np.hypot(c, s) / slot_w.sum()
    sd = float(np.sqrt(-2.0 * np.log(min(max(r, 1e-300), 1.0))))
    mean_h = float((np.arctan2(s, c) % (2 * np.pi)) * 24 / (2 * np.pi))
    return mean_h, sd, n_zero


def clock_str(hours):
    """Clock time HH:MM of a time in hours, rounded to the minute and reduced
    modulo 24 hours (a mean of exactly 24.0 h prints 00:00)."""
    if hours is None or not np.isfinite(hours):
        return ""
    m = int(np.floor(hours * 60 + 0.5)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"

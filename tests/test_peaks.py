"""Peak times with tied maxima and days without demand (D4), and clock times
rounded modulo 24 hours (L13)."""
import numpy as np

from peaks import CLOCK_HOURS, circular_peak_stats, clock_str, peak_weights


def test_slot_zero_is_four_o_clock():
    assert CLOCK_HOURS[0] == 4.0
    assert CLOCK_HOURS[120] == 0.0   # 20 hours after 04:00


def test_tied_maxima_share_the_weight():
    e = np.zeros((1, 144))
    e[0, [12, 96]] = 500.0           # 06:00 and 20:00, equal maxima
    w, has = peak_weights(e)
    assert has.tolist() == [True]
    assert w[0, 12] == w[0, 96] == 0.5
    assert w.sum() == 1.0


def test_plateau_is_not_moved_to_its_start():
    # A plateau from 18:00 to 19:50: the first-slot rule of versions up to
    # 1.0.1 gave 18:00, the shared rule gives the middle of the plateau
    e = np.zeros((2, 144))
    e[:, 84:96] = 1000.0
    mean_h, sd, n_zero = circular_peak_stats(e)
    assert abs(mean_h - (18 + 55 / 60)) < 1e-9
    assert n_zero == 0
    assert sd > 0


def test_days_without_demand_are_left_out():
    e = np.zeros((3, 144))
    e[0, 90] = 300.0                 # 19:00
    e[1, 90] = 200.0                 # 19:00
    mean_h, sd, n_zero = circular_peak_stats(e)
    assert n_zero == 1
    assert abs(mean_h - 19.0) < 1e-9  # no weight at 04:00
    assert sd < 1e-6


def test_too_few_days_with_demand_give_nan():
    e = np.zeros((2, 144))
    e[0, 10] = 1.0
    mean_h, sd, n_zero = circular_peak_stats(e)
    assert np.isnan(mean_h) and np.isnan(sd) and n_zero == 1


def test_clock_str_rounds_and_wraps():
    assert clock_str(18.999) == "19:00"       # rounded, not truncated to 18:59
    assert clock_str(23.9999) == "00:00"      # a mean of 2 pi prints 00:00, not 24:00
    assert clock_str(4 + 5 / 60) == "04:05"
    assert clock_str(float("nan")) == ""

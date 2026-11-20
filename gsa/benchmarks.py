"""Hourly benchmark profiles behind outputs Y9 to Y11 of the sensitivity
analysis, and the shape metrics of the validation of 19 November 2026.

Values are mean loads by local clock hour (00 to 23) over all days, as built
by build_benchmarks.py of the validation (outputs/revision/validation):

SYNERGRID_S21_BE   Synergrid synthetic load profile S21 (residential,
                   below 1.3 MWh), 2012 and 2013, 729 complete days; fractions
                   of annual use per hour (SLPU workbooks, synergrid.be)
ENEDIS_RES1_FR     Enedis dynamic load profile RES1_BASE, 2022 to 2025, 1,457
                   complete days; coefficients per hour (Enedis open data,
                   Licence Ouverte)

Only shapes are compared, so the units do not matter.
"""
import numpy as np

SYNERGRID_S21_BE = np.array([
    2.34183521947874e-05, 1.947367445130317e-05, 1.685358792866941e-05,
    1.5592778566529487e-05, 1.5234090912208493e-05, 1.6088936659807962e-05,
    1.965950493827158e-05, 2.5551138477366225e-05, 2.8081361625514375e-05,
    2.9754704286694085e-05, 3.087563168724278e-05, 3.382083528806584e-05,
    3.393485895061727e-05, 3.120883106995882e-05, 2.9165529595336057e-05,
    2.8335724074074067e-05, 3.0431355555555566e-05, 3.5921851783264734e-05,
    3.9552786385459555e-05, 4.040438161865572e-05, 3.873735658436214e-05,
    3.658536334019204e-05, 3.522285668724278e-05, 2.998069245541424e-05])

ENEDIS_RES1_FR = np.array([
    0.694061125200069, 0.6171626640169876, 0.5757649823267835, 0.554429240406657,
    0.5506926847393617, 0.5737234958817781, 0.6558221336098152, 0.7913282220983189,
    0.8868506764330825, 0.9383172963490042, 0.9707275607210031, 1.0534937359106038,
    1.1284526458364774, 1.0389132868244666, 0.9494167666769037, 0.9001234001992104,
    0.903499401947664, 0.9738615430962622, 1.146841126517158, 1.3090905905561108,
    1.2613918785734364, 1.1268270137469123, 0.9749347784610499, 0.8184355355355181])

# Clock hour of each 10-minute slot of the diary day, which starts at 04:00
CLOCK_HOUR = (4 + np.arange(144) // 6) % 24


def hourly(profile):
    """Mean of a 144-slot profile by clock hour 00 to 23."""
    p = np.asarray(profile, dtype=float)
    return np.array([p[CLOCK_HOUR == h].mean() for h in range(24)])


def shape(x):
    """Hourly shape as in the validation: minimum removed, sum one."""
    y = np.asarray(x, dtype=float) - np.min(x)
    return y / y.sum()


def shape_correlation(profile, benchmark):
    """Pearson correlation of the hourly shapes of a 144-slot profile and a
    24-hour benchmark (r_shape of the validation)."""
    return float(np.corrcoef(shape(hourly(profile)), shape(benchmark))[0, 1])


def evening_to_midday(profile):
    """Ratio of the evening maximum (16:00 to 23:59) to the midday maximum
    (10:00 to 14:59) of the hourly shape (eve_to_midday of the validation)."""
    s = shape(hourly(profile))
    return float(s[16:24].max() / s[10:15].max())

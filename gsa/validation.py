"""Agreement with the measured profiles and uncertainty bands of energy and
peak power, from the shared outputs of the runs (no microdata):

    python3 -m gsa.validation --share SHARE --canonical EXPORTS --out RESULTS

SHARE is the folder of gsa.collect (gsa_profiles.npz: mean profile of every
country and day type in every run) and EXPORTS the folder of
tools/collect_exports.py for the canonical run
(mean_profiles_country_daytype.csv). Two tables are written:

  validation_metrics.csv  for Belgium (Synergrid S21) and France (Enedis RES1),
                          the metrics Reviewer 1 asks for (comment 1): the
                          correlation of the hourly shapes, the error in the
                          hour of the daily maximum and of the evening maximum
                          (16:00 to 23:59), both circular, in hours, and the
                          normalised root-mean-square error of the hourly
                          profiles scaled to a daily sum of one (the RMSE over
                          the mean of the measured profile); canonical run,
                          median and 5th to 95th percentile of the runs, and
                          the share of runs whose daily maximum falls in the
                          evening
  country_bands.csv       for every country (and all together) and day type,
                          the mean daily energy (kWh) and the coincident peak
                          per household (W, maximum of the mean profile) of the
                          canonical run with the median and 5th to 95th
                          percentile of the runs (Reviewer 2, comment 7)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gsa import benchmarks

BENCH = {"BE": ("Synergrid S21", benchmarks.SYNERGRID_S21_BE),
         "FR": ("Enedis RES1", benchmarks.ENEDIS_RES1_FR)}
EVENING = np.arange(16, 24)


def hour_error(h, ref):
    """Signed circular difference h - ref in hours, in (-12, 12]."""
    d = (np.asarray(h) - ref) % 24
    return np.where(d > 12, d - 24, d)


def metrics(profile, bench):
    """Shape correlation, error of the hour of the daily and of the evening
    maximum, and NRMSE of one 144-slot profile against a 24-hour benchmark."""
    h = benchmarks.hourly(profile)
    p, b = h / h.sum(), bench / bench.sum()
    return {"correlation": benchmarks.shape_correlation(profile, bench),
            "peak_hour_error_h": float(hour_error(np.argmax(h), np.argmax(bench))),
            "evening_peak_hour_error_h": float(hour_error(16 + np.argmax(h[EVENING]),
                                                          16 + np.argmax(bench[EVENING]))),
            "nrmse": float(np.sqrt(np.mean((p - b) ** 2)) / b.mean()),
            "evening_maximum": float(np.argmax(h) >= 16)}


def canonical_profile(canonical, country, day_type="all"):
    r = canonical[(canonical.country == country) & (canonical.day_type == day_type)]
    return r[[f"P_{t:03d}" for t in range(144)]].to_numpy(dtype=float)[0] if len(r) else None


def validation_metrics(z, canonical):
    countries, days = list(z["countries"]), list(z["day_types"])
    rows = []
    for c, (name, bench) in BENCH.items():
        runs = pd.DataFrame([metrics(p, bench) for p in z["profiles"][:, countries.index(c), days.index("all"), :]])
        canon = metrics(canonical_profile(canonical, c), bench) if canonical is not None else {}
        for m in runs.columns:
            v = runs[m].to_numpy()
            row = {"country": c, "benchmark": name, "metric": m, "canonical": canon.get(m, np.nan)}
            if m == "evening_maximum":
                row["share_of_runs"] = float(v.mean())
            else:
                row.update({"median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
                            "p95": float(np.percentile(v, 95))})
                if m.endswith("error_h"):
                    row["mean_absolute"] = float(np.abs(v).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def country_bands(z, canonical):
    countries, days = list(z["countries"]), list(z["day_types"])
    prof = z["profiles"].astype(float)
    rows = []
    for i, c in enumerate(countries):
        for j, d in enumerate(days):
            p = prof[:, i, j, :]
            energy = p.sum(axis=1) / 6 / 1000            # kWh: 10-minute slots
            peak = p.max(axis=1)
            row = {"country": c, "day_type": d}
            cp = canonical_profile(canonical, c, d) if canonical is not None else None
            for name, v, cv in (("energy_kwh", energy, None if cp is None else cp.sum() / 6 / 1000),
                                ("coincident_peak_w", peak, None if cp is None else cp.max())):
                row[f"{name}_canonical"] = np.nan if cv is None else cv
                row[f"{name}_median"] = float(np.median(v))
                row[f"{name}_p05"], row[f"{name}_p95"] = np.percentile(v, [5, 95])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--share", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, default=None,
                        help="folder with mean_profiles_country_daytype.csv of the canonical run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    z = np.load(args.share / "gsa_profiles.npz")
    canonical = None
    if args.canonical is not None:
        canonical = pd.read_csv(args.canonical / "mean_profiles_country_daytype.csv")
        canonical["country"] = canonical.country.replace({"All": "ALL", "all": "ALL"})
    args.out.mkdir(parents=True, exist_ok=True)
    v = validation_metrics(z, canonical)
    b = country_bands(z, canonical)
    v.to_csv(args.out / "validation_metrics.csv", index=False)
    b.to_csv(args.out / "country_bands.csv", index=False)
    pd.set_option("display.width", 200)
    print(v.round(3).to_string(index=False))
    print(b[b.day_type == "all"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

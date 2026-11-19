import glob
import math
import os

import numpy as np
import pandas as pd

from paths import PROCESSED_DIR, COUNTRY_DIR, DAYTYPE_DIR, COUNTRY_DAYTYPE_DIR
from disclosure import protect, published_groups, MIN_N_UNFLAGGED
from peaks import circular_peak_stats, clock_str

# --- Configuration ---
ENERGY_COLUMNS = [f"Household_Energy_W_{i}" for i in range(144)]
# List of files to process: the global solution (Stage 05) and the two
# day-type solutions (Stage 06)
FILES_TO_PROCESS = [str(PROCESSED_DIR / "clustering_results.csv"),
    str(DAYTYPE_DIR / "Working_Days_clustering_results.csv"),
    str(DAYTYPE_DIR / "Non_Working_Days_clustering_results.csv"),
]
# Country solutions and country-by-day-type solutions (Stage 06), named by the
# two-letter country code. [A-Z][A-Z] excludes the duplicate *_ALL_DATA_* files.
country_files = sorted(glob.glob(str(COUNTRY_DIR / "[A-Z][A-Z]_clustering_results.csv")))
country_work_files = sorted(glob.glob(str(COUNTRY_DAYTYPE_DIR / "[A-Z][A-Z]_working_days_clustering_results.csv")))
country_nonwork_files = sorted(glob.glob(str(COUNTRY_DAYTYPE_DIR / "[A-Z][A-Z]_non_working_days_clustering_results.csv")))
FILES_TO_PROCESS.extend(country_files)
FILES_TO_PROCESS.extend(country_work_files)
FILES_TO_PROCESS.extend(country_nonwork_files)
# HETUS diary days run from 04:00 to 03:59 the next day, and slot 0 of each
# energy series covers 04:00-04:09 (SUF variable Mact001). Peak times are
# converted to clock time by adding this offset. Earlier versions treated slot
# 0 as midnight, which put every circular mean four hours early (the circular
# standard deviation is unaffected by the offset).
DIARY_START_HOUR = 4
# The output follows Eurostat's publication rules for HETUS (disclosure.py).
# From version 1.1.0 the statistics are computed for the published groups of
# each solution: every cluster of fewer than 50 household-days is reported with
# the largest cluster (published_groups), so that every group in the file has
# at least 50 household-days, its size can be given exactly, and no small
# cluster can be derived by subtraction from the solution's total.
CIRCULAR_STAT_COLS = ['n_zero_demand', 'circular_mean_time',
                      'circular_std_rad', 'circular_std_deg']
# --- End Configuration ---

def circular_statistics_summary(df, energy_cols, file_name):
    """Circular mean and standard deviation of the daily peak time for each
    published group of a solution. Ties at the daily maximum share the weight
    of their household-day and days without demand are left out (peaks.py,
    audit defect D4)."""
    print(f"\n--- Circular Statistics Summary for: {file_name} ---")
    groups, small, largest = published_groups(df['cluster'])
    energy = df[energy_cols].to_numpy(dtype=float)
    summary_data = []
    for g in sorted(pd.unique(groups)):
        members = groups == g
        mean_h, std_rad, n_zero = circular_peak_stats(energy[members])
        summary_data.append({
            'file': file_name,
            'cluster': int(g),
            'n_small_clusters_merged': len(small) if g == largest else 0,
            'n_households': int(members.sum()),
            # Household-days with no electricity demand all day have no peak
            # and are left out of the statistics
            'n_zero_demand': n_zero,
            'circular_mean_time': clock_str(mean_h),
            'circular_std_rad': std_rad,
            'circular_std_deg': math.degrees(std_rad) if np.isfinite(std_rad) else np.nan,
        })
    out = pd.DataFrame(summary_data)
    if out['n_households'].sum() != len(df) or (out['n_households'] < MIN_N_UNFLAGGED).any():
        raise ValueError(f"{file_name}: the published groups do not cover the solution "
                         f"or one has fewer than {MIN_N_UNFLAGGED} household-days")
    return out


def main():
    all_results = []
    for file_path in FILES_TO_PROCESS:
        if not os.path.exists(file_path):
            print(f"Skipping file: {file_path} (Not Found)")
            continue
            
        try:
            df = pd.read_csv(file_path)
            # Ensure the required columns are present
            if 'cluster' not in df.columns or not all(col in df.columns for col in ENERGY_COLUMNS):
                print(f"Skipping file: {file_path} (Missing 'cluster' or energy columns)")
                continue
                
            results = circular_statistics_summary(df, ENERGY_COLUMNS, os.path.basename(file_path))
            all_results.append(results)
            
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    if all_results:
        final_df = protect(pd.concat(all_results, ignore_index=True), n_col='n_households',
                           stat_cols=CIRCULAR_STAT_COLS, group_col='file',
                           count_cols=['n_zero_demand'])
        print("\n--- Consolidated Circular Statistics Results ---")
        print(final_df.to_markdown(index=False))
        final_df.to_csv(PROCESSED_DIR / "consolidated_circular_stats.csv", index=False)
        print(f"\nConsolidated results saved to {PROCESSED_DIR}/consolidated_circular_stats.csv")
    else:
        print("No files were successfully processed for circular statistics.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from scipy.stats import circstd, circmean
import math
import os
import glob
from paths import PROCESSED_DIR, COUNTRY_DIR, DAYTYPE_DIR, COUNTRY_DAYTYPE_DIR
from disclosure import protect

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
# The output follows Eurostat's publication rules for HETUS (disclosure.py):
# clusters of fewer than 20 household-days are listed without size or
# statistics, clusters of 20 to 49 are flagged and listed without their size,
# and the other sizes are rounded within a solution in which a cluster is
# withheld or flagged.
CIRCULAR_STAT_COLS = ['n_zero_demand', 'circular_mean_time',
                      'circular_std_rad', 'circular_std_deg']
# --- End Configuration ---

def calculate_peak_time_in_radians(df, energy_cols):
    """
    Calculates the time of peak energy consumption for each household and converts it to radians.
    """
    # Find the index (0 to 143) of the maximum energy consumption
    peak_index = df[energy_cols].idxmax(axis=1).str.replace('Household_Energy_W_', '', regex=False).astype(int)
    
    # Convert slot index to clock hour (0 to 23.9833...), slot 0 = 04:00
    peak_hour = (DIARY_START_HOUR + peak_index * (24 / 144)) % 24
    
    # Convert hour (0-24) to radians (0-2*pi)
    peak_time_radians = (peak_hour / 24) * 2 * math.pi
    
    return peak_time_radians

def circular_statistics_summary(df, energy_cols, file_name):
    """
    Calculates circular mean and standard deviation of peak time for each cluster.
    """
    print(f"\n--- Circular Statistics Summary for: {file_name} ---")
    
    # Calculate peak time in radians for all households
    df['peak_time_rad'] = calculate_peak_time_in_radians(df, energy_cols)
    
    summary_data = []
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        
        if len(cluster_data) < 2:
            summary_data.append({'file': file_name, 'cluster': cluster_id,
                                 'n_households': len(cluster_data)})
            continue
            
        angles = cluster_data['peak_time_rad'].values
        
        # Calculate circular mean (mu) and standard deviation (rho)
        mean_rad = circmean(angles)
        std_rad = circstd(angles)
        
        # Convert mean back to hour (0-24)
        mean_hour = (mean_rad / (2 * math.pi)) * 24
        
        # Convert mean hour to HH:MM format
        hours = int(mean_hour)
        minutes = int((mean_hour - hours) * 60)
        mean_time_str = f"{hours:02d}:{minutes:02d}"
        
        summary_data.append({
            'file': file_name,
            'cluster': cluster_id,
            'n_households': len(cluster_data),
            # Household-days with no electricity demand all day have no peak;
            # idxmax assigns them slot 0 (04:00). Reported so their weight in
            # the statistics can be judged.
            'n_zero_demand': int((cluster_data[energy_cols].sum(axis=1) == 0).sum()),
            'circular_mean_time': mean_time_str,
            'circular_std_rad': std_rad,
            'circular_std_deg': math.degrees(std_rad)
        })
        
    return pd.DataFrame(summary_data).sort_values(by='cluster')

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

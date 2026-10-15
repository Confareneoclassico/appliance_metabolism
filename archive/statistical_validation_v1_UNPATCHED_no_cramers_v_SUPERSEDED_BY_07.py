import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import numpy as np
import os
import glob

# --- Configuration ---
# Full list of categorical features identified in the original script
CATEGORICAL_FEATURES = [
    "DDV1", "DDV3", "DDV6", "COUNTRY", "INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
    "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"
]
ENERGY_COLUMNS = [f"Household_Energy_W_{i}" for i in range(144)]

# List of files to process
FILES_TO_PROCESS = ["./clustering_results.csv",
    "./DayType_Analysis/Working_Days_clustering_results.csv",
    "./DayType_Analysis/Non_Working_Days_clustering_results.csv",
]
# Add all country-specific files (Country_X.0_clustering_results.csv)
# Assuming country files are in ./Country_Analysis/
country_files = glob.glob("./Country_Analysis/Country_*.0_clustering_results.csv")
country_work_files = glob.glob("./Country_DayType_Analysis/Country_*.0_working_days_clustering_results.csv")
country_nonwork_files = glob.glob("./Country_DayType_Analysis/Country_*.0_non_working_days_clustering_results.csv")
FILES_TO_PROCESS.extend(country_files)
FILES_TO_PROCESS.extend(country_work_files)
FILES_TO_PROCESS.extend(country_nonwork_files)
# --- End Configuration ---

def calculate_daily_energy(df, energy_cols):
    """Calculates the total daily energy consumption."""
    return df[energy_cols].sum(axis=1)

def validate_categorical_drivers(df, cat_features, file_name):
    """Performs Chi-squared test for independence between cluster and categorical features."""
    print(f"\n--- Chi-squared Test Results for: {file_name} ---")
    chi2_results = []
    
    for feature in cat_features:
        if feature not in df.columns:
            continue
        
        # Drop rows where the feature value is the missing value indicator (-99)
        # This is important for Chi-squared validity
        temp_df = df[df[feature] != -99].copy()
        
        if len(temp_df) < 10:
            continue
            
        # Create contingency table
        contingency_table = pd.crosstab(temp_df['cluster'], temp_df[feature])
        
        # Check if the table is valid for Chi-squared (at least 2x2)
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue
            
        # Perform Chi-squared test
        try:
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            chi2_results.append({
                'file': file_name,
                'Feature': feature,
                'Chi2_Statistic': chi2,
                'P_Value': p,
                'Degrees_of_Freedom': dof,
                'Min_Expected_Count': expected.min(),
                'Significant_at_0.05': p < 0.05
            })
        except ValueError:
            # Handle cases where the table is too sparse or has zero rows/columns
            pass
            
    return pd.DataFrame(chi2_results).sort_values(by='P_Value')

def validate_continuous_drivers(df, energy_cols, file_name):
    """Performs ANOVA test on total daily energy consumption across clusters."""
    print(f"\n--- ANOVA Test Results for: {file_name} ---")
    
    # Calculate total daily energy
    df['Total_Daily_Energy'] = calculate_daily_energy(df, energy_cols)
    
    # Prepare data for ANOVA: list of energy values for each cluster
    cluster_groups = [df['Total_Daily_Energy'][df['cluster'] == c].values for c in df['cluster'].unique()]
    
    # Remove empty groups if any
    cluster_groups = [group for group in cluster_groups if len(group) > 0]
    
    if len(cluster_groups) < 2:
        return None
        
    # Perform one-way ANOVA
    f_statistic, p_value = f_oneway(*cluster_groups)
    
    anova_result = {
        'file': file_name,
        'Metric': 'Total_Daily_Energy',
        'F_Statistic': f_statistic,
        'P_Value': p_value,
        'Significant_at_0.05': p_value < 0.05
    }
    
    return pd.DataFrame([anova_result])

def main():
    all_chi2_results = []
    all_anova_results = []
    
    for file_path in FILES_TO_PROCESS:
        if not os.path.exists(file_path):
            print(f"Skipping file: {file_path} (Not Found)")
            continue
            
        try:
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            
            # 1. Categorical Validation
            chi2_results = validate_categorical_drivers(df, CATEGORICAL_FEATURES, file_name)
            if not chi2_results.empty:
                all_chi2_results.append(chi2_results)
            
            # 2. Continuous Validation
            anova_results = validate_continuous_drivers(df, ENERGY_COLUMNS, file_name)
            if anova_results is not None:
                all_anova_results.append(anova_results)
            
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    if all_chi2_results:
        final_chi2_df = pd.concat(all_chi2_results, ignore_index=True)
        print("\n--- Consolidated Chi-squared Results ---")
        print(final_chi2_df.to_markdown(index=False))
        final_chi2_df.to_csv("consolidated_chi2_stats.csv", index=False)
        print("\nConsolidated Chi-squared results saved to consolidated_chi2_stats.csv")
    else:
        print("No files were successfully processed for Chi-squared statistics.")
        
    if all_anova_results:
        final_anova_df = pd.concat(all_anova_results, ignore_index=True)
        print("\n--- Consolidated ANOVA Results ---")
        print(final_anova_df.to_markdown(index=False))
        final_anova_df.to_csv("consolidated_anova_stats.csv", index=False)
        print("\nConsolidated ANOVA results saved to consolidated_anova_stats.csv")
    else:
        print("No files were successfully processed for ANOVA statistics.")

if __name__ == "__main__":
    main()

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import numpy as np
import os
import glob
from paths import PROCESSED_DIR, COUNTRY_DIR, DAYTYPE_DIR, COUNTRY_DAYTYPE_DIR
from disclosure import item_class, MIN_N_PUBLISH

# --- Configuration ---
# Full list of categorical features identified in the original script
CATEGORICAL_FEATURES = [
    "DDV1", "DDV3", "DDV6", "COUNTRY", "INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
    "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"
]
ENERGY_COLUMNS = [f"Household_Energy_W_{i}" for i in range(144)]
# Eurostat's publication rules for HETUS (disclosure.py) also cover item
# non-response: a test is withheld if more than 50% of the household-days the
# variable applies to have no answer, and flagged if more than 20% have none.
# Non-response is counted among the household-days of the countries that
# collected the variable, since a variable absent from a country's microdata is
# not non-response. Job type and contract type apply only to people with a job,
# and HETUS codes missing and not-applicable answers alike (both become -99), so
# their non-response cannot be measured and is not assessed.
NOT_APPLICABLE_ITEMS = {"IND6_1", "IND44"}

# List of files to process: the global solution (Stage 05) and the two
# day-type solutions (Stage 06)
FILES_TO_PROCESS = [str(PROCESSED_DIR / "clustering_results.csv"),
    str(DAYTYPE_DIR / "Working_Days_clustering_results.csv"),
    str(DAYTYPE_DIR / "Non_Working_Days_clustering_results.csv"),
]
# Country solutions and country-by-day-type solutions (Stage 06). Stage 06 names
# them by the two-letter country code (e.g. PL_clustering_results.csv,
# PL_working_days_clustering_results.csv). The pattern [A-Z][A-Z] excludes the
# duplicate PL_ALL_DATA_clustering_results.csv written by Stage 06's first pass.
country_files = sorted(glob.glob(str(COUNTRY_DIR / "[A-Z][A-Z]_clustering_results.csv")))
country_work_files = sorted(glob.glob(str(COUNTRY_DAYTYPE_DIR / "[A-Z][A-Z]_working_days_clustering_results.csv")))
country_nonwork_files = sorted(glob.glob(str(COUNTRY_DAYTYPE_DIR / "[A-Z][A-Z]_non_working_days_clustering_results.csv")))
FILES_TO_PROCESS.extend(country_files)
FILES_TO_PROCESS.extend(country_work_files)
FILES_TO_PROCESS.extend(country_nonwork_files)
# --- End Configuration ---

def calculate_daily_energy(df, energy_cols):
    """Calculates the total daily energy consumption."""
    return df[energy_cols].sum(axis=1)

def cramers_v_effect_strength(v):
    """Interpret Cramer's V effect size using the same bands as the
    global-level analysis (2025_12_02_clustering_analysis.py), so that
    effect sizes are comparable across all stratification levels."""
    if v < 0.1:
        return "Negligible"
    elif v < 0.2:
        return "Weak"
    elif v < 0.3:
        return "Moderate"
    elif v < 0.4:
        return "Relatively Strong"
    else:
        return "Strong"

def validate_categorical_drivers(df, cat_features, file_name):
    """Performs Chi-squared test for independence between cluster and categorical features,
    together with Cramer's V effect size."""
    print(f"\n--- Chi-squared Test Results for: {file_name} ---")
    chi2_results = []
    low_expected = []
    
    for feature in cat_features:
        if feature not in df.columns:
            continue
        
        # Drop rows where the feature value is the missing value indicator (-99)
        # This is important for Chi-squared validity
        temp_df = df[df[feature] != -99].copy()
        
        if len(temp_df) < MIN_N_PUBLISH:
            continue
            
        # Create contingency table
        contingency_table = pd.crosstab(temp_df['cluster'], temp_df[feature])
        
        # Check if the table is valid for Chi-squared (at least 2x2)
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue
            
        # Perform Chi-squared test
        try:
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # Cramer's V effect size, using the same formula as the
            # global-level analysis: V = sqrt(chi2 / (n * min(r-1, c-1)))
            n = contingency_table.sum().sum()
            k = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
            cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
            effect_strength = cramers_v_effect_strength(cramers_v)

            if feature in NOT_APPLICABLE_ITEMS:
                nonresponse_pct = np.nan
            else:
                collected = df['COUNTRY'].isin(temp_df['COUNTRY'].unique()) \
                    if 'COUNTRY' in df.columns else pd.Series(True, index=df.index)
                nonresponse_pct = 100 * (1 - len(temp_df) / collected.sum())
            publication = item_class(int(n), nonresponse_pct)
            # The smallest expected count is not exported: with the published
            # totals it would give the size of the smallest cluster, which may
            # be withheld, and of the smallest category. It is only counted.
            low_expected.append(expected.min() < 5)
            if publication == "withheld":
                chi2 = p = cramers_v = np.nan
                effect_strength = significant = None
            else:
                significant = p < 0.05
            chi2_results.append({
                'file': file_name,
                'Feature': feature,
                'Chi2_Statistic': chi2,
                'P_Value': p,
                'Degrees_of_Freedom': dof,
                'Significant_at_0.05': significant,
                'Cramers_V': cramers_v,
                'Effect_Strength': effect_strength,
                'N_Clusters': contingency_table.shape[0],
                'N_Categories': contingency_table.shape[1],
                'N_Observations': int(n),
                'Item_Nonresponse_Pct': nonresponse_pct,
                'Publication_Class': publication
            })
        except ValueError:
            # Handle cases where the table is too sparse or has zero rows/columns
            pass
            
    if low_expected:
        print(f"Tables with an expected count below five: {sum(low_expected)} of {len(low_expected)}")
    if not chi2_results:
        return pd.DataFrame()
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
        final_chi2_df.to_csv(PROCESSED_DIR / "consolidated_chi2_stats.csv", index=False)
        print(f"\nConsolidated Chi-squared results saved to {PROCESSED_DIR}/consolidated_chi2_stats.csv")
    else:
        print("No files were successfully processed for Chi-squared statistics.")
        
    if all_anova_results:
        final_anova_df = pd.concat(all_anova_results, ignore_index=True)
        print("\n--- Consolidated ANOVA Results ---")
        print(final_anova_df.to_markdown(index=False))
        final_anova_df.to_csv(PROCESSED_DIR / "consolidated_anova_stats.csv", index=False)
        print(f"\nConsolidated ANOVA results saved to {PROCESSED_DIR}/consolidated_anova_stats.csv")
    else:
        print("No files were successfully processed for ANOVA statistics.")

if __name__ == "__main__":
    main()

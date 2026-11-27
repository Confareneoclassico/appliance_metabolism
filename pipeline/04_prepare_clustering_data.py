import pandas as pd
import numpy as np
from paths import PROCESSED_DIR
from age import check_groups

# Load the household time series data
household_timeseries_df = pd.read_csv(PROCESSED_DIR / "household_interval_timeseries.csv")
# Age enters the tests and the sensitivity analysis in its 15 groups (age.py)
check_groups(household_timeseries_df["INC2"], "household_interval_timeseries.csv")

# Select relevant categorical variables for clustering
categorical_cols = [
    "DDV1", # Day of the week
    "DDV3", # Month of the year
    "DDV6", # Employment/student status
    "COUNTRY", # Country of residence
    "INC4_1", # Main activity status
    "IND6_1", # Employment status in main job
    "HHC1", # Household size
    "HHC3", # Number of persons aged <7 years
    "HHC4", # Number of persons aged 7–17 years
    "HHC5", # Number of persons aged 18+ years
    "HHQ5", # Home ownership status
    "HHQ9_1", # Household net monthly income band
    "INC1", # Sex
    "INC2", # Age group
    "IND22_1", # Educational attainment level
    "IND44" # Permanent vs fixed-term contract, main job (employees only;
            # not citizenship -- that is IND42_1, which is not used)
]

# Auxiliary columns are carried through to Stages 05-06 but are NOT clustering
# features. DDV7 (kind of day: 1 work day, 2 school day, 3 day off, 4 sick
# leave, 5 vacation, 6 other leave) is used only to assign the day type in
# Stage 06. It is missing (-99) where the question was not asked.
auxiliary_cols = ["DDV7"]

# Extract energy time series columns (Household_Energy_W_0 to Household_Energy_W_143)
energy_columns = [col for col in household_timeseries_df.columns if col.startswith("Household_Energy_W_")]

# Combine energy time series, categorical features and auxiliary columns
clustering_data = household_timeseries_df[energy_columns + categorical_cols + auxiliary_cols].copy()

# Missing values and negative codes. The SUF code -1 (not applicable) becomes
# NOT_APPLICABLE (-98) and every other negative code or empty cell MISSING
# (-99), so that Stage 07 can count item non-response among the units an item
# applies to (audit of November 2026, D5). Both are left out of the tests.
NOT_APPLICABLE, MISSING = -98, -99
for col in categorical_cols + auxiliary_cols:
    x = pd.to_numeric(clustering_data[col], errors="coerce")
    if x.notna().sum() < clustering_data[col].notna().sum():
        # non-numeric codes: only empty cells are recoded
        clustering_data[col] = clustering_data[col].fillna(MISSING)
        continue
    not_applicable = x == -1
    missing = x.isna() | ((x < 0) & ~not_applicable)
    x = x.mask(not_applicable, NOT_APPLICABLE).mask(missing, MISSING)
    clustering_data[col] = x


# Convert categorical and auxiliary columns to appropriate types
for col in categorical_cols + auxiliary_cols:
    clustering_data[col] = clustering_data[col].astype("category")

# Use absolute energy values (not normalized) to preserve magnitude information
# This captures both profile shape AND consumption level
energy_data = clustering_data[energy_columns].values

# Display info about the prepared data
print("\nClustering Data Info:")
clustering_data.info()
print(f"\nNumber of energy time series features: {len(energy_columns)}")
print(f"Number of categorical features: {len(categorical_cols)}")
print(f"Auxiliary (non-feature) columns: {auxiliary_cols}")
print("\nPower statistics (mean W in each 10-min interval):")
print(f"  Min: {energy_data.min():.2f}")
print(f"  Max: {energy_data.max():.2f}")
print(f"  Mean: {energy_data.mean():.2f}")
print(f"  Std: {energy_data.std():.2f}")

# Save the prepared data for clustering
clustering_data.to_csv(PROCESSED_DIR / "prepared_clustering_data.csv", index=False)
print(f"Prepared clustering data saved to {PROCESSED_DIR}/prepared_clustering_data.csv")
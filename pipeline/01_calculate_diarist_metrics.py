import pandas as pd
import numpy as np
import gc
import os
from paths import RAW_DIR, PROCESSED_DIR, MAPPING_FILE
from countries import COUNTRIES, CODE

# Variables whose negative codes are recoded below (all but -1 become missing).
# The labels are descriptive only; versions up to 1.0.1 named DDV1 Gender,
# DDV3 Age group and DDV6 Education level (audit of November 2026, L12). DDV7
# and IND44 are not listed: their negative codes are left for Stage 04.
SOCIO_LABELS = {
        'COUNTRY': 'Country',
        'DDV1': 'Day of the week of the diary day (1 = Sunday, 7 = Saturday)',
        'DDV3': 'Month of the diary day (recoded below to the first month of its quarter)',
        'DDV6': 'Worked or studied (DDV7 is asked when DDV6 = 1)',
        'HHC1': 'Household size',
        'HHC3': 'Number of infants',
        'HHC4': 'Number of children',
        'HHC5': 'Number of persons aged 18 or over',
        'HHQ5': 'Tenure status (own or rent)',
        'HHQ9_1': 'Household income',
        'INC1': 'Sex',
        'INC2': 'Age group',
        'INC4_1': 'Main activity status',
        'IND6_1': 'Job type',
        'IND22_1': 'Educational attainment',
        'IND42_1': 'Citizenship'
    }

# Canonical names of every raw column used downstream. National files are
# matched to these case-insensitively: the Romanian file delivers several
# variables under lower-case names (e.g. 'ddv1', 'inc4_1'), which pd.concat
# placed in separate columns that were later dropped, so those variables
# appeared entirely missing for Romania.
BASE_COLUMNS = ["YEAR", "HID", "PID", "DIARY", "COUNTRY", "DDV1", "DDV3", "DDV6", "DDV7",
                "INC1", "INC2", "INC3", "INC4_1", "IND6_1", "HHQ5", "HHQ9_1", "IND22_1",
                "IND44", "HHC1", "HHC3", "HHC4", "HHC5"]
SERIES_PREFIXES = ["Mact", "Sactn", "Wherep", "Wpartner", "Wchild", "Wotherh"]
CANONICAL_COLUMNS = BASE_COLUMNS + [f"{p}{i}" for p in SERIES_PREFIXES for i in range(1, 145)]
CANONICAL_BY_KEY = {c.upper(): c for c in CANONICAL_COLUMNS}

DIAGNOSTIC_COLUMNS = ["DDV1", "DDV3", "DDV6", "DDV7", "HHC1", "HHC3", "HHC4", "HHC5",
                      "HHQ5", "HHQ9_1", "INC1", "INC2", "INC4_1", "IND6_1", "IND22_1", "IND44"]


def harmonise_column_names(df, country_code):
    """Rename columns that differ from a canonical name only in case or
    surrounding whitespace. A column is left unchanged when its canonical
    name is already present, so no duplicate columns are created."""
    present = set(df.columns)
    mapping = {}
    for col in df.columns:
        canon = CANONICAL_BY_KEY.get(str(col).strip().upper())
        if canon is None or col == canon:
            continue
        if canon in present or canon in mapping.values():
            print(f"  WARNING [{country_code}]: '{col}' duplicates '{canon}' and was left unchanged")
            continue
        mapping[col] = canon
    if mapping:
        examples = ", ".join(f"{k}->{v}" for k, v in list(mapping.items())[:8])
        print(f"  [{country_code}] {len(mapping)} column name(s) aligned to canonical case: {examples}")
    return df.rename(columns=mapping)


def report_variable_coverage(df):
    """Print, by country, the percentage of records with a valid (non-negative)
    code for each sociodemographic variable. A 0 means the variable is unusable
    for that country, either because it was not collected or because it was
    lost on loading. Check this table before running the later stages."""
    cols = [c for c in DIAGNOSTIC_COLUMNS if c in df.columns]
    valid = df[cols].apply(pd.to_numeric, errors="coerce").ge(0)
    coverage = (100 * valid.groupby(df["COUNTRY"]).mean()).round(0).astype(int)
    print("\nRecords with a valid code (%), by country:")
    print(coverage.to_string())
    print()


# Load the datasets
# import pdb; pdb.set_trace()
hetus_mapped_data = pd.read_csv(MAPPING_FILE)


# Pre-process hetus_mapped_data for faster lookups
activity_power_map = hetus_mapped_data.set_index("activity")["Appliance_Power_Watts"].to_dict()

# Prepare MR lookup maps for each age group
mr_maps = {
    "MR_infant_Watts": hetus_mapped_data.set_index("activity")["MR_infant_Watts"].to_dict(),
    "MR_child_Watts": hetus_mapped_data.set_index("activity")["MR_child_Watts"].to_dict(),
    "MR_teenager_Watts": hetus_mapped_data.set_index("activity")["MR_teenager_Watts"].to_dict(),
    "MR_adult_Watts": hetus_mapped_data.set_index("activity")["MR_adult_Watts"].to_dict(),
    "MR_senior_Watts": hetus_mapped_data.set_index("activity")["MR_senior_Watts"].to_dict()
}

# Function to get metabolic rate column based on INC2
def get_mr_column(inc2_value):
    if inc2_value == 1: # 10-17 years
        return "MR_teenager_Watts"
    elif 2 <= inc2_value <= 11: # 18-64 years
        return "MR_adult_Watts"
    elif inc2_value >= 12: # 65+ years
        return "MR_senior_Watts"
    else:
        return "MR_adult_Watts" # Default

def process_chunk(chunk):
    # Create a copy to avoid the warning
    chunk = chunk.copy()
    
    # Calculate metrics
    mact_cols = [f"Mact{i}" for i in range(1, 145)]
    sactn_cols = [f"Sactn{i}" for i in range(1, 145)]
    wherep_cols = [f"Wherep{i}" for i in range(1, 145)]
    
    energy_data = []
    metabolic_data = []
    
    for index, row in chunk.iterrows():
        energy_series = []
        metabolic_series = []
        mr_col_name = get_mr_column(row["INC2"])
        current_mr_map = mr_maps[mr_col_name]
        
        for i in range(144):
            primary_activity_code = row[mact_cols[i]]
            secondary_activity_code = row[sactn_cols[i]] if sactn_cols[i] in row.index else np.nan
            is_at_home = row[wherep_cols[i]] == 1
            
            current_energy = 0
            current_mr = 0
            
            if is_at_home:
                if pd.notna(primary_activity_code) and primary_activity_code in activity_power_map:
                    current_energy += activity_power_map[primary_activity_code]
                    current_mr = current_mr_map.get(primary_activity_code, 0)
                
                # A code recorded as both main and secondary activity counts
                # once, as in the household rule of Stage 03 (audit of November
                # 2026, L10: it used to count twice in this per-diarist series)
                if (pd.notna(secondary_activity_code) and secondary_activity_code in activity_power_map
                        and secondary_activity_code != primary_activity_code):
                    current_energy += activity_power_map[secondary_activity_code]
                    secondary_mr = current_mr_map.get(secondary_activity_code, 0)
                    current_mr = max(current_mr, secondary_mr)
            
            energy_series.append(current_energy)
            metabolic_series.append(current_mr)
        
        energy_data.append(energy_series)
        metabolic_data.append(metabolic_series)
    
    # Use .loc to avoid the warning
    chunk.loc[:, "Diarist_Energy_Demand_Series"] = energy_data
    chunk.loc[:, "Diarist_Metabolic_Rate_Series"] = metabolic_data
    
    return chunk

def countries_to_load():
    """The countries of the run: all of COUNTRIES, or the comma-separated
    subset in HETUS_COUNTRIES (for instance to run without a country whose
    file is not available)."""
    selected = os.environ.get("HETUS_COUNTRIES", "")
    if not selected.strip():
        return list(COUNTRIES)
    names = [c.strip().upper() for c in selected.split(",") if c.strip()]
    unknown = [c for c in names if c not in CODE]
    if unknown:
        raise ValueError(f"HETUS_COUNTRIES: unknown country code(s) {unknown}; "
                         f"known codes are {COUNTRIES}")
    return [c for c in COUNTRIES if c in names]


def process_data_in_chunks():
    countries = countries_to_load()
    # A missing file stops the run rather than being skipped, so that a run
    # cannot silently leave a country out (audit of November 2026, L1)
    missing = [c for c in countries if not (RAW_DIR / f"{c}.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"raw file(s) not found in {RAW_DIR}: {', '.join(c + '.csv' for c in missing)}. "
            "Add them, or list the countries to use in HETUS_COUNTRIES.")
    
    all_chunks = []
    
    # First: Load all data without filtering
    for country_code in countries:
        # Raw files are read from RAW_DIR (paths.py): data/raw/ by default, or
        # the folder given by the HETUS_DATA_DIR environment variable.
        file_path = RAW_DIR / f"{country_code}.csv"
        print(f"Loading data for {country_code}...")
        chunk = pd.read_csv(file_path)
        chunk = harmonise_column_names(chunk, country_code)

        # Make HID and PID unique by adding country prefix. Whole-column
        # assignment (not .loc) lets the dtype change from int to string,
        # which recent pandas versions require.
        chunk['HID'] = country_code + '_' + chunk['HID'].astype(str)
        chunk['PID'] = country_code + '_' + chunk['PID'].astype(str)
        # The country is the one of the file (its code is set below)
        chunk['COUNTRY'] = country_code
        
        all_chunks.append(chunk)
        print(f"Finished loading {country_code}.")
    
    # Combine all data first
    combined_df = pd.concat(all_chunks, ignore_index=True)
    report_variable_coverage(combined_df)

    # Apply transformations to the full dataset
    combined_df = combined_df.copy()
    # Whole-column assignment so that the dtype can change from string to int
    # (required by pandas 3). The code of a country is its position in
    # countries.COUNTRIES, whichever files are read (audit of November 2026,
    # L1: versions up to 1.0.1 used pd.factorize, so codes followed the order
    # of the files actually loaded, while Stage 06 named them from the full list).
    combined_df['COUNTRY'] = combined_df['COUNTRY'].map(CODE).astype(int)
    combined_df.loc[combined_df.DDV3.isin([2,3]), 'DDV3'] = 1
    combined_df.loc[combined_df.DDV3.isin([5,6]), 'DDV3'] = 4
    combined_df.loc[combined_df.DDV3.isin([8,9]), 'DDV3'] = 7
    combined_df.loc[combined_df.DDV3.isin([11,12]), 'DDV3'] = 10
    
    # Negative codes are missing answers, except -1, which the scientific use
    # files use for "not applicable" (job items for people without a job,
    # education and main activity for diarists under 15). It is kept so that
    # Stage 07 counts item non-response among the units an item applies to
    # (audit of November 2026, D5); Stage 04 recodes it to -98.
    for col in SOCIO_LABELS.keys():
        if col in combined_df.columns and pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df.loc[(combined_df[col] < 0) & (combined_df[col] != -1), col] = np.nan
    
    print(f"Combined data from {len(countries)} countries. Total rows: {len(combined_df)}")
    
    # Filter at-home activities on the full dataset
    wherep_cols = [col for col in combined_df.columns if col.startswith("Wherep")]
    at_home_mask = combined_df[wherep_cols].eq(1).any(axis=1)
    filtered_df = combined_df[at_home_mask].copy()
    
    print(f"Number of records after filtering for at-home activities: {len(filtered_df)}")
    
    # Now process the filtered data in chunks
    chunk_size = 50000
    all_processed_chunks = []
    
    for i in range(0, len(filtered_df), chunk_size):
        chunk = filtered_df.iloc[i:i + chunk_size].copy()
        processed_chunk = process_chunk(chunk)
        all_processed_chunks.append(processed_chunk)
        
        # Free memory
        del chunk
        gc.collect()
    
    return pd.concat(all_processed_chunks, ignore_index=True)

# Main execution
print("Starting processing...")
lu_data_at_home = process_data_in_chunks()

if not lu_data_at_home.empty:
    print(f"Processed data with {len(lu_data_at_home)} rows")
    
    # Define columns to keep. DDV7 (kind of day) is carried through for the
    # day-type classification in Stage 06; it is not a clustering feature.
    columns_to_keep = [
        "YEAR", "HID", "PID", "DIARY", "COUNTRY", "DDV1", "DDV3", "DDV6", "DDV7", "INC1", "INC2", "INC3",
        "INC4_1", "IND6_1", "HHQ5", "HHQ9_1", "IND22_1", "IND44", 
        "HHC1", "HHC3", "HHC4", "HHC5", 
        "Diarist_Energy_Demand_Series", "Diarist_Metabolic_Rate_Series"
    ]
    
    for i in range(1, 145):
        columns_to_keep.append(f"Mact{i}")
        columns_to_keep.append(f"Sactn{i}")
        columns_to_keep.append(f"Wherep{i}")
        columns_to_keep.append(f"Wpartner{i}")
        columns_to_keep.append(f"Wchild{i}")
        columns_to_keep.append(f"Wotherh{i}")
    
    # Save the results
    lu_data_at_home[columns_to_keep].to_csv(PROCESSED_DIR / "lu_data_at_home_with_diarist_metrics.csv", index=False)
    print("Diarist-level metrics calculated and saved successfully!")
else:
    print("No data processed successfully.")
import pandas as pd
import numpy as np
import gc
import os
from paths import RAW_DIR, PROCESSED_DIR, MAPPING_FILE

SOCIO_LABELS = {
        'COUNTRY': 'Country',
        'DDV1': 'Gender',
        'DDV3': 'Age group',
        'DDV6': 'Education level',
        'HHC1': 'Household size',
        'HHC3': 'Number of infants',
        'HHC4': 'Number of children',
        'HHC5': 'Number of adults',
        'HHQ5': 'Own/rent house',
        'HHQ9_1': 'Income',
        'INC1': 'Gender',
        'INC2': 'Age',
        'INC4_1': 'Economic status',
        'IND6_1': 'Employment status',
        'IND22_1': 'Education',
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
                
                if pd.notna(secondary_activity_code) and secondary_activity_code in activity_power_map:
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

def process_data_in_chunks():
    countries = ['BE', 'DE', 'EE', 'EL', 'ES', 'FI', 'FR', 'HU', 'IT', 'LU', 'NL', 'NO', 'PL', 'RO', 'RS', 'UK']
    country_codes_map = {country: idx for idx, country in enumerate(countries, 1)}
    
    all_chunks = []
    
    # First: Load all data without filtering
    for country_code in countries:
        # Raw files are read from RAW_DIR (paths.py): data/raw/ by default, or
        # the folder given by the HETUS_DATA_DIR environment variable.
        file_path = RAW_DIR / f"{country_code}.csv"
        try:
            print(f"Loading data for {country_code}...")
            chunk = pd.read_csv(file_path)
            chunk = harmonise_column_names(chunk, country_code)

            # Make HID and PID unique by adding country prefix. Whole-column
            # assignment (not .loc) lets the dtype change from int to string,
            # which recent pandas versions require.
            chunk['HID'] = country_code + '_' + chunk['HID'].astype(str)
            chunk['PID'] = country_code + '_' + chunk['PID'].astype(str)
            
            all_chunks.append(chunk)
            print(f"Finished loading {country_code}.")
        except FileNotFoundError:
            print(f"Error: File not found for {country_code}. Skipping...")
        except Exception as e:
            print(f"An error occurred while loading {country_code}: {e}")
    
    if not all_chunks:
        return pd.DataFrame()
    
    # Combine all data first
    combined_df = pd.concat(all_chunks, ignore_index=True)
    report_variable_coverage(combined_df)

    # Apply transformations to the full dataset
    combined_df = combined_df.copy()
    # Whole-column assignment so that the dtype can change from string to int
    # (required by pandas 3). Codes follow the load order of `countries`.
    combined_df['COUNTRY'] = pd.factorize(combined_df['COUNTRY'])[0]
    combined_df.loc[combined_df.DDV3.isin([2,3]), 'DDV3'] = 1
    combined_df.loc[combined_df.DDV3.isin([5,6]), 'DDV3'] = 4
    combined_df.loc[combined_df.DDV3.isin([8,9]), 'DDV3'] = 7
    combined_df.loc[combined_df.DDV3.isin([11,12]), 'DDV3'] = 10
    
    for col in SOCIO_LABELS.keys():
        if col in combined_df.columns and pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df.loc[combined_df[col] < 0, col] = np.nan
    
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
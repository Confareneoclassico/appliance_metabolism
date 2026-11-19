"""Reference for Stage 03: the per-household loop of versions up to 1.1.0.

Stage 03 (03_aggregate_household.py) computes the same quantities with sparse
products. This loop, which takes 10 to 15 hours on HETUS 2010, is kept only to
check that stage, for instance on one country: it reads the Stage 02 output
from the processed folder and writes its files to the subfolder
stage03_loop_reference/, where tools/compare_stage03.py compares them with
the files of Stage 03. run_pipeline.sh does not run it.

    python pipeline/aggregate_household_loop_reference.py
"""
import pandas as pd
import numpy as np
import re
import gc
from pathlib import Path
import os
from paths import PROCESSED_DIR as INPUT_DIR, MAPPING_FILE

# Inputs are read from the processed folder, outputs written to a subfolder
PROCESSED_DIR = INPUT_DIR / "stage03_loop_reference"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Starting aggregate_household_metrics.py (loop reference)")

# Load the appliance power of every activity code. The metabolic rates come
# with the diarist series of Stage 01, and the rates of co-present members
# are no longer added (see the metabolic calculation below).
hetus = pd.read_csv(MAPPING_FILE)
activity_power = hetus.set_index("activity")["Appliance_Power_Watts"].to_dict()
print("Loaded HETUS_mapped_data.")

# Energy and MR conversion: 10-min interval power (W) to energy (Wh) multiply by 10/60
interval_hours = 10.0/60.0

# Custom robust parsing function for list-like strings
def parse_list_string_custom(s):
    if pd.isna(s):
        return [np.nan] * 144
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        s = s.strip("[]")
        elements = s.split(", ")
        parsed_list = []
        for e in elements:
            e = e.strip()
            if e == "nan" or e == "np.nan" or e == "":
                parsed_list.append(np.nan)
            else:
                try:
                    parsed_list.append(float(e))
                except ValueError:
                    parsed_list.append(np.nan)
        return parsed_list
    return [np.nan] * 144

# Function to process a single household-day combination WITH DEDUPLICATION
def process_household_day(household_day_data):
    """Process data for a single household on a specific day - WITH DEDUPLICATION"""
    if len(household_day_data) == 0:
        return None, None
    
    hid = household_day_data["HID"].iloc[0]
    diary = household_day_data["DIARY"].iloc[0]
    
    # Parse series columns
    for col in ["Diarist_Energy_Demand_Series", "Diarist_Metabolic_Rate_Series", "Diarist_Dominant_Activity_Series"]:
        household_day_data[col] = household_day_data[col].apply(parse_list_string_custom)
    
    household_intervals = []
    person_metrics = []
    
    # Get household attributes for all intervals. Person-level variables
    # (DDV6, DDV7, INC*, IND*) are taken from the first-listed diarist. The
    # order of extra_cols must match initialize_output_files().
    extra_cols = ["DDV1","DDV3","DDV6","DDV7","COUNTRY","HHC1","HHC3","HHC4","HHC5","INC1","INC2","INC3","INC4_1","IND6_1","HHQ5","HHQ9_1","IND22_1","IND44"]
    household_attrs = household_day_data[extra_cols].iloc[0].to_dict()
    
    # Process each interval
    for interval in range(144):
        # Create interval-specific data for all diarists
        interval_rows = []
        for _, row in household_day_data.iterrows():
            interval_row = {
                "PID": row["PID"],
                "Wherep": row[f"Wherep{interval+1}"],
                "Mact": row[f"Mact{interval+1}"],
                "Sactn": row[f"Sactn{interval+1}"],
                "Wpartner": row[f"Wpartner{interval+1}"],
                "Wchild": row[f"Wchild{interval+1}"],
                "Wotherh": row[f"Wotherh{interval+1}"],
                "Diarist_Energy_W": row["Diarist_Energy_Demand_Series"][interval] 
                    if row["Diarist_Energy_Demand_Series"] and len(row["Diarist_Energy_Demand_Series"]) > interval else np.nan,
                "Diarist_Metabolic_W": row["Diarist_Metabolic_Rate_Series"][interval] 
                    if row["Diarist_Metabolic_Rate_Series"] and len(row["Diarist_Metabolic_Rate_Series"]) > interval else np.nan,
                "Diarist_Dominant_Activity": row["Diarist_Dominant_Activity_Series"][interval] 
                    if row["Diarist_Dominant_Activity_Series"] and len(row["Diarist_Dominant_Activity_Series"]) > interval else np.nan,
                "HHC3": row["HHC3"],
                "HHC4": row["HHC4"],
                # Store all original activities for energy calculation
                "all_activities": set()
            }
            
            # Collect all activities for this diarist at this interval
            if not pd.isna(interval_row["Mact"]):
                interval_row["all_activities"].add(int(interval_row["Mact"]))
            if not pd.isna(interval_row["Sactn"]):
                interval_row["all_activities"].add(int(interval_row["Sactn"]))
                
            interval_rows.append(interval_row)
        
        interval_df = pd.DataFrame(interval_rows)
        
        # Filter for at-home activities
        at_home_data = interval_df[interval_df["Wherep"] == 1].copy()
        
        household_energy = 0.0
        household_metabolic = 0.0
        has_any_at_home = len(at_home_data) > 0
        
        if has_any_at_home:
            # Distinct dominant activities of the diarists at home, kept as the
            # diagnostic column n_unique_activities_after_dedup (they no longer
            # enter the metabolic rate, audit of November 2026, D10)
            at_home_data["Diarist_Dominant_Activity_int"] = at_home_data["Diarist_Dominant_Activity"].fillna(-1).astype(int)
            deduplicated_data = at_home_data.drop_duplicates(
                subset=["Diarist_Dominant_Activity_int"], 
                keep="first"
            ).copy()
            
            # ========================================================
            # Energy calculation: Count UNIQUE activities from ALL at-home diarists
            # (Not just deduplicated ones, because different appliances might be used)
            # ========================================================
            all_activities = set()
            for _, row in at_home_data.iterrows():
                all_activities.update(row["all_activities"])
            
            household_energy = sum(activity_power.get(act, 0.0) for act in all_activities)
            
            # ========================================================
            # Metabolic calculation: every diarist at home counts with his or
            # her own metabolic rate. Versions up to 1.0.1 summed the rates of
            # the distinct dominant activities only, so three people asleep at
            # home counted as one, and then added, for each of those
            # activities, a rate for every kind of co-present household member,
            # who is often a diarist already counted (audit of November 2026,
            # D10). Household members who kept no diary are not counted, since
            # the co-presence flags cannot tell them apart from diarists.
            # ========================================================
            household_metabolic = at_home_data["Diarist_Metabolic_W"].sum(skipna=True)
        
        # Store household interval data
        interval_data = {
            "HID": hid,
            "DIARY": diary,
            "Interval": interval,
            "Household_Energy_W": household_energy,
            "Household_Metabolic_W": household_metabolic,
            "Has_At_Home_Activities": has_any_at_home,
            "n_diarists_at_home": len(at_home_data),
            "n_unique_activities_after_dedup": len(deduplicated_data) if has_any_at_home else 0,
            "n_total_activities": len(all_activities) if has_any_at_home else 0
        }
        interval_data.update(household_attrs)
        household_intervals.append(interval_data)
    
    # Person-level metrics - only for at-home activities
    # NOTE: These will still include ALL diarists' metabolic totals
    # You might want to adjust these too if you want person-level metrics
    # that reflect the deduplication
    for _, row in household_day_data.iterrows():
        # Calculate metabolic total only for at-home intervals
        metabolic_total = 0.0
        if isinstance(row["Diarist_Metabolic_Rate_Series"], list) and len(row["Diarist_Metabolic_Rate_Series"]) == 144:
            # Check each interval to see if it's at home
            for interval in range(144):
                wherep_col = f"Wherep{interval+1}"
                if wherep_col in row and row[wherep_col] == 1:
                    metabolic_total += row["Diarist_Metabolic_Rate_Series"][interval] * interval_hours
        
        person_metrics.append({
            "HID": hid,
            "PID": row["PID"],
            "DIARY": diary,
            "diarist_total_metabolic_Wh": metabolic_total,
            # Flag for diagnostics: this diarist's total includes all intervals
            # even those that might be deduplicated at household level
            "note": "Person-level total includes all at-home intervals"
        })
    
    return household_intervals, person_metrics

def initialize_output_files():
    """Initialize output CSV files with headers"""
    extra_cols = ["DDV1","DDV3","DDV6","DDV7","COUNTRY","HHC1","HHC3","HHC4","HHC5","INC1","INC2","INC3","INC4_1","IND6_1","HHQ5","HHQ9_1","IND22_1","IND44"]
    
    # Long format headers - ADDED NEW COLUMNS FOR DIAGNOSTICS
    long_headers = ["HID", "DIARY", "Interval", "Household_Energy_W", "Household_Metabolic_W", 
                   "Has_At_Home_Activities", "n_diarists_at_home", "n_unique_activities_after_dedup",
                   "n_total_activities"] + extra_cols
    
    # Wide format headers
    wide_headers = ["HID", "DIARY"] + extra_cols
    for metric in ["Household_Energy_W", "Household_Metabolic_W"]:
        for interval in range(144):
            wide_headers.append(f"{metric}_{interval}")
    
    # Person headers - ADDED NOTE COLUMN
    person_headers = ["HID", "PID", "DIARY", "diarist_total_metabolic_Wh", "note"]
    
    # Create files with headers
    with open(PROCESSED_DIR / "household_interval_timeseries_long.csv", 'w') as f:
        f.write(','.join(long_headers) + '\n')
    
    with open(PROCESSED_DIR / "household_interval_timeseries.csv", 'w') as f:
        f.write(','.join(wide_headers) + '\n')
    
    with open(PROCESSED_DIR / "person_daily_metabolic_totals.csv", 'w') as f:
        f.write(','.join(person_headers) + '\n')
    
    return extra_cols

def append_to_files(household_intervals, person_metrics, extra_cols):
    """Append batch data to output files"""
    if household_intervals:
        # Create DataFrame for this batch
        household_df = pd.DataFrame(household_intervals)
        
        # Append to long format file
        household_df.to_csv(PROCESSED_DIR / "household_interval_timeseries_long.csv", mode='a', header=False, index=False)
        
        # Create wide format for this batch
        available_extra_cols = [col for col in extra_cols if col in household_df.columns]
        unique_households = household_df[["HID", "DIARY"] + available_extra_cols].drop_duplicates()
        
        wide_batch_data = []
        for _, household_row in unique_households.iterrows():
            household_mask = (
                (household_df["HID"] == household_row["HID"]) & 
                (household_df["DIARY"] == household_row["DIARY"])
            )
            household_data = household_df[household_mask]
            
            wide_row = household_row.to_dict()
            
            # Add energy and metabolic values for each interval
            for metric in ["Household_Energy_W", "Household_Metabolic_W"]:
                for interval in range(144):
                    interval_mask = household_data["Interval"] == interval
                    if interval_mask.any():
                        value = household_data.loc[interval_mask, metric].values[0]
                    else:
                        value = 0.0
                    wide_row[f"{metric}_{interval}"] = value
            
            wide_batch_data.append(wide_row)
        
        # Append wide format to file
        if wide_batch_data:
            wide_df = pd.DataFrame(wide_batch_data)
            wide_df.to_csv(PROCESSED_DIR / "household_interval_timeseries.csv", mode='a', header=False, index=False)
            del wide_df
        
        del wide_batch_data, household_df
    
    if person_metrics:
        person_df = pd.DataFrame(person_metrics)
        person_df.to_csv(PROCESSED_DIR / "person_daily_metabolic_totals.csv", mode='a', header=False, index=False)
        del person_df
    
    gc.collect()

# Main processing
print("Loading data...")
lu = pd.read_csv(INPUT_DIR / "lu_data_at_home_with_dominant_activity.csv")
print(f"Loaded {len(lu)} rows")

# Check country distribution in the raw data
if "COUNTRY" in lu.columns:
    country_counts = lu["COUNTRY"].value_counts()
    print("\nCountry distribution in raw data:")
    print(country_counts)
else:
    print("COUNTRY column not found in the data. Checking HID for country codes...")
    # Extract country from HID if it follows the pattern COUNTRY_PID
    if "HID" in lu.columns:
        lu["COUNTRY"] = lu["HID"].apply(lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else "Unknown")
        country_counts = lu["COUNTRY"].value_counts()
        print("Country distribution based on HID:")
        print(country_counts)

# Get unique household-day combinations (multiple people can share same HID+DIARY)
household_days = lu[["HID", "DIARY"]].drop_duplicates()
print(f"Found {len(household_days)} unique household-day combinations")

# Initialize output files
extra_cols = initialize_output_files()
print("Initialized output files with headers")

# Process household-day combinations in small batches and write immediately
batch_size = 250  # Smaller batch size for memory safety
processed_countries = set()
missing_countries = set()

# Tracking variables for final summary
total_long_rows = 0
total_wide_rows = 0
total_person_rows = 0

for batch_start in range(0, len(household_days), batch_size):
    batch_end = min(batch_start + batch_size, len(household_days))
    batch_household_days = household_days.iloc[batch_start:batch_end]
    
    print(f"Processing batch {batch_start//batch_size + 1}/{(len(household_days) + batch_size - 1)//batch_size} "
          f"(households {batch_start}-{batch_end})")
    
    batch_household_intervals = []
    batch_person_metrics = []
    
    for idx, (_, household_day_row) in enumerate(batch_household_days.iterrows()):
        hid = household_day_row["HID"]
        diary = household_day_row["DIARY"]
        
        if (batch_start + idx) % 100 == 0 and (batch_start + idx) > 0:
            print(f"  Processing household-day {batch_start + idx + 1}/{len(household_days)}: {hid}-{diary}")
        
        # Get ALL people from this household for this diary day
        household_day_data = lu[(lu["HID"] == hid) & (lu["DIARY"] == diary)].copy()
        
        # Track country
        country = "Unknown"
        if "COUNTRY" in household_day_data.columns:
            country = household_day_data["COUNTRY"].iloc[0]
        elif isinstance(hid, str) and '_' in hid:
            country = hid.split('_')[0]
        
        processed_countries.add(country)
        
        household_intervals, person_metrics = process_household_day(household_day_data)
        
        if household_intervals:
            batch_household_intervals.extend(household_intervals)
        else:
            if country not in missing_countries:
                missing_countries.add(country)
                print(f"    WARNING: No household intervals for country {country}, HID: {hid}")
        
        if person_metrics:
            batch_person_metrics.extend(person_metrics)
        
        # Clean up memory
        del household_day_data
        if idx % 50 == 0:
            gc.collect()
    
    # Write batch to files immediately and clear from memory
    append_to_files(batch_household_intervals, batch_person_metrics, extra_cols)
    
    # Update totals
    total_long_rows += len(batch_household_intervals)
    if batch_household_intervals:
        # Count unique households in this batch for wide format
        batch_df = pd.DataFrame(batch_household_intervals)
        unique_in_batch = len(batch_df[["HID", "DIARY"]].drop_duplicates())
        total_wide_rows += unique_in_batch
        del batch_df
    
    total_person_rows += len(batch_person_metrics)
    
    print(f"  Completed batch. Wrote {len(batch_household_intervals)} interval rows, {len(batch_person_metrics)} person rows to files")
    
    # Clear batch data from memory
    del batch_household_intervals, batch_person_metrics
    gc.collect()

print(f"\nFinal countries processed: {sorted(processed_countries)}")
print(f"Countries with missing data: {sorted(missing_countries)}")

# Versions up to 1.0.1 printed here an "overall deduplication impact" of the
# metabolic rule, next to the 61.5% same-activity figure of an unpublished
# version of Stage 09 (audit of November 2026, L17). Stage 09 now gives the
# share of shared activities.

def daily_totals_from_long(path, extra_cols, chunk_size=10000):
    """Daily totals of every household-day from the long file, read in
    chunks. The sums and slot counts of each chunk are added up by
    household-day after reading, so a household-day whose 144 rows fall in two
    chunks is not split in two (audit of November 2026, D9: 10,000 is not a
    multiple of 144, and versions up to 1.0.1 grouped each chunk on its own)."""
    parts, keys = [], None
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        if keys is None:
            keys = ["HID", "DIARY"] + [col for col in extra_cols if col in chunk.columns]
        # Convert to energy (Wh) by multiplying by interval_hours
        chunk["Household_Energy_Wh"] = chunk["Household_Energy_W"] * interval_hours
        chunk["Household_Metabolic_Wh"] = chunk["Household_Metabolic_W"] * interval_hours
        # dropna=False keeps household-days with missing attributes (e.g. DDV7
        # not asked in a country); the default would silently drop them.
        parts.append(chunk.groupby(keys, dropna=False).agg(
            total_energy_Wh=("Household_Energy_Wh", "sum"),
            total_metabolic_Wh=("Household_Metabolic_Wh", "sum"),
            sum_diarists_at_home=("n_diarists_at_home", "sum"),
            sum_unique_activities=("n_unique_activities_after_dedup", "sum"),
            n_slots=("Interval", "size")))
    if not parts:
        return pd.DataFrame()
    totals = pd.concat(parts).groupby(level=list(range(len(keys))), dropna=False, sort=False).sum()
    totals["avg_diarists_at_home"] = totals["sum_diarists_at_home"] / totals["n_slots"]
    totals["avg_unique_activities"] = totals["sum_unique_activities"] / totals["n_slots"]
    incomplete = int((totals["n_slots"] != 144).sum())
    if incomplete:
        print(f"WARNING: {incomplete} household-days do not have 144 slots in the long file")
    return totals.drop(columns=["sum_diarists_at_home", "sum_unique_activities", "n_slots"]).reset_index()


# Create daily totals from the long format file
print("Creating daily totals from saved data...")
try:
    all_daily_totals = daily_totals_from_long(
        PROCESSED_DIR / "household_interval_timeseries_long.csv", extra_cols)
    daily_totals = [all_daily_totals] if len(all_daily_totals) else []
    
    # Combine all daily totals
    if daily_totals:
        all_daily_totals.to_csv(PROCESSED_DIR / "household_daily_totals.csv", index=False)
        print(f"Exported {PROCESSED_DIR}/household_daily_totals.csv with {len(all_daily_totals)} rows")
        
        # Calculate average metrics
        avg_energy = all_daily_totals["total_energy_Wh"].mean()
        avg_metabolic = all_daily_totals["total_metabolic_Wh"].mean()
        avg_diarists = all_daily_totals["avg_diarists_at_home"].mean()
        avg_unique_acts = all_daily_totals["avg_unique_activities"].mean()
        
        print(f"Average daily household energy: {avg_energy:.1f} Wh")
        print(f"Average daily household metabolic: {avg_metabolic:.1f} Wh")
        print(f"Average diarists at home: {avg_diarists:.2f}")
        print(f"Average unique activities after dedup: {avg_unique_acts:.2f}")
        
        del all_daily_totals
    
    del daily_totals
    gc.collect()
    
except Exception as e:
    print(f"Error creating daily totals: {e}")

print(f"\nProcessing complete!")
print(f"- Long format: {total_long_rows} rows in {PROCESSED_DIR}/household_interval_timeseries_long.csv")
print(f"- Wide format: {total_wide_rows} rows in {PROCESSED_DIR}/household_interval_timeseries.csv") 
print(f"- Person data: {total_person_rows} rows in {PROCESSED_DIR}/person_daily_metabolic_totals.csv")
print("Finished aggregate_household_metrics.py")
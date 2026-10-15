import pandas as pd
import numpy as np
from ast import literal_eval
from paths import PROCESSED_DIR, MAPPING_FILE

# Load intermediate diarist-level metrics
lu = pd.read_csv(PROCESSED_DIR / "lu_data_at_home_with_diarist_metrics.csv")
# Ensure series columns are parsed from strings to lists
for col in ["Diarist_Energy_Demand_Series", "Diarist_Metabolic_Rate_Series"]:
    lu[col] = lu[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

# Load mapping for power and MR per activity
hetus = pd.read_csv(MAPPING_FILE)
mr_maps = {
    "MR_infant_Watts": hetus.set_index("TUS_activity_code")["MR_infant_Watts"].to_dict(),
    "MR_child_Watts" : hetus.set_index("TUS_activity_code")["MR_child_Watts"].to_dict(),
    "MR_teenager_Watts": hetus.set_index("TUS_activity_code")["MR_teenager_Watts"].to_dict(),
    "MR_adult_Watts": hetus.set_index("TUS_activity_code")["MR_adult_Watts"].to_dict(),
    "MR_senior_Watts": hetus.set_index("TUS_activity_code")["MR_senior_Watts"].to_dict(),
}

# Helper to map INC2 to MR column (assume diarist adult unless specified otherwise)
def mr_col_from_inc2(inc2):
    if pd.isna(inc2):
        return "MR_adult_Watts"
    try:
        inc2 = int(inc2)
    except Exception:
        return "MR_adult_Watts"
    if inc2 == 1:
        return "MR_teenager_Watts"
    elif 2 <= inc2 <= 11:
        return "MR_adult_Watts"
    else:
        return "MR_senior_Watts"

# Columns per interval
mact_cols   = [f"Mact{i}"   for i in range(1,145)]
sactn_cols  = [f"Sactn{i}"  for i in range(1,145)]
wherep_cols = [f"Wherep{i}" for i in range(1,145)]

# Determine dominant activity per diarist per interval (activity with higher MR)
# We will compute for each row a list of dominant activity codes length 144

dominant_activity_series = []
for idx, row in lu.iterrows():
    dom = []
    mr_map = mr_maps[mr_col_from_inc2(row.get("INC2"))]
    for i in range(144):
        pa = row[mact_cols[i]] if mact_cols[i] in row else np.nan
        sa = row[sactn_cols[i]] if sactn_cols[i] in row else np.nan
        at_home = (row[wherep_cols[i]] == 1) if wherep_cols[i] in row else False
        if not at_home:
            dom.append(np.nan)
            continue
        pmr = mr_map.get(pa, 0) if pd.notna(pa) else 0
        smr = mr_map.get(sa, 0) if pd.notna(sa) else 0
        if pmr >= smr and pd.notna(pa):
            dom.append(pa)
        elif pd.notna(sa):
            dom.append(sa)
        else:
            dom.append(np.nan)
    dominant_activity_series.append(dom)
lu["Diarist_Dominant_Activity_Series"] = dominant_activity_series

lu.to_csv(PROCESSED_DIR / "lu_data_at_home_with_dominant_activity.csv", index=False)
print(f"Diarist dominant activity series calculated and saved to {PROCESSED_DIR}/lu_data_at_home_with_dominant_activity.csv")



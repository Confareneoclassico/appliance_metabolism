"""Where the pipeline reads its inputs and writes its outputs.

Every stage imports its locations from this module. The locations are absolute
and derived from the position of this file, so a stage gives the same result
whichever directory it is run from. With the scripts in <root>/pipeline/:

    <root>/data/raw/     raw HETUS country files (not part of the repository)
    <root>/data/HETUS_mapped_data.csv
                         lookup table from activity code to appliance power
                         and metabolic rate (part of the repository)
    <root>/processed/    every intermediate and output CSV, with one folder per
                         stratification level (Country_Analysis/,
                         DayType_Analysis/, Country_DayType_Analysis/)
    <root>/Figures/      figures for the article and diagnostic plots

Each location can be changed with an environment variable (HETUS_DATA_DIR,
HETUS_PROCESSED_DIR, HETUS_FIGURES_DIR, HETUS_MAPPING_FILE) or by editing the
defaults below.
The output folders are created on import, so no stage fails at the end of a
long run because a folder is missing.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = Path(os.environ.get("HETUS_DATA_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("HETUS_PROCESSED_DIR", ROOT / "processed"))
FIGURES_DIR = Path(os.environ.get("HETUS_FIGURES_DIR", ROOT / "Figures"))

# Lookup table from activity code to appliance power and metabolic rate. The
# copy in data/ is part of the repository; a copy next to the raw files,
# where versions up to 1.0.1 expected it, is used only when that one is
# missing.
_MAPPING_IN_REPO = ROOT / "data" / "HETUS_mapped_data.csv"
MAPPING_FILE = Path(os.environ.get(
    "HETUS_MAPPING_FILE",
    _MAPPING_IN_REPO if _MAPPING_IN_REPO.is_file() else RAW_DIR / "HETUS_mapped_data.csv"))

# Per-stratum outputs of Stage 06. Country files are named by the two-letter
# country code, e.g. PL_clustering_results.csv in COUNTRY_DIR and
# PL_working_days_clustering_results.csv in COUNTRY_DAYTYPE_DIR.
COUNTRY_DIR = PROCESSED_DIR / "Country_Analysis"
DAYTYPE_DIR = PROCESSED_DIR / "DayType_Analysis"
COUNTRY_DAYTYPE_DIR = PROCESSED_DIR / "Country_DayType_Analysis"

# Figures of Stage 11
DRIVER_FIGURES_DIR = FIGURES_DIR / "Driver_Distributions"

for _folder in (PROCESSED_DIR, COUNTRY_DIR, DAYTYPE_DIR, COUNTRY_DAYTYPE_DIR,
                FIGURES_DIR, DRIVER_FIGURES_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

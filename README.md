# HETUS household electricity profiling pipeline

Reproducible pipeline that derives household electricity demand profiles at
10-minute resolution from the Harmonised European Time-Use Survey (HETUS),
clusters them globally, by day type, by country and by country and day type,
and validates the clusters statistically. It underpins the manuscript
"Deriving Household Electricity Profiles from European Time-Use Surveys: A
Multi-Level Clustering Approach Linking Appliance Use, Activity Patterns, and
Sociodemographic Drivers". `docs/PIPELINE.md` gives the full data-flow
reference, and the manuscript's Data and Methods sections give the
methodology.

## Quick start

```bash
# 1. Place the raw HETUS country files and the appliance mapping table in
#    data/raw/ (not included here, see "Inputs" below)
mkdir -p data/raw
cp /path/to/HETUS/{BE,DE,EE,EL,ES,FI,FR,HU,IT,LU,NL,NO,PL,RO,RS,UK}.csv data/raw/
cp /path/to/HETUS_mapped_data.csv data/raw/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline in order
bash run_pipeline.sh

#    or, when only sociodemographic inputs or the day-type rule changed,
#    skip the slow diarist diagnostics (Stages 09 and 10)
SKIP_DIAGNOSTICS=1 bash run_pipeline.sh

#    or re-run only the clustering and everything after it, reusing
#    processed/prepared_clustering_data.csv (Stages 05 to 08 and 11), with
#    the optional stability diagnostics of the global solution
START_STAGE=5 SKIP_DIAGNOSTICS=1 HETUS_STABILITY=1 bash run_pipeline.sh
```

Optional settings (environment variables): `HETUS_CLUSTER_ON_SOCIO=1` adds the
one-hot background variables to the clustering features, as before October
2026 (for comparison only), `HETUS_STABILITY=1` and `HETUS_STABILITY_REPS`
control the stability diagnostics of Stage 05, and `HETUS_RUN_ALL_DATA_PASS=1`
restores the duplicate first pass of Stage 06. The export files follow
Eurostat's publication rules for HETUS (`pipeline/disclosure.py`): groups of
fewer than 20 household-days are listed without size or statistics and their
household-days are reported with the largest cluster of their solution, groups
of 20 to 49 are flagged and listed without their size, sizes are rounded
wherever a group of the same solution is withheld or flagged, and association
tests are withheld or flagged for item non-response above 50% or 20%.
`HETUS_MIN_N_EXPORT` (20), `HETUS_FLAG_N` (50) and
`HETUS_ROUND_N` (10) hold these thresholds, which should not be lowered for
files that are shared or published.

Every stage takes its file locations from `pipeline/paths.py`, which resolves
them from the repository root, so a stage can be run from any directory
(`python pipeline/05_clustering_analysis_global.py` works as well as running
it from inside `pipeline/`). Raw inputs are read from `data/raw/` (or from the
folder in the `HETUS_DATA_DIR` environment variable), all intermediate and
output CSVs are written to `processed/`, with one subfolder per stratification
level, and figures to `Figures/`. To use other folders, edit the three
defaults in `paths.py` or set `HETUS_DATA_DIR`, `HETUS_PROCESSED_DIR` and
`HETUS_FIGURES_DIR`. The scripts carry a numeric prefix that makes the
execution order explicit. Their internals are unchanged except for the
corrections listed in `docs/PIPELINE.md` Section 3.

## Repository layout

```
.
├── README.md                  This file
├── LICENSE                    MIT License (code)
├── LICENSES/CC-BY-4.0.txt     CC BY 4.0 (outputs/ and docs/)
├── CITATION.cff               Citation metadata
├── .zenodo.json               Metadata for the Zenodo archive
├── requirements.txt           Python dependencies
├── run_pipeline.sh            Runs Stages 01 to 11 in order
├── docs/
│   └── PIPELINE.md            Input/output contract, corrections, and the
│                              file behind each claim in the manuscript
├── pipeline/
│   ├── paths.py                          (all input and output locations)
│   ├── disclosure.py                     (Eurostat publication rules for the exports)
│   ├── 01_calculate_diarist_metrics.py   (aligns column names across countries)
│   ├── 02_calculate_dominant_activity.py
│   ├── 03_aggregate_household.py
│   ├── 04_prepare_clustering_data.py
│   ├── 05_clustering_analysis_global.py
│   ├── 06_clustering_country_day.py
│   ├── 07_statistical_validation.py      (adds Cramér's V at every level)
│   ├── 08_circular_stats_peak_time.py
│   ├── 09_same_diarists_coincidence.py   (excludes missing activity codes)
│   ├── 10_p_h_counting.py
│   ├── 11_visualize_cluster_drivers.py
│   └── plot_country_cluster_profiles.py  (country cluster figure, from aggregates only)
├── archive/
│   ├── clustering_analysis_v1_2025-11-26_SUPERSEDED_BY_05.py
│   └── statistical_validation_v1_UNPATCHED_no_cramers_v_SUPERSEDED_BY_07.py
├── outputs/                   Aggregate results quoted in the manuscript
│                              (run of 14 October 2026, no microdata,
│                              under Eurostat's publication rules)
├── data/raw/                  Raw inputs (not tracked)
├── processed/                 Intermediate and output CSVs (not tracked)
└── Figures/                   Figures (not tracked)
```

## Inputs not included in this repository

The raw HETUS 2010 country microdata and the lookup table from activity to
appliance power and metabolic rate (`HETUS_mapped_data.csv`) are Eurostat
Scientific Use Files under restricted access and are not included. The run
behind the manuscript used the release of November 2020. `docs/PIPELINE.md`
Section 0 lists the expected file names.

## Corrections and open items

`docs/PIPELINE.md` Section 3 describes each correction made to the scripts.
The October 2026 corrections (column names aligned across national files,
the kind-of-day variable `DDV7` used for the day type, daily energy expressed
in Wh rather than as a sum of W, times of day counted from the 04:00 start of
the HETUS diary day, and clustering on the power profiles only) are all in
the run of 14 October 2026, which the manuscript quotes. Its aggregate
results are in `outputs/`, under Eurostat's publication rules for HETUS
(`docs/PIPELINE.md` Section 3.13). Section 5 of the same file lists the output
behind every quantitative claim.

## License

The code is released under the MIT License (`LICENSE`). The aggregate results
in `outputs/` and the documentation in `docs/` are released under the
Creative Commons Attribution 4.0 International License
(`LICENSES/CC-BY-4.0.txt`). Neither license covers the HETUS microdata, which
remain subject to Eurostat's conditions of access.

## Citation

If you use this pipeline, please cite the accompanying manuscript (full
citation to be added on publication), the archived version of this
repository on Zenodo, and the HETUS data source: Eurostat,
Harmonised European Time Use Surveys (HETUS), 2010 wave, scientific use files,
release of November 2020.

The results in `outputs/` are based on data from Eurostat, Harmonised European
Time Use Surveys (HETUS), 2010 wave, scientific use files, release of November
2020. The responsibility for all conclusions drawn from the data lies entirely
with the authors.

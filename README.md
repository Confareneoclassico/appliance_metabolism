# HETUS household electricity profiling pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23412741.svg)](https://doi.org/10.5281/zenodo.23412741)

Reproducible pipeline that derives household electricity demand profiles at
10-minute resolution from the Harmonised European Time-Use Survey (HETUS),
clusters them globally, by day type, by country and by country and day type,
and validates the clusters statistically, with a global sensitivity analysis
of the modelling choices (`gsa/`). It underpins the manuscript
"Activity-Associated Appliance-Load Profiles of European Households from
Time-Use Surveys: Multi-Level Clustering, Drivers and Robustness" (first
submitted as "Deriving Household Electricity Profiles from European Time-Use
Surveys: A Multi-Level Clustering Approach Linking Appliance Use, Activity
Patterns, and Sociodemographic Drivers"). `docs/PIPELINE.md` gives the full
data-flow reference, `docs/GSA.md` the sensitivity analysis, and the
manuscript's Data and Methods sections the methodology.

## Quick start

```bash
# 1. Place the raw HETUS country files in data/raw/ (not included here, see
#    "Inputs" below); the appliance mapping table is in data/
mkdir -p data/raw
cp /path/to/HETUS/{BE,DE,EE,EL,ES,FI,FR,HU,IT,LU,NL,NO,PL,RO,RS,UK}.csv data/raw/

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

# 4. Optionally, run the tests on synthetic data (no microdata needed)
pip install pytest
python -m pytest tests
```

Optional settings (environment variables): `HETUS_CLUSTER_ON_SOCIO=1` adds the
one-hot background variables to the clustering features, as before October
2026 (for comparison only), `HETUS_STABILITY=1` and `HETUS_STABILITY_REPS`
control the stability diagnostics of Stage 05, and `HETUS_RUN_ALL_DATA_PASS=1`
restores the duplicate first pass of Stage 06, `HETUS_COUNTRIES` (for instance
`BE,EE,FR`) restricts a run to some of the countries, `HETUS_DRIVER_TOP_N`
(6) sets how many variables Stage 11 draws, and `HETUS_SAME_DAY_RULE=0` keeps
the household-days whose members kept their diaries on different days, which
Stage 01 otherwise leaves out (`docs/PIPELINE.md` Section 3.17). The export files follow Eurostat's
publication rules for HETUS (`pipeline/disclosure.py`): every cluster of fewer
than 50 household-days is reported together with the largest cluster of its
solution in every file and figure, so that every published size is exact and
none below 50 can be derived from the others; every table of group means is
tested before it is written for means that reveal a smaller group; Stage 06
stops if a country and day type has between 1 and 49 household-days; and
association tests are withheld for fewer than 50 observations or item
non-response above 50%, and flagged for non-response above 20%.
`HETUS_MIN_N_EXPORT` (20), `HETUS_FLAG_N` (50) and `HETUS_ROUND_N` (10) hold
the thresholds, which should not be lowered for files that are shared or
published.

To share the results of a run, `python3 tools/collect_exports.py --log
run_log.txt` copies the aggregate exports and the main figures into
`exports_for_sharing/`, after testing every table with the publication rules,
and never copies a working file (for instance `clustering_results.csv`, one
row per household-day). Its `manifest.txt` lists what was copied, refused or
missing, and `log_extract.txt` keeps the lines of the saved console log with
solution-level figures only (PCA, WCSS, the choice of K).

The global sensitivity analysis of the revision (21 modelling choices, 12
outputs, 4,096 runs) is in `gsa/` and described in `docs/GSA.md`. It builds
its inputs from a canonical run on the machine that holds the microdata,
checks that its model reproduces the pipeline at the values of that run, and
collects only aggregate outputs for sharing:

```bash
python3 -m gsa.bundle --out ~/hetus_gsa/bundle          # inputs (microdata, stay here)
python3 -m gsa.check --bundle ~/hetus_gsa/bundle        # model against the run
python3 -m gsa.run --bundle ~/hetus_gsa/bundle --runs ~/hetus_gsa/runs --m 12 --workers 4 --threads 2
python3 -m gsa.collect --runs ~/hetus_gsa/runs          # files to share
```

The checks on the microdata that the sensitivity analysis cannot make (the
same households on both day types, the drivers with households resampled, the
stability of the stratified solutions and other clustering methods) run in one
command, `python3 tools/reviewer_checks.py --bundle ~/hetus_gsa/bundle`
(`docs/PIPELINE.md` Section 3.19).

Every stage takes its file locations from `pipeline/paths.py`, which resolves
them from the repository root, so a stage can be run from any directory
(`python pipeline/05_clustering_analysis_global.py` works as well as running
it from inside `pipeline/`). Raw inputs are read from `data/raw/` (or from the
folder in the `HETUS_DATA_DIR` environment variable), all intermediate and
output CSVs are written to `processed/`, with one subfolder per stratification
level, and figures to `Figures/`. To use other folders, edit the three
defaults in `paths.py` or set `HETUS_DATA_DIR`, `HETUS_PROCESSED_DIR` and
`HETUS_FIGURES_DIR`; a relative value is taken from the repository root. The
scripts carry a numeric prefix that makes the execution order explicit. Their
internals are unchanged except for the corrections listed in
`docs/PIPELINE.md` Section 3.

## Repository layout

```
.
├── README.md                  This file
├── LICENSE                    MIT License (code)
├── LICENSES/CC-BY-4.0.txt     CC BY 4.0 (outputs/ and docs/)
├── CITATION.cff               Citation metadata
├── CHANGELOG.md               Changes between archived versions
├── .zenodo.json               Metadata for the Zenodo archive
├── requirements.txt           Python dependencies
├── run_pipeline.sh            Runs Stages 01 to 11 in order
├── docs/
│   ├── PIPELINE.md            Input/output contract, corrections, and the
│   │                          file behind each claim in the manuscript
│   └── GSA.md                 The global sensitivity analysis
├── pipeline/
│   ├── paths.py                          (all input and output locations)
│   ├── disclosure.py                     (Eurostat publication rules for the exports)
│   ├── countries.py                      (country codes shared by Stages 01 and 06)
│   ├── peaks.py                          (daily peak times and their circular statistics)
│   ├── 01_calculate_diarist_metrics.py   (aligns column names across countries)
│   ├── 02_calculate_dominant_activity.py
│   ├── 03_aggregate_household.py         (household power as a sparse product, minutes)
│   ├── 04_prepare_clustering_data.py
│   ├── 05_clustering_analysis_global.py
│   ├── 06_clustering_country_day.py
│   ├── 07_statistical_validation.py      (adds Cramér's V at every level)
│   ├── 08_circular_stats_peak_time.py
│   ├── 09_same_diarists_coincidence.py   (activities shared at home by the diarists)
│   ├── 10_p_h_counting.py
│   ├── 11_visualize_cluster_drivers.py
│   ├── aggregate_household_loop_reference.py  (the former loop of Stage 03, for checks only)
│   └── plot_country_cluster_profiles.py  (country cluster figure, from aggregates only)
├── gsa/
│   ├── design.py              Factors, outputs and the Sobol' sample
│   ├── bundle.py              Inputs of the model, from a run (microdata)
│   ├── model.py               One run: power, clusterings, tests, outputs
│   ├── sun.py                 Darkness at the capitals (lighting factor)
│   ├── benchmarks.py          Measured profiles of the validation (Y9 to Y11)
│   ├── check.py               The model against the run it comes from
│   ├── run.py                 The runs of the sample (parallel, resumable)
│   ├── collect.py             The outputs that may be shared
│   ├── reseed.py              Runs again with the seeds redrawn
│   ├── estimators.py          Sobol', Shapley and optimal-transport indices
│   ├── convergence.py         Bootstrap, dummies and convergence
│   ├── analyse.py             The analysis of the shared outputs (no microdata)
│   ├── validation.py          Agreement with measured profiles, bands by country
│   ├── figures.py, tables.py  Figures and tables of the paper and its SI
├── tests/                     Tests on synthetic data (python -m pytest tests)
├── tools/collect_exports.py   Copies the exports that may be shared
├── tools/reviewer_checks.py   Runs the checks on the microdata in one command
│                              (paired_day_types.py, driver_robustness.py,
│                              clustering_robustness.py, run_folder.py)
├── tools/pca_summary.py       Principal components of the global solution
├── tools/compare_stage03.py   Compares Stage 03 with a reference run, value by value
├── archive/
│   ├── clustering_analysis_v1_2025-11-26_SUPERSEDED_BY_05.py
│   └── statistical_validation_v1_UNPATCHED_no_cramers_v_SUPERSEDED_BY_07.py
├── outputs/                   Aggregate results quoted in the revised
│                              manuscript (canonical run of 20 November 2026,
│                              checks, sensitivity analysis; no microdata,
│                              under Eurostat's publication rules; see
│                              outputs/README.md)
├── data/HETUS_mapped_data.csv Lookup table from activity to appliance power
│                              and metabolic rate
├── data/raw/                  Raw inputs (not tracked)
├── processed/                 Intermediate and output CSVs (not tracked)
└── Figures/                   Figures (not tracked)
```

## Inputs not included in this repository

The raw HETUS 2010 country microdata are Eurostat Scientific Use Files under
restricted access and are not included. The run behind the manuscript used the
release of November 2020. The lookup table from activity to appliance power and
metabolic rate, `data/HETUS_mapped_data.csv`, is the author's own and is
included (`docs/PIPELINE.md` Section 0 describes its columns). `docs/PIPELINE.md`
Section 0 lists the expected file names.

## Corrections and open items

`docs/PIPELINE.md` Section 3 describes each correction made to the scripts.
The October 2026 corrections (column names aligned across national files,
the kind-of-day variable `DDV7` used for the day type, daily energy expressed
in Wh rather than as a sum of W, times of day counted from the 04:00 start of
the HETUS diary day, and clustering on the power profiles only) are all in
the run of 14 October 2026, which the first submission of the manuscript
quotes, and in every later run. Section 5 of the same file lists the output
behind every quantitative claim of the revised manuscript.

Version 1.0.1 withdrew `outputs/stratified_cluster_mean_profiles.csv`, two
columns of `outputs/country_daytype_combined_summary.csv` and the stratified
rows of `outputs/consolidated_circular_stats.csv` of that run, from which the
sizes of small clusters could be derived (`CHANGELOG.md`, `docs/PIPELINE.md`
Section 3.14).
The issues found by the audit of November 2026 are corrected in version
1.1.0 (`docs/PIPELINE.md` Section 3.15, `CHANGELOG.md`), each with a test in
`tests/`. From that version, `outputs/` holds the results of the corrected
pipeline on the microdata (the canonical run of 20 November 2026), the checks
for the revision and the outputs of the sensitivity analysis, which the
revised manuscript quotes; the results of the run of 14 October 2026 remain in
versions 1.0.0 and 1.0.1 of the archive.

## License

The code is released under the MIT License (`LICENSE`). The aggregate results
in `outputs/`, the mapping table `data/HETUS_mapped_data.csv` and the
documentation in `docs/` are released under the
Creative Commons Attribution 4.0 International License
(`LICENSES/CC-BY-4.0.txt`). Neither license covers the HETUS microdata, which
remain subject to Eurostat's conditions of access.

## Citation

If you use this pipeline, please cite the accompanying manuscript (full
citation to be added on publication), the archived version of this
repository on Zenodo (all versions: https://doi.org/10.5281/zenodo.23412741;
version 1.0.0: https://doi.org/10.5281/zenodo.23412742; version 1.0.1:
https://doi.org/10.5281/zenodo.24368824; version 1.1.0: the DOI of its record
under the same concept DOI),
and the HETUS data source: Eurostat,
Harmonised European Time Use Surveys (HETUS), 2010 wave, scientific use files,
release of November 2020.

The results in `outputs/` are based on data from Eurostat, Harmonised European
Time Use Surveys (HETUS), 2010 wave, scientific use files, release of November
2020. The responsibility for all conclusions drawn from the data lies entirely
with the authors.

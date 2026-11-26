# HETUS electricity profiling pipeline: stage-by-stage reference

This document maps how data flow through the eleven pipeline stages. It was
reconstructed from the scripts' own `pd.read_csv` and `to_csv` calls, and it
records every correction made to the scripts while the manuscript was being
prepared, together with the pipeline output that supports each quantitative
claim in the manuscript.

## 0. Inputs not included in this repository

The raw HETUS country files (`{COUNTRY_CODE}.csv`, one per country: BE, DE,
EE, EL, ES, FI, FR, HU, IT, LU, NL, NO, PL, RO, RS, UK) are not included. They
are restricted-access Eurostat Scientific Use Files and must be obtained
separately and placed in `data/raw/` (or in the folder given by
`HETUS_DATA_DIR`) before Stage 01 is run. The run of 14 October 2026 used the
release of November 2020. From version 1.1.0 Stage 01 stops if one of the files
is missing; `HETUS_COUNTRIES` (for instance `BE,EE,FR`) restricts a run to
some of the countries, which keep the codes of the full list
(`pipeline/countries.py`).

The lookup table from activity code to appliance power and to metabolic rate,
`data/HETUS_mapped_data.csv`, is part of the repository (until version 1.0.1 it
was expected next to the raw files, and a copy there is still used when the
one in `data/` is missing). It has one row per HETUS activity code: `activity`
and `activity_name` are the 108 codes and headings of the activity coding list
of the 2008 guidelines (ACL 2008, Annex V), the codes used in `Mact` and
`Sactn`. `TUS_activity_code` and `TUS_activity_label` give a corresponding code
of a finer four-digit list (for instance 3611, shopping mainly for food, for
361), and `Mapping` the author's code for how the two correspond (in 106 of the
108 rows, 0 marks a one-to-one match, 1 one finer code chosen among several,
and 2 a finer code under another three-digit number, as 4240 for 421, a
reading still to be confirmed). The table also gives the activity of the
Compendium of Physical Activities behind the metabolic rate (`MET_info`), the
representative appliance power in W (`Appliance_Power_Watts`), the metabolic
rate in W for infants, children, teenagers, adults and seniors
(`MR_infant_Watts` to `MR_senior_Watts`), and flags for activities that need
light (`Light_use`) or hot water (`Water_use`), which no stage of the
pipeline reads; the sensitivity analysis reads `Light_use` for its lighting
factor X19 (`docs/GSA.md`), with 342 and 343 corrected (Section 3.17, K7). Its 108 codes have whole-watt powers, 28 of them above zero (30 up to
version 1.1.0 before the check of the codes, Section 3.17).

## 1. Input/output contract

All locations are defined once in `pipeline/paths.py` and resolved from the
repository root, so the stages do not depend on the directory they are run
from. Raw inputs are read from `data/raw/`, every CSV in the table below is
written to and read from `processed/`, the per-stratum outputs of Stage 06 go
to `processed/Country_Analysis/`, `processed/DayType_Analysis/` and
`processed/Country_DayType_Analysis/`, and figures to `Figures/` (Stage 06
keeps its per-stratum figures next to the corresponding CSVs). Country files
are named by the two-letter country code, for example
`PL_clustering_results.csv` and `PL_working_days_clustering_results.csv`.

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 01 | `calculate_diarist_metrics.py` | `{COUNTRY}.csv` (16 raw HETUS files), `HETUS_mapped_data.csv` | `lu_data_at_home_with_diarist_metrics.csv`, plus a console table of variable coverage by country |
| 02 | `calculate_dominant_activity.py` | Stage 01 output, `HETUS_mapped_data.csv` | `lu_data_at_home_with_dominant_activity.csv` |
| 03 | `aggregate_household.py` | Stage 02 output, `HETUS_mapped_data.csv` | `household_interval_timeseries.csv` (wide), `person_daily_metabolic_totals.csv`, `household_daily_totals.csv` (diagnostic only, not read by any later stage); `household_interval_timeseries_long.csv` (long) only with `HETUS_WRITE_LONG=1`, and `activity_matrix.npz` only with `HETUS_SAVE_ACTIVITY_MATRIX=1` (Section 3.16) |
| 04 | `prepare_clustering_data.py` | `household_interval_timeseries.csv` | `prepared_clustering_data.csv` (energy series, 16 clustering features, and `DDV7` as an auxiliary column) |
| 05 | `clustering_analysis_global.py` | `prepared_clustering_data.csv` | `clustering_results.csv`, `cluster_summary.csv`, `categorical_feature_statistics.csv` (superseded, see Section 2), elbow and PCA figures, `Figures/cluster_energy_profiles_10min_combined.png` (Figure 2 of the article) and `Figures/cluster_energy_profiles_10min_panels.png` (small-multiple version), console log |
| 06 | `clustering_country_day.py` | `prepared_clustering_data.csv` | `DayType_Analysis/`, `Country_Analysis/` (one set per country), `Country_DayType_Analysis/`, `Country_Analysis/country_comparison_summary.csv`, `Country_DayType_Analysis/country_daytype_combined_summary.csv`, `Country_DayType_Analysis/{working,non_working}_days_cross_country.csv`, figures |
| 07 | `statistical_validation.py` | `clustering_results.csv` and every per-stratum file from Stage 06 | `consolidated_anova_stats.csv`, `consolidated_chi2_stats.csv` (with Cramér's V) |
| 08 | `circular_stats_peak_time.py` | same file set as Stage 07 | `consolidated_circular_stats.csv` (circular mean peak time in clock time, circular standard deviation, and the number of household-days with no demand) |
| 09 | `same_diarists_coincidence.py` | `lu_data_at_home_with_dominant_activity.csv` | `same_activity_household_summary.csv` (working file, one row per household-day), `same_activity_by_diarists.csv`, `activity_combination_counts.csv`, `shared_activities_frequency.csv`, figure (Section 3.15) |
| 10 | `p_h_counting.py` | `person_daily_metabolic_totals.csv` | `diarists_per_household_summary.csv` (one row per household-day), figures |
| 11 | `visualize_cluster_drivers.py` | per-stratum files from Stage 06, `consolidated_chi2_stats.csv` | driver-distribution figures (no CSV) |

Stages 09 and 10 depend only on Stages 01 to 03 and can run in parallel with
Stages 04 to 08. They use activity codes and diarist counts only, so the
October 2026 corrections (Section 3) leave their outputs unchanged, and
`run_pipeline.sh` skips them when `SKIP_DIAGNOSTICS=1` is set. Stage 11
depends on Stage 06.

## 2. Redundancies and how they were resolved

Two versions of the global clustering script existed.
`archive/clustering_analysis_v1_2025-11-26...py` and Stage 05 both load
`prepared_clustering_data.csv`, run the same PCA, elbow and K-means sequence,
and write `clustering_results.csv` and `cluster_summary.csv`. Stage 05 does
everything the November version does and adds a Chi-squared and Cramér's V
step, so the November version is archived and should not be run. Running both
would risk two diverging copies of `clustering_results.csv`.

Two Chi-squared and Cramér's V implementations also exist. The one embedded
in Stage 05 (`categorical_feature_statistics.csv`) covers the global solution
only and keeps the missing-value code `-99` in its contingency tables. Stage
07 (`consolidated_chi2_stats.csv`) covers every stratification level and
excludes `-99`. The manuscript uses Stage 07 exclusively, including at the
global level. The Stage 05 block is retained only because it sits beside the
PCA and elbow diagnostics, and it could be deleted without affecting the
manuscript.

Stage 11 hard-coded `TOP_FEATURES = ["HHC5", "HHC1", "INC2", "DDV6"]` instead
of reading the strongest drivers from Stage 07. From version 1.1.0 it takes
them from `consolidated_chi2_stats.csv` (Section 3.15, L16).

## 3. Corrections made to the scripts

### 3.1 Stage 09: missing activity codes counted as shared activities

The original script treated missing and non-response activity codes as
ordinary values, so two diarists with no recorded activity in the same
interval were counted as sharing an activity. The patched script excludes
`{-99, -9, -8, -7, -6, -1}` from the comparison. It was re-run (author's
vectorised version) and the manuscript reports the corrected values: 61.5%
overall, and 56.0%, 82.5% and 97.9% for household-days with two, four and
seven diarists.

### 3.2 Stage 07: Cramér's V added to every stratum

The original script reported only the Chi-squared statistic and p-value below
the global level. The patched script adds Cramér's V, with the same formula
and effect-size bands as Stage 05, to every stratum. This is what allowed the
driver analysis to rest on effect sizes rather than on significance alone.

### 3.3 Stage 01: column names aligned across national files (October 2026)

In the previous run, Romania lost 12 of the 15 sociodemographic clustering variables (`DDV1`,
`DDV3`, `DDV6`, `INC1`, `INC4_1`, `IND6_1`, `IND22_1`, `IND44`, `HHC1`, `HHC3`,
`HHQ5`, `HHQ9_1`). Only `HHC4`, `HHC5` and `INC2` survived, and all Romanian
household-days were coded `-99` for the others. A direct read of the raw
Romanian file shows `DDV1` complete for codes 1 to 7 and `INC4_1` complete for
codes 1 to 8, so the loss happened in the pipeline. The national files are
read one at a time and concatenated, and `pd.concat` aligns columns by exact
name. A variable delivered under a differently cased name (for example `ddv1`
instead of `DDV1`) ends up in a separate column that is later dropped.

Stage 01 now renames every column that differs from a canonical name only in
case or surrounding whitespace (`harmonise_column_names`), prints the renamed
columns for each country, and prints a table of the percentage of records with
a valid code for each variable by country (`report_variable_coverage`). The
table is printed before the slow per-diarist step starts, so the run can be
stopped early if something is wrong. A value of 0 for Romania would mean the
mismatch is of another kind, and the header of `RO.csv` should then be compared
with that of `BE.csv`:

```python
import pandas as pd
ro = set(pd.read_csv("data/raw/RO.csv", nrows=0).columns)
be = set(pd.read_csv("data/raw/BE.csv", nrows=0).columns)
print(sorted(be - ro))
print(sorted(ro - be))
```

Because 29,243 Romanian household-days now enter the clustering with their
true attributes, every global, day-type and country-by-day-type result
changes, and so does Romania's country-level solution. The country-level
solutions of the other fourteen countries do not change unless the renaming
log reports renamed columns for them too.

### 3.4 Day type from `DDV7`, then `DDV1` (October 2026)

Stage 06 has always contained a rule that classifies a household-day as a
working day when `DDV7` (kind of day) is 1 (ordinary work day) or 2 (ordinary
school day) and as a non-working day when it is 3 to 6 (day off, sick leave,
vacation, other leave), falling back to the weekday (`DDV1`: Monday to Friday
working, Saturday and Sunday non-working) when `DDV7` is missing or
negative. `DDV7` was dropped in Stage 01, however, so every day was classified
by weekday. `DDV7` is now carried through Stages 01 to 04 as an auxiliary
column (it is not a clustering feature), and the rule runs as written.
`determine_day_type` also treats a missing `DDV7` as negative. `DDV7` has no
valid values for DE, ES, IT and LU, where the weekday rule still applies. The
SUF documentation also lists EL among the countries that did not ask it, but
the Stage 01 coverage table of the 11 October 2026 run shows valid values for
41% of Greek diary records, so the rule uses it for Greece. It is a
person-level variable, and the household value is that of the first-listed
diarist (see Section 4).

### 3.5 Daily energy units (October 2026)

The `Household_Energy_W_*` columns hold mean power in W for each 10-minute
interval. Several summaries in Stages 05 and 06 summed the 144 values and
labelled the sum Wh, which overstates daily energy six-fold. Daily energy is
now the sum multiplied by 10/60 h (`INTERVAL_HOURS` in Stage 06) wherever it
was computed this way. Stage 06's two plotting helpers that already divided by
6 were correct and are unchanged. In `cluster_summary.csv` the column
`avg_peak_energy_wh` is renamed `avg_peak_power_w`, because it holds the mean
of each household-day's highest 10-minute mean power, in W. Peak power values
were always correct.

### 3.6 Stage 03: household-days with missing attributes kept in the daily totals

`household_daily_totals.csv` grouped by all household attributes with the
pandas default `dropna=True`, which silently dropped every household-day with
a missing attribute. The grouping now uses `dropna=False`. No later stage
reads this file.

### 3.7 Figure 2 (October 2026)

The global cluster profiles are now drawn at a size meant for a full-width
journal figure, with larger text, lines only, a y-axis labelled as mean power
in W (it was labelled Wh), an x axis that runs over the diary day from 04:00
to 04:00 in clock time (Section 3.8), no title, and a legend below the plot. Colours
follow a fixed order checked for colour-vision deficiency, and a ninth or
later cluster reuses a colour with a dashed line. Stage 05 also writes a
small-multiple version with one cluster per panel and the others in grey.

### 3.8 Clock time of the 10-minute slots (October 2026)

HETUS diary days run from 04:00 to 03:59 on the next calendar day. The SUF
documentation defines `Mact001` as the activity from 04:00 to 04:09 and
`Mact144` as the activity from 03:50 to 03:59, and Stages 01 to 03 keep this
order, so slot 0 of every `Household_Energy_W_*` series is 04:00. Every
script nevertheless converted slot positions to the hour of day as slot/6,
treating slot 0 as midnight, so all times reported so far (peak times,
circular means, the time windows of the principal components, and the time
axes of every figure) were four hours early. The mean profiles confirm the
offset: read as clock time from midnight, they show almost no demand from
20:00 to 02:00 and a rise in demand at 02:00, whereas with slot 0 at 04:00 the
quiet period falls between midnight and 06:00 and the morning rise at 06:00.

Stages 05, 06 and 08 now convert slot positions to clock time as 04:00 plus
10 minutes per slot (modulo 24 h). Figures keep the diary order on the x axis
(04:00 to 04:00) and label it with clock time. The correction shifts every
reported time by +4 h and leaves cluster assignments, circular standard
deviations and all energy and power values unchanged. Stage 08 also reports
`n_zero_demand`, the number of household-days without any demand, because
these have no peak and are assigned slot 0 (04:00) by `idxmax`.

### 3.9 Smaller fixes (October 2026)

Stage 01 assigns the factorised `COUNTRY` codes to the whole
column, as it already did for `HID` and `PID`, which pandas 3 requires.
Stage 04 replaces missing values without the chained `inplace=True` call,
which has no effect under pandas copy-on-write.

The full chain (Stages 01 to 11) was run on a small synthetic dataset that
reproduces the Romanian header problem (lower-case names for 13 variables)
and the countries without `DDV7`. All stages completed, Romania's variables
were recovered, `DDV7` reached the day-type rule, daily energy matched the
Stage 03 total in Wh, and peak times were reported in clock time.

### 3.10 File locations and country file names (October 2026)

The scripts had been given relative paths written for different working
directories: the raw files were read from `./data/raw` in Stage 01 but from
`./raw` in Stages 02 and 03, figures went to `../Figures`, `./Figures`,
`../processed/Figures` or the working directory, and Stages 07 and 08 read the
global solution from a `Clustering_Results/` subfolder that no stage writes.
No single working directory satisfied all of them. All locations now come
from `pipeline/paths.py` (Section 1), which also creates the output folders
before any stage writes, so Stage 01 can no longer fail at the end of its long
run because `processed/` is missing.

Stage 06 names the country and country-by-day-type files by the two-letter
country code, whereas Stages 07, 08 and 11 looked for the older names
(`Country_12.0_clustering_results.csv` and similar). Their searches matched
nothing, so the consolidated statistics silently covered only the global and
day-type solutions. Stages 07, 08 and 11 now use the code-based names, with
the pattern `[A-Z][A-Z]_...` so that the duplicate `*_ALL_DATA_*` files
written by Stage 06's first pass are not counted twice. That first pass
clusters each country on exactly the same data as the second one and could
be removed to save time.

The chain was re-run on the synthetic dataset from a working directory
outside the repository (Stage 09 on a subset, since it is slow). Every output
appeared under `processed/` and `Figures/`, nothing was written to the working
directory, and the consolidated statistics included the 15 countries and every
combination of country and day type that met the size threshold, against only
the global and day-type solutions before the fix.

### 3.11 Figure S1 colours (October 2026)

In the country-by-day-type summary figure
(`country_daytype_combined_summary.png`), the energy and peak-power panels
overlaid the bars of the two day types, with non-working days in blue,
whereas the cluster-count panel showed working days in blue. Stage 06 now
draws side-by-side bars in all three panels, with working days in blue and
non-working days in orange throughout. The figure depends only on
`country_daytype_combined_summary.csv`, and the version in the 11 October
manuscript was redrawn from that file with the corrected function, so the
values are those of the run.

### 3.12 Clustering on the power profiles only, and new outputs (October 2026)

Stages 05 and 06 formed the clusters on the 144 standardised power slots
together with a one-hot encoding of the 16 background variables, and Stages
05 and 07 then tested the same variables against the clusters. Because the
variables had helped to form the clusters, those tests were partly circular.
Both stages now cluster on the power slots only. The background variables stay
in every results file and are tested exactly as before.
`HETUS_CLUSTER_ON_SOCIO=1` restores the joint feature space, for comparison
with the run of 11 October 2026.

Stage 06 kept `min(0.95, 144 / number of columns)` of the variance in its PCA,
about 59% for the two day types and 65 to 71% for the countries and the
combinations of country and day type, against 95% in the global solution. It
now keeps 95% at every level.

New outputs:

| File | Content |
|---|---|
| `cluster_summary.csv` | New columns `mean_profile_peak_w` and `mean_profile_peak_time`: maximum of each global cluster's mean profile and its clock time (previously read off Figure 2) |
| `cluster_mean_profiles.csv` | Mean profile of each global cluster (144 slots, W, slot 0 = 04:00) |
| `stratified_cluster_mean_profiles.csv` | The same for every cluster of every stratified solution, with size, share, daily energy and both peaks, under the publication rules of Section 3.13 (withdrawn from `outputs/` in version 1.0.1, Section 3.14) |
| `Country_DayType_Analysis/mean_profiles_country_daytype.csv` | Mean profile, daily energy, mean daily peak and coincident peak for each country and day type, each country over both day types, each day type over all countries, and the whole sample |
| `country_comparison_summary.csv`, `country_daytype_combined_summary.csv`, `peak_power_daytype_comparison.csv` | New coincident-peak columns |
| `Country_DayType_Analysis/country_daytype_profiles_small_multiples.png` | Working-day and non-working-day mean profiles, one panel per country, with the maximum of each profile marked |
| `cluster_stability.csv` (with `HETUS_STABILITY=1`) | Silhouette width on 10,000 household-days, and adjusted Rand index and cluster-wise Jaccard similarity over `HETUS_STABILITY_REPS` (default 10) half-samples |

The coincident peak is the maximum of a group's mean profile, the
after-diversity maximum demand per household that a feeder serving many such
households would see. It differs from the mean daily peak (`avg_peak_power_w`),
which averages each household-day's own maximum. Stage 06 prints both, for
working and non-working days, country by country.

Other changes: the duplicate first pass of Stage 06 (the `XX_ALL_DATA` files)
is skipped unless `HETUS_RUN_ALL_DATA_PASS=1`, figure titles no longer
describe the day types as weekdays and weekends, the day-type colours are the
same in every figure, and `run_pipeline.sh` accepts `START_STAGE=5` to re-run
from the clustering. The changed chain ran end to end on the synthetic dataset
with both feature settings and with the stability diagnostics.

### 3.13 Publication rules, the record of K and the country cluster figure (October 2026)

Publication rules. Eurostat's guidelines for publications based on its
scientific use files (Access to confidential data for scientific purposes,
Guidelines for publication, last updated 3 September 2026, section 7.7: HETUS
follows the EU-SILC rules of section 7.2) require that results based on fewer
than 20 observations, or on an item with more than 50% non-response, are not
published, that results based on 20 to 49 observations, or on an item with
more than 20% non-response, are flagged, that sample sizes in these two ranges
are given only as "less than 20" and "20 to 49", not as the actual number, and
that a sample size below 20 cannot be derived from other published figures,
such as row or column totals. Earlier exports used a threshold of 10 and kept
the sizes of the small groups. `pipeline/disclosure.py` now applies the rules
to every export file meant to leave the machine that holds the microdata.

The files `cluster_summary.csv`, `cluster_mean_profiles.csv` and
`cluster_stability.csv` (Stage 05), `stratified_cluster_mean_profiles.csv`
and `mean_profiles_country_daytype.csv` (Stage 06) and
`consolidated_circular_stats.csv` (Stage 08) pass through `protect`. A group
of fewer than 20 household-days keeps only its identifiers and
`n_class = "<20"`. A group of 20 to 49 keeps its results with
`n_class = "20-49"` but not its size or share. Wherever a group is withheld or
flagged, the sizes of the other groups of the same solution are rounded to the
nearest 10 and their shares recomputed from the rounded sizes, in whole
percentage points. Counts of household-days with some property (the
zero-demand days of Stage 08) are left empty between 1 and 49.

In `stratified_cluster_mean_profiles.csv` the household-days of withheld
clusters are reported together with the largest cluster of their solution,
whose row has `includes_withheld = True`. Since the mean profile of every
stratum is published in `mean_profiles_country_daytype.csv`, the profile of a
withheld cluster could otherwise be recovered, approximately even with rounded
sizes, by subtracting the published clusters. The circular statistics of
Stage 08 have no published stratum total and keep each cluster's own values.
Stage 05 warns if a global cluster falls below 20, which would need the same
treatment.

`country_daytype_combined_summary.csv` (Stage 06) gives the largest cluster of
a solution with a withheld or flagged cluster rounded to the nearest 10, with
its share computed from the rounded size, because its exact size would give
that of the small cluster.

`consolidated_chi2_stats.csv` (Stage 07) no longer has the smallest expected
count of each table, which with the published totals gave the size of the
smallest cluster (for the pooled working days, a withheld cluster) and of the
smallest category. The console log reports how many tables have an expected
count below five. Each test now has `Item_Nonresponse_Pct`, the share of
household-days without an answer among those of the countries that collected
the variable, and `Publication_Class`, which is empty, `flagged` or `withheld`
(no statistics). Job type (`IND6_1`) and contract type (`IND44`) apply only to
people with a job, and HETUS codes their missing and not-applicable answers
alike, so their non-response is not assessed. Tests on fewer than 20
observations are not run.

The stratified export also carries `stratum_n` and `size_rank`, so that
figures can be drawn without the exact sizes. In the run of 14 October 2026,
32 of the 234 clusters of the stratified solutions are withheld and 17 are
flagged, and three tests are withheld (household income in the three Serbian
strata, with about 80% non-response) and six flagged (household income and
educational attainment in the three British strata, 22 to 27%). The other
Stage 06 files (`XX_cluster_statistics.csv`, `XX_cluster_summary.csv`, the
day-type summaries) and the console log are working files that are not
protected and must not be shared.

Record of K. When `KneeLocator` finds no knee, Stage 06 uses `min(4, K_max)`
without saying so, so a stratum with `K = 4` could not be told apart from one
whose knee is at 4. Stage 06 now prints the WCSS curve of every stratum and
marks a fallback value in the log.

Country cluster figure. `pipeline/plot_country_cluster_profiles.py` draws the
mean profiles of the clusters of every country (Figure S2 of the manuscript)
from `stratified_cluster_mean_profiles.csv` alone, so it needs no microdata
(the published copy of that file was withdrawn in version 1.0.1, Section
3.14, and the figure is drawn from the copy in `processed/`).
`run_pipeline.sh` runs it after Stage 06, and it can be run on its own with
the path of an export file as argument. Clusters are coloured by their size
rank within the country, withheld clusters are not drawn (their
household-days are part of the largest cluster), and flagged ones are drawn
with dashed lines.

### 3.14 Sizes recoverable from published means, and version 1.0.1 (November 2026)

An audit in November 2026 found that the rules of Section 3.13 did not hold in
the files published with version 1.0.0. Slot powers are whole watts, so the
household power in every slot is an integer and the mean profile of a cluster
of n household-days is a vector of 144 integers divided by n. The smallest n
for which all 144 values times n are integers is therefore the size of the
cluster. In `stratified_cluster_mean_profiles.csv` this gives the sizes of all
17 clusters of 20 to 49 household-days and the exact sizes behind all 140
sizes rounded to the nearest 10, and the recovered sizes add up to the
published stratum size in all 47 solutions. The merged size of each largest
cluster, with the size of the largest cluster alone in
`country_daytype_combined_summary.csv`, bounds the household-days of the
withheld clusters of a solution to at most ten values. Rounding the published
profiles does not prevent this, because the stratum profiles and sizes of
`mean_profiles_country_daytype.csv` let the cluster sizes be estimated by
least squares from the 144 equations of each stratum: with the profiles
rounded to 1 W, 5 of the 17 sizes between 20 and 49 are still recovered
exactly and 15 to within five household-days, and with 10 W none exactly but
5 to 7 to within five. The rounded sizes of the stratified clusters in
`consolidated_circular_stats.csv`, with the exact stratum sizes, also bound
the withheld clusters more narrowly than the class "fewer than 20" (to at most
11 household-days for the withheld cluster of Poland on working days).

Version 1.0.1 therefore withdraws `stratified_cluster_mean_profiles.csv`, the
columns `largest_cluster_size` and `largest_cluster_pct` of
`country_daytype_combined_summary.csv`, and the rows of the stratified
solutions in `consolidated_circular_stats.csv`, which keeps the nine global
clusters quoted in the manuscript. No remaining file holds a group of fewer
than 50 household-days or a breakdown of a stratum into clusters. The code is
unchanged: Stage 06 still writes the stratified export to `processed/`, where
it is a working file that must not be shared. Version 1.1.0 will report every
cluster of fewer than 50 household-days together with the largest cluster of
its solution in every export, so that sizes can be published exactly, and
`disclosure.py` will run the two tests above on every file before it is
written.

### 3.15 Corrections from the audit of November 2026 (version 1.1.0)

The audit of November 2026 numbered the defects that affected the run of
14 October 2026 D1 to D10 and the latent ones, correct in that run but wrong
under other inputs or settings, L1 to L17. All are corrected below. Each
correction has a test on synthetic data in `tests/` (`python -m pytest tests`,
about 30 seconds). Stages 01 to 04, 09 and 10 also ran on a small synthetic
set of raw files, and Stages 05 to 08 and 11 on a synthetic prepared dataset
of 19,227 household-days (one tenth of the real country sizes, whole-watt
profiles) with a planted cluster of six household-days, and every export of
that run passed the tests of `disclosure.py`. None of the corrections has been
run on the microdata yet. The comparison of every output with version 1.0.0
comes next, and Section 5 will then give the corrected numbers.

| ID | Where | Correction | What changes in the outputs | Test |
|---|---|---|---|---|
| D1, D2 | `disclosure.py`, Stages 05, 06, 08, 11, country cluster figure | Every cluster of fewer than 50 household-days is reported together with the largest cluster of its solution in every export and figure (`published_groups`, Option A); exports with group means pass `check_export` before they are written | Small clusters no longer have rows of their own; the row of the largest cluster gives `n_small_clusters_merged`; all published sizes are exact | `test_disclosure.py` |
| D3 | Stage 09 | Rewritten: a slot is comparable when at least two distinct diarists are at home (`Wherep` = 1) with a valid code, and shared when at least two of them record the same code as main or secondary activity, each diarist counted once per code | Every share of Section 5 that comes from Stage 09 | `test_stage09_sharing.py` |
| D4 | `peaks.py`, Stages 05, 06, 08 | The weight of a household-day is shared equally among the slots at its daily maximum, and days without demand are left out | Circular means and SDs of peak times, `peak_time_circular_mean` | `test_peaks.py` |
| D5 | Stages 01, 04, 07, 11 | Not applicable (−1, recoded to −98) is kept apart from refusal, don't know and no answer (−99); non-response is counted among the household-days an item applies to, without the exemption of `IND6_1` and `IND44` | `Item_Nonresponse_Pct` and `Publication_Class` of Stage 07 | `test_stages01_04.py`, `test_stage07.py` |
| D6 | Stage 06 | Each export records how K was chosen (`k_selection`: knee or fallback), and the stratified export gives K | New columns only | end-to-end run |
| D7 | Stage 05 | `energy_std` (pooled SD of 10-minute power, W) is replaced by `daily_energy_sd_wh`, the SD of daily energy | `cluster_summary.csv` | end-to-end run |
| D8 | Stage 11 | No transpose: each bar is a cluster and adds up to 100% | Driver figures | `test_figures_and_drivers.py` |
| D9 | Stage 03 | Daily totals are added up by household-day after the long file is read in chunks | `household_daily_totals.csv` | `test_stages01_04.py` |
| D10 | Stage 03 | The household metabolic rate is the sum of the rates of the diarists at home; the co-presence additions are dropped | Metabolic columns only | `test_stages01_04.py` |
| L1 | `countries.py`, Stages 01, 06 | Country codes from the name of the raw file, and a missing file stops Stage 01 (`HETUS_COUNTRIES` selects a subset) | None in a run with all 16 files | `test_stages01_04.py` |
| L2 | Stage 07 | `chi2_contingency(..., correction=False)` | V of 2 x 2 tables (none published) | `test_stage07.py` |
| L3 | Stage 07 | `N_Observations` only from 50 observations, with `N_Class` otherwise; tests on fewer than 50 observations withheld | None published | `test_stage07.py` |
| L4 | Stage 06 | `mean_profiles_country_daytype.csv` is one set for rounding, its ALL rows included, and passes `check_export` | None while every group has 50 or more | `test_disclosure.py` |
| L5 | Stage 06 | Every threshold from `disclosure.py`; a country and day type of 1 to 49 household-days stops the stage before anything is written (`check_strata`); every summary writer checks its sizes (`assert_sizes`) | None in HETUS 2010 (smallest stratum 946) | `test_stage06_guard.py` |
| L6 | Stage 05 | Figure 2 draws the published groups with their exact sizes | None while every global cluster has 50 or more | end-to-end run |
| L7 | Stage 05 | The elbow plot labels a fallback K as such | Elbow figure | end-to-end run |
| L8 | Country cluster figure | Eight colours in fixed order, ranks from the ninth in grey, ordinals generated | A country with nine or more clusters no longer fails | `test_figures_and_drivers.py` |
| L9 | `paths.py`, `run_pipeline.sh` | A relative `HETUS_*` location is taken from the repository root; `run_pipeline.sh` reads all locations from `paths.py` | None with absolute settings | `test_paths.py` |
| L10 | Stage 01 | A code recorded as main and secondary activity counts once in the per-diarist series | `Diarist_Energy_Demand_Series` (not read later) | `test_stages01_04.py` |
| L11 | Stage 02 | Metabolic rates are looked up by `activity`, as in Stages 01 and 03 | Dominant activity, so the diagnostic `n_unique_activities_after_dedup` | `test_stages01_04.py` |
| L12 | Stage 01, this section | `SOCIO_LABELS` corrected (`DDV1` day of the week, `DDV3` month, `DDV6` employed or student, `HHC3` and `HHC4` by age, Section 3.17); the quarter recoding and the at-home filter documented below | None | none needed |
| L13 | `peaks.py`, Stage 08 | Clock times rounded to the minute and reduced modulo 24 h | At most one minute | `test_peaks.py` |
| L14 | Stages 09, 10 | Diarists counted as distinct persons (`HID`, `PID`) per household-day | Console totals of Stage 10, `diarists_per_household_summary.csv` now per household-day | `test_stage09_sharing.py` |
| (export) | Stage 09 | `shared_activities_frequency.csv` gives the household-days in which each code is shared, and pools as "other" the codes shared in fewer than 50 household-days | Codes of rare shared activities no longer listed one by one | `test_stage09_sharing.py` |
| L15 | Stage 06 | A day type without published results is titled by its size class and left out of the count of countries | Figure titles, console | end-to-end run |
| L16 | Stage 11 | Variables from the Stage 07 ranking (the six largest V of the global solution, `HETUS_DRIVER_TOP_N`) | Choice of driver figures | `test_figures_and_drivers.py` |
| L17 | Stage 03 | The "deduplication impact" printout and its 61.5% are removed | Console | none needed |

Option A has a cost: the mean profile of the largest group of a solution
includes its small clusters, so it is not the profile of the largest cluster
alone. In the synthetic run the six planted night-peak days show as a small
bump at 02:30 in the profile of the largest Serbian group. The small clusters
of the run of 14 October 2026 (32 of fewer than 20 household-days and 17 of 20
to 49, out of 234) will be examined against this when the stratified results
are compared with version 1.0.0.

Stage 06 publishes the household-days of every country, of every country and
day type and of each day type over all countries. A country and day type below
50 could therefore be derived from the others even if its own rows were
withheld, and suppressing enough further figures to prevent that would remove
the day-type totals the manuscript quotes. Since the smallest country and day
type of HETUS 2010 has 946 household-days, Stage 06 instead stops with a
message naming the stratum, and the country can be left out with
`HETUS_COUNTRIES`. For the same reason Stage 07 withholds a test on fewer than
50 observations rather than flagging it, because its chi-squared statistic and
V together give N.

The number of clusters is unchanged in this version: K is still chosen by
`KneeLocator` on the grid 2 to 15 with a fallback of 8 for the global solution
and on 2 to 10 with a fallback of 4 for the stratified ones. The run on the
microdata of 19 November 2026 answers check C3: no knee was found, and K = 4 is
the fallback, in 15 of the 47 stratified solutions (the national solutions of
Italy, the Netherlands and Serbia, the working days of all countries, and 11
of the 30 national day-type solutions), while the global K = 9 comes from the
knee. The rule is kept, `k_selection` records it in every export, and K enters
the sensitivity analysis as a factor (X16, X17).

Stage 01, documented here for L12, recodes `DDV3` (month of the diary day) to
the first month of its quarter (1, 4, 7 and 10), and keeps only the diarist
records with at least one slot at home (`Wherep` = 1). A diarist away from
home all day is therefore not in the data, and the first-listed diarist whose
person-level attributes a household-day takes (Section 4) is the first listed
among those at home at least once.

The matching of household members to the same diary day (`HID` and `DIARY`,
open risk of the audit and check C1) is not changed by this version.

`tools/collect_exports.py` gathers the files of a run that may leave the
machine holding the microdata: the aggregate exports, each tested again with
the rules of `disclosure.py` (size columns and, for mean profiles,
`check_export`), the main figures, and the solution-level lines of a saved
console log. It never copies a working file, and its manifest lists what was
copied, refused or missing (`test_collect_exports.py`).

### 3.16 Stage 03 as sparse products (version 1.1.0)

Stage 03 used to loop over household-days, slots and diarists, building a
small table for every slot, and took 10 to 15 hours on HETUS 2010. It now
reads the diarist records once into arrays and computes every slot of every
household-day together. The power of a household-day in a slot is the sum of
the appliance powers of the distinct activity codes, main or secondary, of
the diarists at home, so the power of all household-days is a sparse product
p = A w: A has one row per household-day and slot and one column per activity
code, with a 1 where the code is in use at home, and w holds the powers of the
mapping table. The sensitivity analysis needs only other vectors w, and other
matrices for its structural factors: `HETUS_SAVE_ACTIVITY_MATRIX=1` writes the
matrix of codes in use (union), the matrix of codes used as a main activity by
at least one diarist at home, and the number of diarists at home using each
code, each diarist counted once per code, to `activity_matrix.npz`, with the
household-days of its row blocks in `activity_matrix_households.csv`. The
sensitivity analysis now builds its own inputs from the outputs of Stages 02
to 04 and the raw files (`gsa/bundle.py`, `docs/GSA.md`), checked against the
power of Stage 03, so it does not need these matrices.

The new stage reproduces the loop exactly. Power is a sum of whole watts. The
metabolic rate of a slot is summed over the diarists at home in the order of
the loop and with the same arithmetic (sequential for up to seven terms,
numpy's own summation for eight or more), and the person totals are
accumulated slot by slot as before. The loop is kept in
`pipeline/aggregate_household_loop_reference.py`, which writes to
`processed/stage03_loop_reference/`, and `tools/compare_stage03.py` compares
every value of the two (`test_stage03_fast.py`: households of two to nine
diarists with empty, negative and unmapped secondary codes and empty
locations, compared value by value, the long file included). On 48,182
synthetic diarist records it took 21 seconds and 1.1 GB of memory, so a few
minutes for HETUS 2010. The long file, which no stage reads, is written only
with `HETUS_WRITE_LONG=1`, and the daily totals are computed in memory.

The stage printed, by country, the share of at-home diarist-slots with
secondary code 999 (other unspecified time use), to check whether the
Netherlands, where no at-home slot has an empty secondary code (audit check
C5), code an absent secondary activity as 999. They do (89.9% of the Dutch
at-home slots), and Stage 01 now sets the code to no secondary activity and
prints the shares (Section 3.17).

### 3.17 Changes after the check of the codes (version 1.1.0, November 2026)

Every code used by the pipeline, the mapping table and the sensitivity
analysis was checked against the variable list of the scientific use files
(Annex 2 of Eurostat's anonymisation guide of HETUS wave 2010 data, 23 January
2020) and the 2008 guidelines (Annex V for activities and locations), with
`check_codes.py` run on the microdata for what the documents leave open
(Section 6). The run of 19 November 2026 compared with version 1.0.0 is
reported separately. Seven changes follow, each with a test where one is
needed.

| ID | Where | Change | What changes in the outputs | Test |
|---|---|---|---|---|
| K1 | Stage 01 | Same-day rule: a household-day (`HID`, `DIARY`) whose members at home kept their diaries on different weekdays (`DDV1`), or on the same weekday in different months (`DDV3` before its recoding to quarters) or years (`DDV4`), is left out; Stage 01 prints the counts by country, in classes below 50; `HETUS_SAME_DAY_RULE=0` keeps every household-day | Every output: on the microdata, about 470 to 680 household-days (0.24 to 0.36%) are left out | `test_sameday_999.py` |
| K2 | Stage 01 | Secondary code 999 is set to no secondary activity in every country, and Stage 01 prints its share of the at-home slots by country (NL 89.9%, IT 2.2%, UK 0.05%, FR 0.03%, elsewhere 0) | Metabolic columns and the dominant activity (999 has 0 W); the activity matrices | `test_sameday_999.py` |
| K3 | Stage 09 | Code 999 is not a valid code, as main or as secondary activity: two diarists whose activities are both unspecified are not known to share one | Shares of Stage 09, slightly (999 was 0.2% of the shared instances) | `test_stage09_sharing.py` |
| K4 | `data/HETUS_mapped_data.csv` | 422 (help in employment and farming) and 424 (other childcare as help to another household) draw 0 W instead of 5 W, like paid work (111) and childcare (381 to 389, 423); in the 2000 list these numbers meant household upkeep and construction and repairs as help (Annex V, Table 10.1), where 5 W would match 329 and 351 to 359 | Power at home in the slots with these codes (rare) | `test_stages01_04.py` reads the table |
| K5 | Stages 01, 11 | Labels of `DDV6` (employed or student, diary question 6a), `HHC3` (persons aged under 7), `HHC4` (persons aged 7 to 17) and `HHC5` (one, or two or more, persons aged 18 or over) | Driver figures | none needed |
| K6 | `tools/collect_exports.py`, `run_pipeline.sh` | The collector also extracts the lines of Stage 01 tagged `[same-day rule]` and `[secondary 999]`, and stops with a message when none of the tables is found; the runner prints the processed folder and how to collect the exports | None | `test_collect_exports.py` |
| K7 | `data/HETUS_mapped_data.csv` | `Light_use` of 342 (tending domestic animals) set to 0 and of 343 (caring for pets) to 1: the two were swapped against the rule of the table, by which outdoor activities (322, 341, 344, 349) need no light | None in the pipeline, which reads no `Light_use`; the lighting factor X19 of the sensitivity analysis | `test_gsa_model.py` reads the table |

The same-day rule cannot see a postponement by whole weeks within one month,
which the guidelines prescribe (a diary is moved by 7, 14 or 21 days, to the
same weekday, Annex I, 6.2), because the day of the month (`DDV2`) is not in
the scientific use files. `check_codes.py` found the detectable cases with the
same weekday in another month in fewer than 20 multi-diarist household-days in
ten countries, none in Romania and 1.95% in Luxembourg, which bounds the
undetectable ones at well under 1% of the multi-diarist household-days.

### 3.18 Console output under the publication rules (version 1.1.0, November 2026)

The console log of a run is a working file, like the processed folder. A
check of the log of the canonical run of 20 November 2026 found figures in it
that the publication rules do not allow off the machine. Stage 10 printed the
number of household-days for every number of diarists, down to classes of a
few household-days, and the largest number of diarists by country, and drew
the same in its figures. Stage 05 printed, for each background variable, the
table of shares by cluster and category, the shares of single categories in
the most distinctive clusters and in the cluster profiles, from which the
sizes give the cell counts, small ones included, and it printed the sizes and
profiles of the clusters as found, so also of any cluster below 50
household-days.

Stage 10 now pools the largest numbers of diarists into one class of at least
50 household-days (`disclosure.pool_tail`; 7 and more in HETUS 2010, as in
Stage 09), gives no minimum or maximum, and draws its box plot from the 5th to
the 95th percentile without single household-days. Stage 05 reports the
published groups in every printout and figure of its categorical analysis
(sizes, distances, pairwise comparisons, profiles and the figure of each
variable), prints no table of shares, and prints a share only where neither
the count behind it nor the rest of its group is below 20 (`share_text`). Its
test statistics, the files `categorical_feature_statistics.csv` and
`clustering_results.csv`, and every export are unchanged. The diagnostic
figures of Stages 05 and 11 still show the composition of published groups by
category, which for rare categories rests on few household-days, so they stay
on the machine with the log. `tools/collect_exports.py` takes several logs,
for a run made in parts (`--log run_log.txt run_log_stage05.txt`), and copies
only its extract of them, and `run_log*.txt` is ignored by git. Tests:
`test_stage10_console.py`, `test_disclosure.py` (`pool_tail`) and
`test_gsa_model.py` (the console of Stage 05 on a run with clusters below 50
household-days).

### 3.19 Checks for the reviewers (November 2026)

Three scripts in `tools/` answer requests of the reviewers that the
sensitivity analysis does not cover. Each reads the processed folder of a run
(the canonical run for the paper), runs in minutes on the machine that holds
the microdata, and writes aggregates only, under the rules of
`pipeline/disclosure.py`; `tools/collect_exports.py` copies their tables when
they exist.

`paired_day_types.py` (Reviewer 2, comment 5) compares working and
non-working days of the same households. A household with at least one
household-day of each type is paired, its days of one type averaged, and for
every country with at least 50 paired households it gives the change of the
coincident peak per household, of the mean daily peak and of the daily energy,
among all household-days as in Stage 06 and among the paired households, with
95% intervals from resampling households and Wilcoxon's signed-rank test on
the paired differences (`Country_DayType_Analysis/paired_daytype_comparison.csv`
and a figure; `--day-type ddv1` applies the weekday rule to every country).

`driver_robustness.py` (Reviewer 1, comment 5) repeats the tests of Stage 07
with Cramer's V corrected for bias (Bergsma 2013), a 95% interval from
resampling households rather than household-days, V on one household-day per
household, missing answers as a category, and the number of adults within
each number of diarists (`driver_robustness.csv`). The rows of each solution
are matched to the household-days by their order and checked against their
daily energy.

`clustering_robustness.py` (Reviewer 1, comment 3) gives the silhouette width
and the half-sample stability of Stage 05 for every stratified solution, and
compares the global K-means partition with a Gaussian mixture, K-means without
PCA and Ward's clustering on a subsample (`clustering_robustness.csv`).

Run after Stages 01 to 07, in one command that finds the processed folder of
the run, stops unless it holds the solutions of Stages 05 and 06, runs the four
checks and collects their results (only these) for sharing:

```bash
python3 tools/reviewer_checks.py --bundle ~/hetus_gsa/bundle   # the run the bundle was built from
python3 tools/reviewer_checks.py --processed "/path/to/processed" --out exports_checks
```

Each script also runs on its own with `--processed FOLDER` or `--bundle BUNDLE`
(else the folder of `pipeline/paths.py`), and `tools/collect_exports.py --checks`
collects their results only. Corrections of 26 November 2026, after the first
run on the microdata: a processed folder without the solutions stopped
`driver_robustness.py` with an error of pandas and let
`clustering_robustness.py` write an empty table, on which the collector failed;
the scripts now stop with a message saying what is missing and write nothing,
and the collector refuses an empty or unreadable table. On a complete run the
country files of Stage 06 could not be matched, because
`household_interval_timeseries.csv` codes the countries (Stage 01) while the
files name them (Stage 06), and the paired comparison labelled its countries
by code; both now use the names of `pipeline/countries.py`. The paired table
keeps all its columns when no country has 50 paired households (its printout
failed then). `run_folder.py` holds the shared lookup and checks of the
processed folder.

`pca_summary.py` prints the principal components of the global clustering as
the manuscript reports them (Section 3.2): the components that 80, 90 and 95%
of the variance need, the share of the first five, and the window of the day of
each (`python3 tools/pca_summary.py --bundle ~/hetus_gsa/bundle`).

Tests: `test_paired_day_types.py`, `test_driver_robustness.py`,
`test_clustering_robustness.py`, on synthetic data with known answers, and
`test_reviewer_checks.py`, which runs Stages 01 to 07 on synthetic raw files in
which most households keep a weekday and a weekend diary, then the one command
as a user runs it, and the stops on an incomplete folder.

## 4. Design choices that constrain interpretation

Person-level attributes come from the first-listed diarist. Stage 03 sets
`household_attrs = household_day_data[extra_cols].iloc[0]`, so `INC1`, `INC2`,
`INC4_1`, `IND6_1`, `IND22_1`, `IND44`, `DDV6` and `DDV7` on a household row
belong to one member (SUF person identifiers are random). This is stated in
the manuscript and limits how person-level drivers can be read.

Hungary, the Netherlands and Norway collected one diary per household, so
their household profiles describe one person. The manuscript keeps them in
all analyses and flags them wherever they affect a comparison.

## 5. Manuscript claims and the files that support them

Status refers to version 1.1.0, which the revised manuscript quotes
throughout: the canonical run of 20 November 2026 (Sections 3.15 to 3.18),
the checks of 26 November 2026 (Section 3.19) and the global sensitivity
analysis of 4,096 runs (`docs/GSA.md`). The files are in `outputs/`, whose
README describes its folders. The results of the run of 14 October 2026,
quoted in the first submission, are in versions 1.0.0 and 1.0.1 of the
archive. Section numbers are those of the revised manuscript, and S numbers
those of its Supporting Information.

| Claim (section) | Source file in `outputs/` | Value |
|---|---|---|
| Household-days analysed and the same-day rule (2.1, 2.7, Table 1) | `pipeline/country_comparison_summary.csv`; `pipeline/run_log_extract.txt` (`[same-day rule]` lines) | 191,740 household-days; 528 of 192,268 left out |
| Household-days with two or more diarists by country (Table 1) | `checks/driver_robustness.csv` (rows `HHC5 \| diarists 1` of the country solutions, counts rounded to ten) | 34% (PL) to 71% (EL), 44% overall |
| Mean power and daily energy (2.7) | `pipeline/cluster_summary.csv` (size-weighted mean of `avg_daily_energy_wh`) | 99 W, 2.38 kWh |
| Diarist coincidence (2.8) | `pipeline/same_activity_by_diarists.csv`, `activity_combination_counts.csv`, `shared_activities_frequency.csv` | mean 70.4% over 84,351 household-days, 68.7 to 90.1% by number of diarists; 89.8, 7.8 and 2.4% of the shared pairs; sleep 4.05 million, television 0.85 million, eating 0.57 million |
| Principal components (3.2) | `pipeline/run_log_extract.txt`; `checks/pca_summary.txt` | 91 components at 95% (68 to 106 in the stratified solutions), 40 and 66 at 80 and 90%; first five 8.23, 7.22, 4.91, 4.61 and 3.95% |
| Elbow, knee and fallbacks (3.3, 4.1, S10.4) | `pipeline/run_log_extract.txt`; `pipeline/stratified_cluster_mean_profiles.csv` (`k`, `k_selection`) | WCSS 24.8 to 18.5 million; knee at K = 8; fallback K = 4 in 16 of the 47 stratified solutions |
| Small clusters reported with the largest of their solution (3.4, 4.3) | `pipeline/stratified_cluster_mean_profiles.csv` (`n_small_clusters_merged`) | 45 of the 233 stratified clusters |
| Tests withheld or flagged (3.4) | `pipeline/consolidated_chi2_stats.csv` (`Publication_Class`) | 3 withheld (household income, Serbia), 5 flagged |
| Global clusters: sizes, energy, peaks, modal categories (4.1, Table 4) | `pipeline/cluster_summary.csv` | 8 clusters of 53 to 114,663 household-days |
| Mean profiles and sample-coincident peaks of the clusters (4.1, Figure 2) | `pipeline/cluster_mean_profiles.csv`, `cluster_summary.csv` | |
| Circular statistics of the peak time (4.1, Table 5) | `pipeline/consolidated_circular_stats.csv` | weighted mean SD 4.69 h; 469 household-days without demand |
| Stability and separation of the global solution (4.1) | `pipeline/cluster_stability.csv` | ARI 0.82 (0.67 to 0.92), silhouette 0.16, mean Jaccard 0.41 to 0.92 |
| Share of the retained variance between clusters (4.1) | `pipeline/run_log_extract.txt` (WCSS at K = 8) | about 21%, 1 - WCSS / (0.95 x 144 x 191,740) |
| Stability of the stratified solutions, other methods (3.5, 4.1, S10.3) | `checks/clustering_robustness.csv` | ARI 0.60 to 0.95 (median 0.81); without PCA 0.90, Ward 0.40, Gaussian mixture 0.04 |
| ANOVA (4.1, 4.2) | `pipeline/consolidated_anova_stats.csv` | F = 45,629.2 (global), 45,863.6 (working), 38,214.8 (non-working) |
| Day types: sizes and K (4.2) | `pipeline/mean_profiles_country_daytype.csv` (`n`); `pipeline/stratified_cluster_mean_profiles.csv` | 100,046 and 91,694 household-days; K = 4 (fallback) and 5 |
| Sample-coincident and mean daily peaks by country and day type (4.2, Table 6, Figure 3) | `pipeline/mean_profiles_country_daytype.csv`, `peak_power_daytype_comparison.csv` | rise in 15 of 15 countries, median 18.5% |
| The same households on both day types (4.2, S10.1) | `checks/paired_daytype_comparison.csv`, `paired_daytype_comparison_ddv1.csv` | 57,850 households in 12 countries; 22.8% (20.7 to 25.1) pooled |
| Countries: K, energy, peaks and their times (4.3, Figure 4) | `pipeline/country_comparison_summary.csv`, `country_daytype_combined_summary.csv` | |
| Recurring profile shapes in the national solutions (5.1, Figure A2) | `pipeline/stratified_cluster_mean_profiles.csv` | |
| Associations with the background variables (4.4, Tables 8 and 9) | `pipeline/consolidated_chi2_stats.csv` (Chi-squared, plain V); `checks/driver_robustness.csv`, `driver_summary.csv` (corrected V, household intervals, one household-day per household, missing answers as a category, number of diarists, number of adults within each number of diarists) | number of adults 0.177 (0.172 to 0.181) on the global solution |
| Outputs over the runs and percentiles of the canonical run (4.5, Table 10) | `gsa/results/summary.csv`; `gsa/share/gsa_outputs.csv` | |
| Total, first-order and OT indices, seeds (4.5, Figure 5, S5, S6) | `gsa/results/totals.csv`, `indices.csv`, `seeds.csv` | |
| Variance budget (4.5, S8) | `gsa/results/budget.csv`, `first_order_sums.csv`, `totals_check.csv` | |
| Convergence (4.5, S7) | `gsa/results/convergence_indices.csv`, `convergence_totals.csv` | |
| Agreement with the measured profiles (4.5, Table 11, Figure 7) | `gsa/results/validation_metrics.csv`, `profile_bands.npz` | |
| Bands of energy and sample-coincident peak by country (4.3, Table 7, S2) | `gsa/results/country_bands.csv` | 2.38 kWh (runs 1.82 to 4.99), 236 W (runs 145 to 355) |
| Country code mapping, exclusion of Germany | author confirmation | Verified |
| `INC4_1` categories | Eurostat (pers. comm., 2026) | Eight categories, not the five in the published list |
| `IND44` meaning | SUF Annex 2 | Permanent or fixed-term contract (employees only, not collected in NO) |

The aggregate files in `outputs/` contain no microdata. Every one of them
passed the publication rules of `pipeline/disclosure.py` (Sections 3.13 to
3.15, with Option A of Section 3.15: every cluster of fewer than 50
household-days is reported together with the largest cluster of its
solution) before it left the machine that holds the microdata.

## 6. Notes on the data documentation

`INC4_1` (main activity status) has eight categories in the data: 1 employed
full-time, 2 employed part-time, 3 on leave, 4 unemployed, 5 pupil, student,
further training or unpaid traineeship, 6 in retirement or early retirement or
has given up business, 7 fulfilling domestic tasks, 8 other person. Missing
codes are −6 and −9 only. The published list shows five categories, and
Eurostat has confirmed it will amend the documentation. `INC4_1` was not
collected in Luxembourg. Its apparent absence for Romania in the previous run
was the pipeline loss described in Section 3.3. Only four categories occur for
the Netherlands.

`IND44` is the permanent or fixed-term contract variable (employees only, not
collected in Norway). `04_prepare_clustering_data.py` previously described it
as citizenship, which is `IND42_1` and is not used.

`DDV7` (kind of day) is asked of persons who are employed or students
(`DDV6` = 1, diary question 6a, "Are you employed or a student?").
Its codes are 1 ordinary work day, 2 ordinary school day, 3 day off, 4 sick
leave, 5 vacation and 6 other leave, with −1 not applicable and −6 to −9 for
not asked, refusal, don't know and not available. The SUF documentation
states that it was not asked in DE, EL, ES, IT and LU, but the Greek microdata
contain it (41% of diary records valid in the 11 October 2026 run, with days
off on 28% of weekend records), so in practice it is missing for DE, ES, IT
and LU only. Estonia merges work and school days in code 1, as documented, and
never uses code 3: its days off carry codes 4 to 6 (38.9% of weekend records,
against 11.8% of weekday records). Since Stage 06 treats codes 3 to 6 alike,
the Estonian day types are not affected.

`Wherep` has 13 aggregated codes in the scientific use files, which the
documentation does not list (the 2008 guidelines code home as 11, not 1).
Their content on the microdata (`check_codes.py`, 15 countries pooled) fixes
their meaning: 1 home (75.0% of slots, 47% of them asleep, no travel), 2 a
weekend home or holiday apartment, 3 the workplace or school (84% paid work),
4 other people's homes, 5 restaurants, cafes and pubs (55% eating), 6 shops
and markets, 7 hotels and campsites (56% asleep), 8 to 12 travel (80 to 96%
travel codes) and 13 other or unspecified locations (5.3% of slots). The
German file codes every slot that is not travel as 13, which is why German
household-days cannot be used.

`HHQ6c` is the number of television sets, top-coded at one (0 none, 1 one or
more), while the other appliance items of `HHQ6` code 1 yes and 2 no. The
numbers of computers and mobile phones (`HHQ6o_1`, `HHQ6m_1`) are not in the
scientific use files, and `HHQ6p` is the number of cars or vans. `HHC5`
(persons aged 18 or over) takes the values 1 and 2 only (2 for two or more),
`HHC3` and `HHC4` (persons aged under 7 and 7 to 17) 0 to 2.

The files hold fewer records than Annex 1 of the anonymisation guide for
Finland (7,453 against 7,480), France (27,699 against 27,903), Poland (76,657
against 78,759) and the United Kingdom (16,078 against 16,118): exactly the
countries for which the variable list notes that some diarists have no person
weight (`WGHT2`), those without an individual interview in Finland, France and
the United Kingdom, and those under 15 in Poland. Every record of the files
has a person weight, so these diaries were left out of the release. Poland's
share of diarists aged 10 to 17 is accordingly 2.5%, close to Serbia's 3.1%
(diarists aged 15 or over), against 6 to 17% elsewhere except France (2.7%).
The diary dates (`DDV3`, `DDV4`) also differ from the fieldwork years of
Annex 1 for Belgium (January 2013 to January 2014, against 2012/2013),
Luxembourg (2014, against 2014/2015), Poland (2013, against 2012/2013) and
Romania (September 2011 to September 2012, against 2010/2011).

Section 4 of the anonymisation guide calls `HHQ9_1` the personal income and
`IND13_1` the household income. Its Annex 2, which the pipeline follows, gives
`HHQ9_1` as the household's net monthly income band and `IND13_1` as the
income from the main job.

The Stage 01 coverage table also shows variables with no valid values in
some countries: `DDV6` in ES and LU, `HHQ5` in BE, ES, FI and PL, `HHQ9_1` in
ES, HU, IT and NL, `INC4_1` in LU, `IND22_1` in NL and `IND44` in NO.

Hungary, the Netherlands and Norway collected one diary per household (SUF
Annex 1).

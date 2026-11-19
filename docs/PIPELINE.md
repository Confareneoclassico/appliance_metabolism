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
one in `data/` is missing). It has one row per HETUS activity code
(`activity`, the code used in `Mact` and `Sactn`) with its name
(`activity_name`), the corresponding code and label of the 2008 HETUS
guidelines (`TUS_activity_code`, `TUS_activity_label`), the activity of the
Compendium of Physical Activities behind the metabolic rate (`MET_info`), the
author's code for how the two codings correspond (`Mapping`), the
representative appliance power in W (`Appliance_Power_Watts`), the metabolic
rate in W for infants, children, teenagers, adults and seniors
(`MR_infant_Watts` to `MR_senior_Watts`), and flags for activities that need
light (`Light_use`) or hot water (`Water_use`), which no stage reads at
present. Its 108 codes have whole-watt powers, 30 of them above zero.

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
| 03 | `aggregate_household.py` | Stage 02 output, `HETUS_mapped_data.csv` | `household_interval_timeseries.csv` (wide), `household_interval_timeseries_long.csv` (long), `person_daily_metabolic_totals.csv`, `household_daily_totals.csv` (diagnostic only, not read by any later stage) |
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
| L12 | Stage 01, this section | `SOCIO_LABELS` corrected (`DDV1` day of the week, `DDV3` month, `DDV6` worked or studied); the quarter recoding and the at-home filter documented below | None | none needed |
| L13 | `peaks.py`, Stage 08 | Clock times rounded to the minute and reduced modulo 24 h | At most one minute | `test_peaks.py` |
| L14 | Stages 09, 10 | Diarists counted as distinct persons (`HID`, `PID`) per household-day | Console totals of Stage 10, `diarists_per_household_summary.csv` now per household-day | `test_stage09_sharing.py` |
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
and on 2 to 10 with a fallback of 4 for the stratified ones. Whether to use one
grid and rule at every level, as the audit recommends, depends on check C3,
which counts the fallbacks of the run of 14 October 2026, and K enters the
sensitivity analysis as a factor in any case.

Stage 01, documented here for L12, recodes `DDV3` (month of the diary day) to
the first month of its quarter (1, 4, 7 and 10), and keeps only the diarist
records with at least one slot at home (`Wherep` = 1). A diarist away from
home all day is therefore not in the data, and the first-listed diarist whose
person-level attributes a household-day takes (Section 4) is the first listed
among those at home at least once.

The matching of household members to the same diary day (`HID` and `DIARY`,
open risk of the audit and check C1) is not changed by this version.

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

Status refers to the energy-only run of 14 October 2026 (Section 3.12), which
the manuscript quotes throughout (from v6). Its aggregate results are in
`outputs/`. The diarist coincidence results, the national
daily energy, peak power and peak times, and the comparisons of working and
non-working days do not depend on the clustering and are identical to those
of the run of 11 October 2026.

| Claim | Source file | Status |
|---|---|---|
| ANOVA F-statistics, all levels | `consolidated_anova_stats.csv` | Refreshed (global F = 40,946.7, working days 52,289.7, non-working days 30,913.5, smallest 287.9) |
| Chi-squared and Cramér's V, all levels | `consolidated_chi2_stats.csv` | Refreshed (48 strata). Compared with the 11 October run: median ratio of V 0.97 over 684 pairs |
| PCA variance shares, component count, elbow K | Stage 05 console log | Refreshed (144 slots reduced to 91 components at 95%, 40 at 80%, 66 at 90%, knee detected at K = 9) |
| Share of retained variance between clusters (22.5%) | Stage 05 console log | 1 - WCSS(K = 9) / (cumulative variance at 91 components x 144 x 192,268) |
| Global cluster sizes and modal categories | `cluster_summary.csv` | Refreshed (K = 9) |
| Global cluster daily energy, mean daily peak and coincident peak | `cluster_summary.csv`, `cluster_mean_profiles.csv` | Refreshed (coincident peaks now read from the file, not from Figure 2) |
| Circular peak-time statistics | `consolidated_circular_stats.csv` | Refreshed (466 zero-demand days, all in Cluster 0). Global solution only from version 1.0.1 (Section 3.14) |
| Stability of the global solution | `cluster_stability.csv`, Stage 05 console log (ARI of each half-sample) | New (silhouette 0.13, mean ARI 0.89, Jaccard 0.69 to 0.95) |
| Day-type shares and day-type clustering | Stage 06 console log, `DayType_Analysis/` | Refreshed (100,306 working and 91,962 non-working household-days, K = 4 and 6) |
| Coincident and mean daily peaks by country and day type | `mean_profiles_country_daytype.csv`, `peak_power_daytype_comparison.csv`, Stage 06 console log | New (coincident peak higher on non-working days in 15 of 15 countries) |
| Per-country K, daily energy, peaks and peak times | `country_comparison_summary.csv` | Refreshed (K between 4 and 7) |
| Country-by-day-type comparison (energy, peak, peak time, K) | `country_daytype_combined_summary.csv` | Refreshed |
| Recurring profile shapes in the national solutions | `stratified_cluster_mean_profiles.csv` | Checked for all 15 countries (Figure S2). Not published from version 1.0.1 (Section 3.14) |
| Diarist coincidence percentages | `same_activity_by_diarists.csv` | Stage 09 rewritten for version 1.1.0 (Section 3.15, D3); the figures of the manuscript, from an unpublished version, will be replaced by those of the run on the microdata |
| Shared activity counts (sleep, TV, eating) | `shared_activities_frequency.csv`, `activity_combination_counts.csv` | Stage 09 rewritten for version 1.1.0 (Section 3.15, D3): each diarist counted once per code, at home only |
| Country code mapping, exclusion of Germany | author confirmation | Verified |
| `INC4_1` categories | Eurostat (pers. comm., 2026) | Eight categories, not the five in the published list |
| `IND44` meaning | SUF Annex 2 | Permanent or fixed-term contract (employees only, not collected in NO) |

The aggregate files in `outputs/` contain no microdata and follow the
publication rules of Section 3.13, with the withdrawals of Section 3.14. The files of the run of 14 October 2026
were brought under these rules after the run, with the same functions the
stages now apply. The rows that join each largest cluster with the withheld
clusters of its solution were computed from the mean of the stratum minus the
other clusters, which the aggregates of that run hold exactly.

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

`DDV7` (kind of day) is asked of persons who worked or studied (`DDV6` = 1).
Its codes are 1 ordinary work day, 2 ordinary school day, 3 day off, 4 sick
leave, 5 vacation and 6 other leave, with −1 not applicable and −6 to −9 for
not asked, refusal, don't know and not available. The SUF documentation
states that it was not asked in DE, EL, ES, IT and LU, but the Greek microdata
contain it (41% of diary records valid in the 11 October 2026 run), so in
practice it is missing for DE, ES, IT and LU only. Estonia deviates from the
common definition.

The Stage 01 coverage table also shows variables with no valid values in
some countries: `DDV6` in ES and LU, `HHQ5` in BE, ES, FI and PL, `HHQ9_1` in
ES, HU, IT and NL, `INC4_1` in LU, `IND22_1` in NL and `IND44` in NO.

Hungary, the Netherlands and Norway collected one diary per household (SUF
Annex 1).

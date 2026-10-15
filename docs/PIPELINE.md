# HETUS electricity profiling pipeline: stage-by-stage reference

This document maps how data flow through the eleven pipeline stages. It was
reconstructed from the scripts' own `pd.read_csv` and `to_csv` calls, and it
records every correction made to the scripts while the manuscript was being
prepared, together with the pipeline output that supports each quantitative
claim in the manuscript.

## 0. Inputs not included in this repository

The raw HETUS country files (`{COUNTRY_CODE}.csv`, one per country: BE, DE,
EE, EL, ES, FI, FR, HU, IT, LU, NL, NO, PL, RO, RS, UK) and
`HETUS_mapped_data.csv` (the lookup table from activity code to appliance
power and to metabolic rate) are not included. They are restricted-access
Eurostat Scientific Use Files and must be obtained separately and placed
together in `data/raw/` (or in the folder given by `HETUS_DATA_DIR`) before
Stage 01 is run. The run of 14 October 2026 used the release of November
2020.

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
| 09 | `same_diarists_coincidence.py` | `lu_data_at_home_with_dominant_activity.csv` | `same_activity_household_summary.csv`, `same_activity_interval_details.csv`, `activity_combination_counts.csv`, `shared_activities_frequency.csv`, figures |
| 10 | `p_h_counting.py` | `person_daily_metabolic_totals.csv` | `diarists_per_household_summary.csv`, figures |
| 11 | `visualize_cluster_drivers.py` | per-stratum files from Stage 06 | driver-distribution figures (no CSV) |

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

Stage 11 hard-codes `TOP_FEATURES = ["HHC5", "HHC1", "INC2", "DDV6"]` instead
of reading the strongest drivers from Stage 07. The list should be checked
against the re-run `consolidated_chi2_stats.csv` before new driver figures are
produced, because `HHC1` and `INC2` were only weakly associated with cluster
membership in most strata of the previous run.

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
| `stratified_cluster_mean_profiles.csv` | The same for every cluster of every stratified solution, with size, share, daily energy and both peaks, under the publication rules of Section 3.13 |
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
from `stratified_cluster_mean_profiles.csv` alone, so it needs no microdata.
`run_pipeline.sh` runs it after Stage 06, and it can be run on its own with
the path of an export file as argument. Clusters are coloured by their size
rank within the country, withheld clusters are not drawn (their
household-days are part of the largest cluster), and flagged ones are drawn
with dashed lines.

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
| Circular peak-time statistics | `consolidated_circular_stats.csv` | Refreshed (466 zero-demand days, all in Cluster 0) |
| Stability of the global solution | `cluster_stability.csv`, Stage 05 console log (ARI of each half-sample) | New (silhouette 0.13, mean ARI 0.89, Jaccard 0.69 to 0.95) |
| Day-type shares and day-type clustering | Stage 06 console log, `DayType_Analysis/` | Refreshed (100,306 working and 91,962 non-working household-days, K = 4 and 6) |
| Coincident and mean daily peaks by country and day type | `mean_profiles_country_daytype.csv`, `peak_power_daytype_comparison.csv`, Stage 06 console log | New (coincident peak higher on non-working days in 15 of 15 countries) |
| Per-country K, daily energy, peaks and peak times | `country_comparison_summary.csv` | Refreshed (K between 4 and 7) |
| Country-by-day-type comparison (energy, peak, peak time, K) | `country_daytype_combined_summary.csv` | Refreshed |
| Recurring profile shapes in the national solutions | `stratified_cluster_mean_profiles.csv` | Checked for all 15 countries (Figure S2) |
| Diarist coincidence percentages | `same_activity_household_summary.csv` | Verified. Unaffected |
| Shared activity counts (sleep, TV, eating) | `shared_activities_frequency.csv` | Verified (7,980,406, 1,172,172 and 1,048,520). Unaffected |
| Country code mapping, exclusion of Germany | author confirmation | Verified |
| `INC4_1` categories | Eurostat (pers. comm., 2026) | Eight categories, not the five in the published list |
| `IND44` meaning | SUF Annex 2 | Permanent or fixed-term contract (employees only, not collected in NO) |

The aggregate files in `outputs/` contain no microdata and follow the
publication rules of Section 3.13. The files of the run of 14 October 2026
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

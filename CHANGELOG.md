# Changelog

## Unreleased

### Fixed (26 November 2026, checks for the reviewers)

- `tools/driver_robustness.py` matched no country solution of Stage 06 on a real run (the
  household file codes the countries, the solution files name them) and failed with a pandas
  error when the processed folder held no solutions; `tools/clustering_robustness.py` then wrote
  an empty table and `tools/collect_exports.py` failed on it. The scripts now check the folder
  first and stop with a message, the collector refuses empty tables, and the paired comparison
  names its countries and keeps all its columns.
- New `tools/reviewer_checks.py` (one command: `--bundle ~/hetus_gsa/bundle` or `--processed`)
  and `tools/run_folder.py`; `collect_exports.py --checks`; end-to-end tests on a synthetic run
  with two diary days (`tests/test_reviewer_checks.py`).

Checks for the reviewers: `tools/paired_day_types.py` (working and
non-working days of the same households), `tools/driver_robustness.py`
(Cramer's V corrected for bias, households resampled, one day per household,
missing answers as a category, the number of adults within each number of
diarists) and `tools/clustering_robustness.py` (stability of the stratified
solutions, other clustering methods), each with tests on synthetic data and
picked up by `tools/collect_exports.py` (docs/PIPELINE.md Section 3.19).
`gsa.validation` gives the agreement with the measured profiles in the
metrics of the reviewers (correlation, error in the hour of the peak,
normalised RMSE) and the bands of daily energy and coincident peak by country.
The convergence figures show every factor (square-root scale, the influential
ones labelled), for the total, first-order and OT indices, and the variance
budget separates the part of the factors (one less the total index of the
seeds) into what the surrogate attributes and what it does not.

`gsa/analyse.py`, `gsa/figures.py` and `gsa/tables.py` compute, draw and
tabulate the sensitivity analysis from the shared files (`docs/GSA.md`
Section 8). The Sobol' columns turned out to be dependent at the scale of the
classes of the given-data estimators, so the first-order and OT indices are
adjusted by raking the runs of each class to the shares of the leading
factors of the output, quantile classes never number a power of two, and
the Shapley effects come from a gradient-boosting surrogate evaluated on
independent points, with the nearest-neighbour estimator kept as a check. An
index counts as non-zero only above zero as well as above its dummy. The
analysis takes every nested design up to the runs present (4,096 in the
sample of 21 November), where the columns are close to balanced and the raw
indices agree with the adjusted ones of 1,024 runs. A stability table follows
the screening at 0.05 and the ranking over the last two doublings, the
adjustment table and figure compare 1,024 runs with the largest sizes, and
the kNN table compares the two estimators at the same size. The analysis also
gives the uncertainty analysis of the outputs (moments, bimodality, spikes;
figure of their distributions by diarist coverage), the mean, spread and
shape parts of the OT index against the first-order index, and the
first-order index, Shapley effect and total index of the surrogate for every
factor, the appliance powers one by one. `gsa.reseed` reruns points of the
sample with the seeds redrawn, and the analysis turns the pairs of runs into
the total index of the seeds on the model and a variance budget of every
output (factors, seeds, error of the surrogate). The first-order and OT
indices are now the plain class-based estimators by default (`--adjusted`
restores the raking, which at 4,096 runs moves no first-order index by more
than 0.006). Part `totals` gives the first-order and total indices of the
surrogate on every nested size, with bootstrap intervals, screening and
convergence statistics at the largest, and the figures and tables lead with
the total indices (`fig_totals`, `tab_totals`, `tab_convergence_totals`, the
stability table and the convergence figure). The reruns also say which
factors modulate the effect of the seeds (`seed_dispersion.csv`,
`seed_modulators.csv`, `tab_seeds`).

The console follows the publication rules as the exports do (`docs/PIPELINE.md`
Section 3.18). Stage 10 pools the largest numbers of diarists into one class
of at least 50 household-days and gives no minimum or maximum, in the console
and in its figures. Stage 05 reports only the published groups in its
categorical analysis, prints no table of shares by cluster and category, and
prints a share only where no count below 20 can be worked out from it.
`tools/collect_exports.py` reads several logs, and saved logs are ignored by
git. The sensitivity analysis uses 20 K-means starts by default, as the
pipeline does. In the mapping table, `Light_use` of 342 (tending domestic
animals) and 343 (caring for pets) are swapped back, so that outdoor
activities need no light; only the lighting factor of the sensitivity
analysis reads the column.

The lookup table from activity code to appliance power and metabolic rate is
now part of the repository, as `data/HETUS_mapped_data.csv` under CC BY 4.0,
and `pipeline/paths.py` reads it from there (or from `HETUS_MAPPING_FILE`),
falling back to a copy next to the raw files. The README and `docs/PIPELINE.md`
no longer describe it as a Eurostat file.

All the defects found by the audit of November 2026 are corrected, each with a
test on synthetic data in `tests/` (`docs/PIPELINE.md` Section 3.15 lists
them). Every cluster of fewer than 50 household-days is now reported together
with the largest cluster of its solution in every export and figure, every
table of group means is tested before it is written, and Stage 06 stops if a
country and day type has between 1 and 49 household-days. Stage 09 is
rewritten: a slot is comparable when at least two distinct diarists are at
home with a valid activity code, and shared when at least two of them record
the same code. Peak times share the weight of a household-day among tied
maxima and leave out days without demand. Not applicable is kept apart from
non-response, which is counted among the units an item applies to. Stage 06
records whether each K came from a knee or the fallback. `cluster_summary.csv`
gives the SD of daily energy (`daily_energy_sd_wh`) instead of `energy_std`.
The Stage 11 figures are no longer transposed and draw the variables ranked
first by Stage 07. Stage 03 adds up daily totals by household-day and sums the
metabolic rate over the diarists at home. Country codes come from the file
names, and a missing raw file stops Stage 01 (`HETUS_COUNTRIES` selects a
subset). Relative `HETUS_*` locations are taken from the repository root. The
smaller corrections are in Section 3.15. The outputs in `outputs/` are still
those of the run of 14 October 2026 and will be replaced by those of the
corrected pipeline. `tools/collect_exports.py` copies the exports of a run that
may be shared, tested again with the publication rules, and Stage 09 pools the
codes shared in fewer than 50 household-days.

The codes were checked against the variable list of the scientific use files
and the 2008 HETUS guidelines (`docs/PIPELINE.md` Section 3.17 and Section 6).
Stage 01 leaves out the household-days whose members kept their diaries on
different weekdays, or in different months or years (`HETUS_SAME_DAY_RULE=0`
keeps them), and treats secondary code 999, with which the Dutch files mark an
absent secondary activity, as no secondary activity, and Stage 09 does not
count code 999 as a shared activity. In the mapping table, activities 422 and 424
draw 0 W instead of 5 W. The labels of `DDV6`, `HHC3`, `HHC4` and `HHC5`
follow the variable list. The collector also extracts the tagged lines of
Stage 01 and stops when it finds none of the tables, and the runner prints the
processed folder and how to collect the exports.

The global sensitivity analysis of the revision is in `gsa/` (`docs/GSA.md`).
`gsa.bundle` builds the inputs of its model from a run and stops unless they
give the power of Stage 03 exactly; `gsa.model` repeats Stages 03 and 05 to 08
for any values of the 21 factors (appliance powers, country differences,
shared appliances, ownership, secondary activities, base load, diarist
coverage, day type, survey weights, K, PCA, lighting in darkness at the
capital, laundry run-on and the K-means seed) and gives the 11 outputs and the
mean profiles; `gsa.check` verifies that at the values of the run the model
gives the same partitions and statistics as the pipeline; `gsa.run` makes the
runs of the Sobol' sample in parallel and resumes an interrupted sample; and
`gsa.collect` gathers the aggregate outputs that may be shared. The tests run
Stages 01 to 08 on synthetic raw files and check each rule of the model
against a computation of its own.

Stage 03 computes the household power as a sparse product p = A w and runs in
minutes instead of 10 to 15 hours, with the same values to the last bit (its
former loop is kept in `pipeline/aggregate_household_loop_reference.py` and
`tools/compare_stage03.py` compares the two). It writes the long file only with
`HETUS_WRITE_LONG=1`, and with `HETUS_SAVE_ACTIVITY_MATRIX=1` it saves the
matrices of activity codes in use that the sensitivity analysis needs
(`docs/PIPELINE.md` Section 3.16).

## 1.0.1 (19 November 2026)

doi:10.5281/zenodo.24368824

This version withdraws published figures from which the sizes of small
clusters could be derived. The code is unchanged, and so are the results quoted
in the manuscript.

`outputs/stratified_cluster_mean_profiles.csv` is withdrawn. Slot powers are
whole watts, so each published mean profile is a vector of integers divided by
the size of its cluster, and the smallest size that makes all 144 products
integers is the size itself. This recovered the sizes of the clusters of 20 to
49 household-days, which Eurostat's rules allow to be reported only as that
range, and the exact sizes behind the values rounded to the nearest 10.
Together with the size of the largest cluster alone in
`country_daytype_combined_summary.csv`, it also bounded the household-days of
the withheld clusters (fewer than 20) to at most ten values. Rounding the
profiles would not have been enough, because the stratum profiles and sizes in
`mean_profiles_country_daytype.csv` allow the cluster sizes to be estimated by
least squares.

`country_daytype_combined_summary.csv` no longer has the columns
`largest_cluster_size` and `largest_cluster_pct`, and
`consolidated_circular_stats.csv` keeps only the nine clusters of the global
solution, the only ones quoted in the manuscript, since the rounded sizes of
the stratified clusters, with the exact stratum sizes, bounded the withheld
clusters more narrowly than the class "fewer than 20". No remaining file holds
a group of fewer than 50 household-days or a breakdown of a stratum into
clusters. `docs/PIPELINE.md` Section 3.14 gives the details.

Version 1.1.0 will publish the stratified results again, with every cluster of
fewer than 50 household-days reported together with the largest cluster of its
solution in every file, and with a test of every export before it is written.

The same audit found the following issues, which do not concern disclosure and
will be corrected in version 1.1.0, together with the revision of the
manuscript. Stage 09 counts all 144 slots in the denominator of the share of
slots with a shared activity, does not use the location of the diarists, and
counts twice a code that a diarist records as both main and secondary
activity. Peak times are taken at the first of several slots that reach the
daily maximum, and days without any demand enter the circular statistics at
04:00. Stages 01 and 04 merge the code for not applicable (−1) with refusal,
don't know and no answer (−7 to −9), and Stage 07 exempts `IND6_1` and `IND44`
from the non-response rule on the mistaken premise that the scientific use
files do not separate the two. With the grid of 2 to 10 clusters,
`KneeLocator` rarely finds a knee in the stratified solutions, which then take
the fallback value of 4 clusters. `energy_std` in `cluster_summary.csv` is the
pooled standard deviation of 10-minute power in W, not that of daily energy.
Stage 02 looks up metabolic rates by `TUS_activity_code`, which differs from
the activity codes in every row of the mapping table, so the dominant activity
is always the main one, which affects the metabolic columns only.

## 1.0.0 (16 October 2026)

First archived version, behind the manuscript as submitted (run of 14 October
2026), doi:10.5281/zenodo.23412742.

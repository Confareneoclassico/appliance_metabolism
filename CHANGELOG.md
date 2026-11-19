# Changelog

## 1.0.1 (19 November 2026)

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

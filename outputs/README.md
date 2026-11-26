# Aggregate results of version 1.1.0

This folder holds the aggregate results behind the tables and figures of the
revised manuscript ("Activity-Associated Appliance-Load Profiles of European
Households from Time-Use Surveys: Multi-Level Clustering, Drivers and
Robustness"). No file holds microdata: every value is an aggregate over at
least 50 household-days and passed the publication rules of
`pipeline/disclosure.py` (Eurostat's guidelines for the HETUS scientific use
files) before it left the machine that holds the microdata. Section 5 of
`docs/PIPELINE.md` gives the file behind each claim of the manuscript.

The results of the run of 14 October 2026, quoted in the first submission,
are in versions 1.0.0 and 1.0.1 of the archive.

| Folder | Content | Made by |
| --- | --- | --- |
| `pipeline/` | Exports of the canonical run of 20 November 2026 (Stages 01-11 of version 1.1.0 on the HETUS 2010 microdata, release of November 2020): cluster summaries, mean profiles, stability, ANOVA and Chi-squared tests, circular statistics, country and day-type summaries, diarist coincidence; `run_log_extract.txt` keeps the console lines with solution-level figures only (PCA, WCSS, the choice of K, the same-day rule in classes) and `manifest.txt` the record of the collector | `tools/collect_exports.py` |
| `pipeline/figures/` | Figures of the run drawn from the published groups only. `country_comparison_summary.png` and `country_daytype_combined_summary.png` were drawn from the exported tables with the functions of Stage 06, since the collector of that run did not yet copy them | Stages 05, 06, `plot_country_cluster_profiles.py` |
| `checks/` | Checks on the microdata for the revision, on the processed folder of the canonical run (26 November 2026): the same households on both day types, under the day types of Stage 06 and under the weekday rule; Cramér's V with the bias correction, households resampled, one household-day per household and missing answers as a category; the stability of the 47 stratified solutions and other clustering methods; the principal components of the global solution (`pca_summary.txt`) | `tools/reviewer_checks.py`, `tools/pca_summary.py` |
| `gsa/share/` | Outputs of the 4,096 runs of the global sensitivity analysis (`gsa_outputs.csv`, one row per run) and the mean profile of every country and day type in every run (`gsa_profiles.npz`, profiles of fewer than 50 household-days blanked), with the check of the model against the canonical run (`baseline_check.txt`) | `gsa.run`, `gsa.collect`, `gsa.check` |
| `gsa/reseed/` | The first 256 points run again with both seeds redrawn (`seeds`), with the seed of K-means alone redrawn (`kmeans`), and three points run again unchanged (`same`, the check that the model is deterministic) | `gsa.reseed`, `gsa.collect` |
| `gsa/uniform/` | 256 runs with the appliance powers uniform in watts, an earlier check of the law of the powers that the paper no longer reports | `gsa.run --uniform-in-watts` |
| `gsa/results/` | Everything the analysis computes from the shares: summaries of the outputs, first-order, optimal-transport and total indices with their bootstrap intervals, convergence on the nested designs, the seeds, the variance budget, the agreement with the measured profiles and the bands of energy and peak by country | `gsa.analyse`, `gsa.validation` |

`gsa.figures` and `gsa.tables` redraw the figures and tables of the paper and
its Supporting Information from `gsa/results/` and `gsa/share/`
(`docs/GSA.md` Section 8).

The results are based on data from Eurostat, Harmonised European Time Use
Surveys (HETUS), 2010 wave, scientific use files, release of November 2020.
The responsibility for all conclusions drawn from the data lies entirely with
the author. They are released under the Creative Commons Attribution 4.0
International License (`LICENSES/CC-BY-4.0.txt`).

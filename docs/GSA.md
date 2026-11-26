# Global sensitivity analysis

The sensitivity analysis asks how much each modelling choice of the pipeline
moves the results that the manuscript reports. It answers comments R1.1 to
R1.6 and R2.1 to R2.6 of the review of the manuscript (major revision
ENB-D-26-09048), and its design is that of the note of 19 November 2026
(Section 4 of `2026.11.19_validation_and_gsa_design`): 21 factors, 11 outputs,
1,024 runs of a scrambled Sobol' sequence, and three families of indices
(first-order Sobol' indices, Shapley effects and optimal-transport indices)
computed from the same runs. This document describes the code in `gsa/`.

The model of one run repeats the pipeline from the activity codes onwards:
the household power of every household-day and slot (Stage 03), with the
run's appliance powers and rules, the mean profiles by country and day type
and the comparison of the day types (Stage 06), the global solution (Stage
05) and the 47 stratified ones (Stage 06), the tests of the background
variables (Stage 07) and the peak times of the global groups (Stage 08), under
the publication rules of `pipeline/disclosure.py`. At the values of the
canonical run it gives the same partitions and the same statistics as the
pipeline, which `gsa.check` verifies on the run the inputs come from.

## 1. Running it

The commands below run on the machine that holds the microdata, from the root
of the repository, after a complete canonical run of the pipeline (version
1.1.0 with the changes of Section 3.17 of `docs/PIPELINE.md`, and
`HETUS_CLUSTER_ON_SOCIO` unset). The folder `~/hetus_gsa` is an example.

```bash
# 1. Inputs of the model, built from the processed files and the raw files
#    (a few minutes). The bundle holds microdata: it never leaves the machine.
python3 -m gsa.bundle --out ~/hetus_gsa/bundle

# 2. The model at the values of the canonical run against the files of the run
#    (2 minutes on an eight-core laptop); the report goes to ~/hetus_gsa/share/
python3 -m gsa.check --bundle ~/hetus_gsa/bundle --threads 8

# 3. Pilot: the first runs with two settings, which print the time per run
python3 -m gsa.run --bundle ~/hetus_gsa/bundle --runs ~/hetus_gsa/runs --first 4 --workers 1 --threads 8
python3 -m gsa.run --bundle ~/hetus_gsa/bundle --runs ~/hetus_gsa/runs --first 12 --workers 4 --threads 2

# 4. The 1,024 runs with the faster setting (an interrupted run resumes), and
#    the 256 runs of the check of the distributions of the powers
python3 -m gsa.run --bundle ~/hetus_gsa/bundle --runs ~/hetus_gsa/runs --workers 4 --threads 2
python3 -m gsa.run --bundle ~/hetus_gsa/bundle --runs ~/hetus_gsa/runs_uniform --m 8 \
    --uniform-in-watts --workers 4 --threads 2

# 5. The files that can be shared, in ~/hetus_gsa/share/ and share_uniform/
python3 -m gsa.collect --runs ~/hetus_gsa/runs
python3 -m gsa.collect --runs ~/hetus_gsa/runs_uniform --out ~/hetus_gsa/share_uniform

# 6. The seeds: the first 256 points again with both seeds redrawn, and with
#    the K-means seed alone redrawn, after three points rerun unchanged (the
#    check that the model gives the same outputs); then their shares
python3 -m gsa.reseed --bundle ~/hetus_gsa/bundle --base ~/hetus_gsa/runs --out ~/hetus_gsa/reseed \
    --which same --list 0,1,2409 --workers 3 --threads 2
python3 -m gsa.reseed --bundle ~/hetus_gsa/bundle --base ~/hetus_gsa/runs --out ~/hetus_gsa/reseed \
    --which seeds --points 256 --workers 4 --threads 2
python3 -m gsa.reseed --bundle ~/hetus_gsa/bundle --base ~/hetus_gsa/runs --out ~/hetus_gsa/reseed \
    --which kmeans --points 256 --workers 4 --threads 2
for v in same seeds kmeans; do
    python3 -m gsa.collect --runs ~/hetus_gsa/reseed/$v --out ~/hetus_gsa/reseed_share/$v
done
```

`gsa.bundle` takes its locations from `pipeline/paths.py`, so the same
`HETUS_*` settings as the run apply. It stops if its power differs from that
of Stage 03 in any household-day and slot, or if the household-days, the
order of `prepared_clustering_data.csv` or the diarist records of the raw
files do not match. `gsa.check` exits with status 1 if any comparison fails.
Run i of `gsa.run` is always point i of the same sequence, so the pilot runs
are the first runs of the sample, and the runs can be made in any number of
sittings; a folder holds the runs of one set of settings (number of K-means
starts, distribution of the powers, bundle, and the versions of Python, NumPy,
SciPy and scikit-learn, since other versions can give other partitions), and
`gsa.run` refuses to add runs made with other settings or in another
environment. A folder begun before the versions were kept takes those of the
next call, so that call must be made in the environment of the earlier runs. A run that fails leaves its traceback in
`run_iiiii.error` and is made again by the next call.
`gsa.reseed` reruns points of the sample with the seed of the country
multipliers (X8) and the seed of K-means (X21) redrawn (variant `seeds`), the
K-means seed alone (`kmeans`), or nothing (`same`), each point from a stream
of its own, so that its new seeds do not depend on which other points are run.
It takes the settings of the base folder and refuses another bundle or
another environment, and its files are those of `gsa.run`, under one subfolder
per variant, so that an interrupted variant resumes and `gsa.collect` gathers
it. The total index of the seeds follows from the pairs of runs by Jansen's
estimator (Section 8).

## 2. The inputs (gsa/bundle.py)

The bundle is a folder of uncompressed `.npy` arrays, which every worker of
`gsa.run` memory-maps, so that one copy serves all of them:

| Folder | Contents |
| --- | --- |
| `entries/` | One element per household-day, slot and activity code in use at home, for the 28 codes with a power: the number of diarists at home with the code as main activity (`cm`) and as secondary activity only (`cs`, with another main activity), and whether the first-listed diarist has it as main (`fm`) or secondary activity only (`fs`) |
| `slots/` | For every household-day and slot: whether a diarist at home, or the first-listed diarist at home, is in an activity, main or secondary, with `Light_use` = 1, and the number of slots since laundry (331) was last in use, 1 to 12, with whether laundry was a main activity in that last slot |
| `households/` | Country, the day type by `DDV7` then `DDV1` (the rule of Stage 06) and by `DDV1` only, the survey weights (`WGHT1` of the first-listed diarist and its mean over the household-day's diarists, each scaled to mean 1 within its country, 1 where missing), the row of the darkness table, `HHQ6c`, `HHQ6h` and `HHQ6o`, the 16 background variables as in `prepared_clustering_data.csv`, the clusters of the canonical run (for Y8) and the number of diarists |
| `darkness/` | For every country, year and month of the diaries, the share of the days of the month on which each slot is dark at the capital |
| `meta.json` | Sizes, codes, feature names, the result of the power check and the sources |

Household-days are in the order of `household_interval_timeseries.csv`, and
the first-listed diarist is the first record of the household-day, as in
Stage 03. The weights, months, years and ownership items are read from the raw
files, matched on `HID`, `PID` and `DIARY` as Stage 01 builds them. Darkness
(`gsa/sun.py`) uses sunrise and sunset at the capital from the NOAA
approximation of the solar position, within two minutes of published times
(`test_gsa_sun.py`), with the country's standard time and the European
summer-time rule; the day of the month is not in the scientific use files, so
each slot is weighted by the share of the days of the diary month on which it
falls in darkness, and the slots after midnight are compared with the sunrise
of the same day.

## 3. The model (gsa/model.py)

The power in slot t of household-day h, in W, is

```
p = m_country * ( sum over the codes c in use at home of own_c * w_c * q_c
                  + own_331 * w_331 * r * v )
    + X19 * darkness(country, year, month, t) * [a diarist at home is in an activity that needs light]
    + X12
```

with q_c given by the shared-appliance rule (X9), the weight of codes recorded
only as secondary activities (X11) and the diarists covered (X13):

| Rule | q_c |
| --- | --- |
| union (pipeline) | 1 if a diarist at home has c as main activity, X11 otherwise |
| per person | (diarists with c as main activity) + X11 (diarists with c as secondary activity only) |
| television once | per person, with television (821) counted as in the union, since the files only tell whether a household has a set |
| first-listed diarist (X13) | 1, X11 or 0 for the first-listed diarist alone, whatever X9 |

The run-on of laundry (X20, in minutes) adds r = clip(X20 / 10 - (k - 1), 0, 1)
in the k-th slot after laundry stops, k = 1 to 12, with v = 1 if laundry was
a main activity in its last slot and X11 if it was only a secondary one. The
country multipliers are m = exp(X7 z), z drawn from a standard normal with
seed X8 and centred over the countries of the bundle, so that they change the
differences between countries and not the overall level. With ownership
applied (X10), own_c = 0 where the household lacks the appliance of code c:
`HHQ6c` = 0 (no television set) for 821, `HHQ6h` = 2 (no washing machine) for
331 and its run-on, and `HHQ6o` = 2 (no computer) for 721 to 729 and 733;
households not asked are taken to own it. At the values of the canonical run
every q_c is 1, every own_c and m is 1 and X12, X19 and X20 are 0, so p is the
power of Stage 03 exactly.

The weights of X15 enter every mean and share, the contingency tables of the
tests (scaled to the number of observations) and K-means (`sample_weight`),
while standardisation and PCA are unweighted, as in the pipeline. Every
solution is standardised, reduced by PCA to the share X18 of the variance and
clustered by K-means with K = X16 (global) or X17 (stratified, at most a tenth
of the stratum as in Stage 06), seed X21 and 20 starts, as in the final fits
of the pipeline (`--n-init`). The base load shifts every slot equally and
moves no peak, so peak times are taken from the power without it, and a
household-day without other demand has no peak, as in the pipeline.

## 4. Factors and outputs

| ID | Factor | Range or levels | Canonical run |
| --- | --- | --- | --- |
| X1 | Cooking (311) | 400 to 1,200 W | 810 W |
| X2 | Laundry (331); ironing (332) | 250 to 800 W; 1,000 to 2,800 W | 530 W; 2,800 W |
| X3 | Cleaning (321) | 300 to 1,000 W | 620 W |
| X4 | Television (821); radio (831) | 50 to 200 W; 5 to 30 W | 71 W; 5 W |
| X5 | Computing (721 to 729) and games (733) | 40 to 250 W | 201 W; 210 W |
| X6 | Household management (371); the 16 other activities at 5 W | 20 to 150 W; 0 to 20 W | 70 W; 5 W |
| X7, X8 | Country differences in appliance power | SD of the log multipliers 0 to 0.3, and their seed | 0 |
| X9 | Shared-appliance rule | union; per person; television once | union |
| X10 | Ownership | ignored; applied | ignored |
| X11 | Weight of codes recorded only as secondary | 0 to 1 | 1 |
| X12 | Base load | 0 to 100 W | 0 |
| X13 | Diarist coverage | all; first-listed | all |
| X14 | Day type | `DDV7`, then `DDV1`; `DDV1` only | `DDV7`, then `DDV1` |
| X15 | Survey weight | none; `WGHT1` of the first-listed diarist; mean `WGHT1` | none |
| X16 | K of the global solution | 6 to 12 | knee (8 in the run of 20 November 2026; 9 before the same-day rule) |
| X17 | K of the stratified solutions | 3 to 8 | knee, or 4 |
| X18 | Share of variance kept by PCA | 0.80 to 0.99 | 0.95 |
| X19 | Lighting | 0 to 150 W per household while lit in darkness | 0 |
| X20 | Laundry run-on | 0 to 120 minutes | 0 |
| X21 | Seed of K-means | integer | 42 |

Powers are log-uniform on their ranges, except where a range starts at zero,
and within a factor every code sits at the same quantile of its own range.
`gsa/design.py` holds the factors, the sample (scrambled Sobol', seed
20261119, with one coordinate per factor and five matched dummies) and the
Shapley players (the appliance powers X1 to X6 and the country factors X7 and
X8 as two groups).

| ID | Output |
| --- | --- |
| Y1 | Share of household-days in the global group with the lowest mean daily energy |
| Y2 | Share of household-days in global groups whose circular SD of peak time is 2 h or less; Y2b, the mean circular SD weighted by household-days, has no threshold |
| Y3 | Median over the countries of the change in the coincident peak (maximum of the mean profile) from working to non-working days, in % |
| Y4 | Largest absolute change over the countries in the mean own daily peak between the day types, in % |
| Y5 | Largest bias-corrected Cramér's V (Bergsma 2013) of the global solution, country aside, and its variable |
| Y6 | Number of the 48 solutions in which the number of adults (`HHC5`) has the largest Cramér's V, country aside |
| Y7 | Ratio of the mean daily energy in the single-diarist countries (Hungary, the Netherlands, Norway) to that in the others |
| Y8 | Adjusted Rand index between the run's global partition and that of the canonical run |
| Y9 | Correlation of the hourly shape of Belgium, all days, with the Synergrid S21 profile |
| Y10 | Correlation of the hourly shape of France, all days, with the Enedis RES1 profile |
| Y11 | Ratio of the evening to the midday maximum of the hourly shape of France, all days |

The global groups are those under which the pipeline publishes the clusters
(every cluster of fewer than 50 household-days with the largest), the tests
are those Stage 07 publishes (at least 50 valid answers and item non-response
of at most 50%), and the shapes of Y9 to Y11 are those of the validation of 19
November 2026 (`gsa/benchmarks.py`: hourly means, minimum removed, sum one),
over the household-days whose day type is known, as the profiles of Stage 06.

## 5. What can be shared

The bundle and the run folders stay on the machine that holds the microdata.
`gsa.collect` writes the files that can be shared: `gsa_outputs.csv` (one row
per run: coordinates, factor values, outputs and diagnostics),
`gsa_profiles.npz` (the mean profiles of every run by country and day type)
and `manifest.txt` (the settings, the runs present, and the range and the
missing values of every output). Every value in them is an aggregate over at
least 50 household-days: shares and means of the published groups, test
statistics of published tests, and mean profiles of countries and day types,
which have several hundred household-days or more each. The numbers of household-days
stay in the run files. The report of `gsa.check` holds differences between the
model and the pipeline, counts of matching solutions and the outputs at the
canonical values, so it can be shared too.

## 6. Checks

`gsa.check` runs the model once at the canonical values, with K chosen by the
knee in every solution as in Stages 05 and 06 and 20 starts, and compares it
with the files of the run: the labels and K of all 48 solutions, every
Cramér's V of `consolidated_chi2_stats.csv` and which tests are published,
the changes between the day types of `peak_power_daytype_comparison.csv`, the
profiles of `mean_profiles_country_daytype.csv` and the circular SD of the
global groups in `consolidated_circular_stats.csv`. On the synthetic run of
the tests every label is the same and the statistics agree to 1e-15.

`tests/test_gsa_model.py` runs Stages 01 to 08 on synthetic raw files (three
countries, households of one to three diarists, weights, months, ownership
items) and checks the bundle (built in one block or many, the same arrays),
the comparison above, each rule of the power against a computation of its own
from the diarist records (every diarist counted, against the sum of the
diarists' series of Stage 01; the first-listed diarist, against its series;
television once; secondary codes left out; ownership; run-on; lighting; base
load; multipliers), the published groups against `disclosure.py`, the peak
times against `peaks.py`, the tests against `scipy`, the weights, and the
runner (resume, refusal of other settings) and the collected files.
`tests/test_gsa_design.py` and `tests/test_gsa_estimators.py` cover the sample
and the estimators.

## 7. Cost

On the microdata (191,740 household-days, an eight-core laptop of 2026),
`gsa.check` took 2 minutes and a pilot with 10 K-means starts took 7 seconds
per run with four workers of two threads (24 seconds in each worker) and 13
seconds with one worker of eight threads, so 1,024 runs should take about 2
hours with 10 starts and about twice as long with the default 20. On a random
bundle of the same size a run needed about 1.5 GB of memory, K-means taking
nine tenths of the time; the memory of each worker sets how many can run
together. Fewer starts make the partitions depend more on the seed X21,
which the analysis then shows.

## 8. The analysis

`gsa/analyse.py` computes everything from the shared files alone
(`gsa_outputs.csv`, `gsa_profiles.npz`), so it runs anywhere:

```bash
python3 -m gsa.analyse --share ~/hetus_gsa/share --uniform ~/hetus_gsa/share_uniform \
    --reseed ~/hetus_gsa/reseed_share --out ~/hetus_gsa/results
python3 -m gsa.figures --results ~/hetus_gsa/results --out ~/hetus_gsa/figures \
    --share ~/hetus_gsa/share
python3 -m gsa.tables --results ~/hetus_gsa/results --out ~/hetus_gsa/tables
python3 -m gsa.validation --share ~/hetus_gsa/share --canonical exports_for_sharing \
    --out ~/hetus_gsa/results
```

`--parts` runs a subset (summary, dependence, indices, knn, surrogate,
profiles, uniform, otparts, brackets, totals, seeds), so that two processes
can share the work; with one thread each (`OMP_NUM_THREADS=1`) the whole
analysis takes about two hours on two cores, most of it in brackets and
totals.

**Dependence between the columns.** The given-data estimators read the effect
of a factor from the runs in each of its classes, which is right only if each
class holds the other factors in the same shares as the whole sample. On the
first 2^m points of a Sobol' sequence this holds at the scale of the dyadic
intervals for well-behaved pairs of columns, but not for all of them: at 256
runs each of the 16 dyadic classes of X2 holds a single level of X13, and at
1,024 runs the share of X13 = first_listed in the 25 classes of X3 ranges from
0.22 to 0.78, so that X3 took on part of the effect of X13 on the outputs that
X13 drives (first-order index 0.07 to 0.09, OT index up to 0.23, for an effect
that the surrogate puts at zero). Scrambling keeps this structure, which is a
property of the generator matrices. The dependence fades as the sample grows:
five ordered pairs lie above 0.1 at 1,024 runs, one at 2,048 (where the
classes of X3 hold between 0.49 and 0.51 of first-listed coverage) and none
above 0.05 at 4,096 (largest 0.024, against 0.006 for independent columns).
At 2,048 and 4,096 runs the raw indices agree with the adjusted ones of 1,024
runs. `design_dependence.csv` gives the index of every column on the classes
of every other at each size.

**First-order and OT indices.** Both are computed from the runs by classes:
one class per level of a discrete factor and min(25, floor(sqrt n)) quantile
classes of a continuous one, one fewer when that is a power of two, so that
the classes never coincide with the dyadic intervals; the OT index uses the
exact one-dimensional W2. With `--adjusted`, the runs within each class of a
factor are reweighted by raking (iterative proportional fitting) so that the
class holds the classes of the three factors with the largest raw first-order
index (levels, or four quantile classes) in the shares of the whole sample,
the dummies alike, and `indices.csv` gives both the adjusted and the raw
estimates. At 4,096 runs the adjustment moves no first-order index by more
than 0.006 and no OT index by more than 0.019, so the plain estimators are the
default and the ones reported.

**Shapley effects.** A gradient-boosting surrogate of each output on the
factors other than the seeds and on the continuous dummy is fitted to the
runs (cross-validated R2 reported), and its Shapley effects are computed by
the random-permutation estimator of Song, Nelson and Staum (2016) on
independent points, where the dependence of the sample plays no part; they
are given as shares of Var Y, so that they add up to the share the surrogate
reproduces. Intervals come from 50 bootstrap refits of the surrogate at the
largest size. The surrogate also gives first-order and OT indices on 16,384
independent points, a check of the class-based estimates. The nearest-neighbour
estimator of Broto, Bachoc and Depecker (2020), screening with the dummy and
recomputed on the retained players, is kept as a check only: at 2,048 runs its
errors reach 0.27 (an effect of 1.13 for X13 on Y6, the appliance group
understated by up to 0.17), since seven of the fifteen players (the dummy included) are discrete
and their ties leave the choice of neighbours to the order of the runs.

**Total indices.** Part `totals` fits the same surrogate on every nested
size and computes the first-order and total indices of each factor (the six
powers one by one and as a group, closed indices) by the pick-freeze
estimators of Saltelli et al. (2010) and Jansen (1999) on 2^15 pairs of
independent points, the same for every size, as shares of Var Y of the runs.
At the largest size, 50 bootstrap refits of the surrogate (runs resampled with
replacement, 2^14 pairs each) give the intervals and the screening against the
continuous dummy, and `convergence_totals.csv` holds the statistics of
Sarrazin et al. (2016) for the total indices with the largest change of a
total index from the size before (`totals.csv`; about 35 minutes on two
cores). The total index of a factor on the surrogate holds every term of the
factor without a seed; its interactions with the seeds are counted with the
seeds (below).

**Agreement with the measured profiles and bands.** `gsa.validation` reads
the mean profiles of every run from the share and those of the canonical run
from its exports, and writes `validation_metrics.csv` (for Belgium and France:
correlation of the hourly shapes, circular error in the hour of the daily and
of the evening maximum, normalised RMSE of the hourly profile scaled to a
daily sum of one, for the canonical run and over the runs) and
`country_bands.csv` (daily energy and coincident peak per household of every
country and day type: canonical, median, 5th and 95th percentile). Run it
before `gsa.tables`, which turns them into `tab_validation.tex` and
`tab_country_bands.tex`.

**Screening and convergence.** An index reads as non-zero when the 2.5th
percentile of its bootstrap resamples lies above the 95th percentile of the
resamples of its matched dummy and above zero. The reported intervals of the
class-based indices are the percentile intervals shifted by the bootstrap
bias. Convergence is judged by the statistics of Sarrazin, Pianosi and
Wagener (2016) on the nested designs (`convergence_indices.csv`,
`convergence_knn.csv`; criteria in `gsa/convergence.py`) and by the change of
the surrogate's Shapley effects between the two largest sizes. The stability
table of `gsa/tables.py` follows the screening at 0.05 and the ranking over
the last two doublings: the indices that cross 0.05 either way, and the
outputs whose order above 0.05, top two and top five are kept. At 4,096 runs
the first-order indices meet all four criteria for every output, and the OT
indices miss only the change criterion for Y9 and Y10, through the index of
the seed X8, which is mostly the bias of the estimator (0.03, 0.02 and 0.01 at
1,024, 2,048 and 4,096 runs).

**Uncertainty analysis and relations between the indices.** `summary.csv`
describes every output over the runs: percentiles, moments, the bimodality
coefficient of Pfister et al. (2013) and the share of the runs at the most
frequent value, which shows the zero spike of Y2. `ot_parts.csv` splits the
OT index of every factor by the decomposition of the squared Wasserstein
distance into mean, spread and shape (Irpino and Verde 2015): the mean part is
S/2, and a factor that only shifts a Gaussian output has the OT index
(S + (1 - sqrt(1 - S))^2) / 2, so that the OT and first-order rankings can
differ only through effects on the spread and the shape. `brackets.csv` gives
the first-order index, the Shapley effect and the total index of the
surrogate on independent points for every factor one by one (the six powers
included) and for the powers as a group: first-order and total indices by
the pick-freeze estimators of Saltelli et al. (2010) and Jansen (1999) on
2^16 pairs of points, Shapley effects from ten independent runs of the
permutation estimator with 10,000 permutations each, whose spread gives the
Monte Carlo error. For independent inputs S <= Sh <= T for every player (Owen
2014), and the sum of the Shapley effects of the powers one by one is at
least the effect of their group, with equality when no interaction joins two
powers and another factor.

**The seeds and the variance budget.** The two seeds carry no order that an
estimator or a surrogate could use, so their share of the variance stays out
of the Shapley effects of the surrogate. `gsa.reseed` reruns points with the
seeds redrawn, and `gsa.analyse --reseed` (part `seeds`) gives the total
index of the seeds on the model by Jansen's estimator, T = mean((y - y')^2) /
(2 Var Y), with a bootstrap interval over the points (`seeds.csv`). A
surrogate that reproduced E[Y | other factors] exactly would reproduce
1 - T of the variance, so the variance of every output splits into the
first-order effects and interactions of the factors (surrogate), the K-means
seed and the country multipliers (reruns), and the error of the surrogate by
difference (`budget.csv`, `fig_budget.pdf`, `tab_budget.tex`). The variant
`same` checks that the model is deterministic, and the outputs that no
clustering enters (Y3, Y4, Y7, Y9, Y10, Y11) do not move when only the
K-means seed is redrawn. Which factors modulate the effect of the seeds
follows from the same reruns: (y - y')^2 / 2 estimates without bias the
variance that the seeds leave at a point, so its mean within a band of a
factor, over Var Y, is the total index of the seeds within that band
(`seed_dispersion.csv`, bands of the spread X7 and of K), and its rank
correlation with a factor, with a permutation p-value, ranks the modulators
(`seed_modulators.csv`). This is the dispersion part of the joint modelling of
stochastic models of Marrel et al. (2012). `tab_seeds.tex` gives the seeds'
total indices and the bands of X7.

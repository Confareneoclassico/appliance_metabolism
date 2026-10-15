#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh: orchestrates the full HETUS electricity-profiling
# pipeline (Stages 01-11), in the dependency order documented in
# docs/PIPELINE.md.
#
# Every stage takes its locations from pipeline/paths.py, which resolves them
# from the repository root, so the stages give the same result whichever
# directory they are run from. Raw inputs are read from HETUS_DATA_DIR
# (default data/raw/), outputs are written to processed/ and figures to
# Figures/ (see paths.py to change these).
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${REPO_ROOT}/pipeline"
RAW_DIR="${HETUS_DATA_DIR:-${REPO_ROOT}/data/raw}"
export HETUS_DATA_DIR="${RAW_DIR}"

echo "============================================================"
echo "HETUS pipeline, raw data in: ${RAW_DIR}"
echo "============================================================"

# START_STAGE=N skips the stages numbered below N, so that a re-run can
# start from existing intermediate outputs (for example START_STAGE=5 reuses
# processed/prepared_clustering_data.csv and re-runs the clustering).
START_STAGE="${START_STAGE:-1}"
PROCESSED="${HETUS_PROCESSED_DIR:-${REPO_ROOT}/processed}"

if (( START_STAGE <= 3 )) && [[ ! -f "${RAW_DIR}/HETUS_mapped_data.csv" ]]; then
  echo "ERROR: HETUS_mapped_data.csv not found in ${RAW_DIR}."
  echo "Place it there, next to the country files, before running (see README.md)."
  exit 1
fi
if (( START_STAGE == 5 )) && [[ ! -f "${PROCESSED}/prepared_clustering_data.csv" ]]; then
  echo "ERROR: ${PROCESSED}/prepared_clustering_data.csv not found."
  echo "Run Stages 01 to 04 first (START_STAGE=1)."
  exit 1
fi

run_stage () {
  local stage_script="$1"
  local stage_number=$((10#${stage_script:0:2}))
  if (( stage_number < START_STAGE )); then
    echo "Skipping ${stage_script} (START_STAGE=${START_STAGE})"
    return
  fi
  echo ""
  echo "------------------------------------------------------------"
  echo "Running: ${stage_script}"
  echo "------------------------------------------------------------"
  python3 "${PIPELINE_DIR}/${stage_script}"
}

# --- Linear chain: raw HETUS -> diarist metrics -> dominant activity ->
#     household aggregation -> clustering feature table ------------------
run_stage "01_calculate_diarist_metrics.py"
run_stage "02_calculate_dominant_activity.py"
run_stage "03_aggregate_household.py"
run_stage "04_prepare_clustering_data.py"

# --- From here, two independent branches can in principle run in
#     parallel: (a) clustering + validation, (b) diarist-coincidence /
#     household-composition diagnostics. Run sequentially here for
#     simplicity and clear console output; parallelise if needed. -------

# Branch (a): clustering and statistical validation
run_stage "05_clustering_analysis_global.py"
run_stage "06_clustering_country_day.py"
# Supplementary figure of the country cluster profiles, drawn from the
# aggregate export of Stage 06 (needs no microdata)
if (( START_STAGE <= 6 )); then
  python3 "${PIPELINE_DIR}/plot_country_cluster_profiles.py"
fi
run_stage "07_statistical_validation.py"
run_stage "08_circular_stats_peak_time.py"
run_stage "11_visualize_cluster_drivers.py"

# Branch (b): diarist-level diagnostics (depends only on Stage 02/03 output).
# These stages use activity codes and diarist counts only. Set
# SKIP_DIAGNOSTICS=1 to skip them when only sociodemographic inputs or the
# day-type rule have changed (as in the October 2026 re-run), since their
# outputs are then unchanged and Stage 09 is slow.
if [[ "${SKIP_DIAGNOSTICS:-0}" != "1" ]]; then
  run_stage "09_same_diarists_coincidence.py"
  run_stage "10_p_h_counting.py"
else
  echo ""
  echo "SKIP_DIAGNOSTICS=1: Stages 09 and 10 skipped (previous outputs remain valid)."
fi

echo ""
echo "============================================================"
echo "Pipeline complete. Key outputs (paths set in pipeline/paths.py):"
echo "  processed/clustering_results.csv, processed/cluster_summary.csv"
echo "  processed/consolidated_anova_stats.csv, processed/consolidated_chi2_stats.csv"
echo "  processed/consolidated_circular_stats.csv"
echo "  processed/Country_Analysis/country_comparison_summary.csv"
echo "  processed/Country_DayType_Analysis/country_daytype_combined_summary.csv"
echo "  Figures/ (elbow plot, Figure 2 and its panel version, country cluster"
echo "  profiles, driver plots)"
echo "See docs/PIPELINE.md for the full input/output contract."
echo "============================================================"

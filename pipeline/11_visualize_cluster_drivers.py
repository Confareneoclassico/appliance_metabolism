"""Stage 11: composition of the clusters by the background variables most
associated with them (diagnostic figures, none of them in the manuscript).

The variables are the TOP_N most associated with the global clusters, ranked
by Cramér's V in the Stage 07 results (consolidated_chi2_stats.csv), among the
tests that are published. Versions up to 1.0.1 used a fixed list of four
variables that no longer matched the ranking (audit of November 2026, L16:
COUNTRY, INC4_1, IND6_1 and HHQ5 were left out).

Each bar is a cluster, split by the categories of the variable, and adds up to
100%. Versions up to 1.0.1 transposed the table before plotting, so the bars
were categories on an axis labelled as clusters, split by cluster, and did not
add up to 100% (audit of November 2026, D8). Clusters of fewer than
MIN_N_UNFLAGGED household-days are shown together with the largest cluster
of their solution, as in every export (disclosure.py).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import PROCESSED_DIR, DAYTYPE_DIR, COUNTRY_DIR, DRIVER_FIGURES_DIR
from disclosure import published_groups

# --- Configuration ---
# Number of variables drawn, and the variables used when the Stage 07 results
# are not available (the four of versions up to 1.0.1)
TOP_N = int(os.environ.get("HETUS_DRIVER_TOP_N", "6"))
FALLBACK_FEATURES = ["HHC5", "HHC1", "INC2", "DDV6"]
CHI2_FILE = PROCESSED_DIR / "consolidated_chi2_stats.csv"
RANKING_SOLUTION = "clustering_results.csv"   # the global solution of Stage 05

# Files representing granular clustering solutions
# Country files are named by the two-letter country code (Stage 06)
FILES_TO_PROCESS = {
    "Working_Days": DAYTYPE_DIR / "Working_Days_clustering_results.csv",
    "Non_Working_Days": DAYTYPE_DIR / "Non_Working_Days_clustering_results.csv",
    "PL": COUNTRY_DIR / "PL_clustering_results.csv"
}

OUTPUT_DIR = DRIVER_FIGURES_DIR
# --- End Configuration ---

# Plot labels of the HETUS variables
FEATURE_LABELS = {
    "COUNTRY": "Country",
    "DDV1": "Day of the Week",
    "DDV3": "Quarter of the Diary Day",
    "DDV6": "Worked or Studied",
    "HHC1": "Household Size",
    "HHC3": "Number of Infants",
    "HHC4": "Number of Children",
    "HHC5": "Number of Persons Aged 18+",
    "HHQ5": "Tenure Status",
    "HHQ9_1": "Household Income",
    "INC1": "Sex",
    "INC2": "Age Group",
    "INC4_1": "Main Activity Status",
    "IND6_1": "Job Type",
    "IND22_1": "Educational Attainment",
    "IND44": "Contract Type",
}


def top_features(path=CHI2_FILE, solution=RANKING_SOLUTION, top_n=TOP_N):
    """The `top_n` variables with the largest Cramér's V in the published
    tests of `solution`, or FALLBACK_FEATURES if the results are missing."""
    try:
        tests = pd.read_csv(path)
    except FileNotFoundError:
        print(f"{path} not found: using {FALLBACK_FEATURES}")
        return list(FALLBACK_FEATURES)
    tests = tests[(tests["file"] == solution) & tests["Cramers_V"].notna()]
    if "Publication_Class" in tests:
        tests = tests[tests["Publication_Class"].fillna("") != "withheld"]
    if tests.empty:
        print(f"No published test for {solution} in {path}: using {FALLBACK_FEATURES}")
        return list(FALLBACK_FEATURES)
    ranked = tests.sort_values("Cramers_V", ascending=False)["Feature"].tolist()
    return ranked[:top_n]


def cluster_composition(df, feature):
    """Percentage of each category of `feature` within each published group
    of clusters (rows add up to 100), leaving out missing answers (-99) and
    household-days to which the item does not apply (-98)."""
    groups, small, largest = published_groups(df["cluster"])
    df = pd.DataFrame({"group": groups, feature: df[feature].to_numpy()})
    valid = df[~df[feature].isin([-99, -98]) & df[feature].notna()]
    if valid.empty:
        return None, small, largest
    table = pd.crosstab(valid["group"], valid[feature])
    return table.div(table.sum(axis=1), axis=0) * 100, small, largest


def visualize_driver_distribution(file_path, analysis_name, feature):
    """Plot the composition of every cluster of one solution by one feature."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping visualization for {analysis_name} - {feature}.")
        return

    if feature not in df.columns or 'cluster' not in df.columns:
        print(f"Missing required columns in {file_path}. Skipping.")
        return

    distribution_pct, small, largest = cluster_composition(df, feature)
    if distribution_pct is None:
        print(f"No valid data for feature {feature} in {analysis_name}. Skipping.")
        return
    if distribution_pct.shape[1] < 2:
        print(f"{feature} takes a single value in {analysis_name}. Skipping.")
        return
    if small:
        distribution_pct.index = [f"{g} (with {len(small)} small)" if g == largest else str(g)
                                  for g in distribution_pct.index]

    # Clusters on the x axis, categories stacked (no transpose, D8)
    fig, ax = plt.subplots(figsize=(12, 7))
    distribution_pct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title(f'Distribution of {FEATURE_LABELS.get(feature, feature)} Across Clusters\n'
                 f'({analysis_name} Analysis)', fontsize=14)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Percentage of Cluster Household-Days (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title=feature, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{analysis_name}_{feature}_distribution.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    features = top_features()
    print("Starting visualization of cluster driver distributions...")
    print(f"Variables (largest Cramér's V in the global solution first): {features}")

    for analysis_name, file_path in FILES_TO_PROCESS.items():
        for feature in features:
            visualize_driver_distribution(file_path, analysis_name, feature)

    print(f"\nVisualization complete. Plots are saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

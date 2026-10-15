import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import warnings
import os
from paths import COUNTRY_DIR, DAYTYPE_DIR, COUNTRY_DAYTYPE_DIR, PROCESSED_DIR
from disclosure import protect, round_size, MIN_N_PUBLISH, MIN_N_UNFLAGGED
from scipy import stats
warnings.filterwarnings("ignore")

# Household_Energy_W_* columns hold mean power (W) in each 10-minute
# interval. Daily energy in Wh is the sum over the 144 intervals multiplied
# by the interval length in hours (10/60). Summing the W values alone
# overstates daily energy six-fold.
INTERVAL_HOURS = 10 / 60

# HETUS diary days run from 04:00 to 03:59 on the next calendar day, and slot
# 0 of every energy series covers 04:00-04:09 (SUF variable Mact001). Positions
# along a series are therefore hours since 04:00, and clock time is
# 04:00 + slot x 10 min (mod 24 h). Earlier versions labelled slot/6 directly
# as the hour of day, which put every reported time four hours early.
DIARY_START_HOUR = 4


def slot_to_clock_hour(slot):
    """Clock hour (0-24) of a 10-minute slot index (may be fractional)."""
    return (DIARY_START_HOUR + np.asarray(slot, dtype=float) / 6) % 24


def slot_to_clock_str(slot):
    """HH:MM clock time of a 10-minute slot index (may be fractional)."""
    total_min = int(round(float(slot_to_clock_hour(slot)) * 60)) % (24 * 60)
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


def set_clock_axis(ax, step=2):
    """x axis in hours since 04:00 (diary order), labelled with clock time."""
    ticks = list(range(0, 25, step))
    ax.set_xlim(0, 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{(DIARY_START_HOUR + t) % 24:02d}:00" for t in ticks])


def clock_span(ax, start_hour, end_hour, **kwargs):
    """Shade a clock-time interval on a diary-order axis, splitting it when it
    crosses 04:00. Only the first piece carries the legend label."""
    a = (start_hour - DIARY_START_HOUR) % 24
    b = (end_hour - DIARY_START_HOUR) % 24 or 24
    if a < b:
        ax.axvspan(a, b, **kwargs)
    else:
        ax.axvspan(a, 24, **kwargs)
        kwargs.pop("label", None)
        ax.axvspan(0, b, **kwargs)

# --- Configuration of the clustering (October 2026 re-run) ------------------
# The clusters are formed on the 144 standardised power slots only. The 16
# background variables are kept in the results files to describe and test the
# clusters (Stage 07), but no longer enter the clustering, so the tests are not
# run on variables that helped to form the clusters. HETUS_CLUSTER_ON_SOCIO=1
# restores the earlier joint feature space, for comparison.
CLUSTER_ON_SOCIO = os.environ.get("HETUS_CLUSTER_ON_SOCIO", "0") == "1"
# Share of variance kept by PCA, the same as in the global solution (Stage 05).
PCA_VARIANCE = 0.95
# The profile exports follow Eurostat's publication rules for HETUS (see
# disclosure.py): groups of fewer than 20 household-days are listed without
# size or statistics, groups of 20 to 49 are flagged and listed without their
# size, and the other sizes are rounded where a group of the same solution is
# withheld or flagged.
PROFILE_STAT_COLS = ['avg_daily_energy_wh', 'avg_peak_power_w',
                     'mean_profile_peak_w', 'mean_profile_peak_time']
# The first pass of this stage clustered every country on all its data under
# the name XX_ALL_DATA, which repeats the country solutions of the second pass
# exactly. It is skipped unless HETUS_RUN_ALL_DATA_PASS=1.
RUN_ALL_DATA_PASS = os.environ.get("HETUS_RUN_ALL_DATA_PASS", "0") == "1"

# Colours shared with Stage 05 (fixed categorical order, checked for colour
# vision deficiency): working days in slot 1, non-working days in slot 2.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
WORK_COLOUR, NONWORK_COLOUR = SERIES_COLOURS[0], SERIES_COLOURS[1]
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# Rows of cluster mean profiles, written to one file at the end of the stage.
PROFILE_EXPORT = []

print(f"Clustering features: "
      f"{'power slots and one-hot background variables' if CLUSTER_ON_SOCIO else 'power slots only'}; "
      f"PCA keeps {PCA_VARIANCE:.0%} of the variance")

# Output folders (COUNTRY_DIR, DAYTYPE_DIR, COUNTRY_DAYTYPE_DIR) are created by
# paths.py when it is imported.

# Load the prepared clustering data
print("Loading data...")
clustering_data = pd.read_csv(PROCESSED_DIR / "prepared_clustering_data.csv")

# ============================================================================
# COUNTRY LABELS
# ============================================================================

country_map = {
    0: "BE",
    1: "DE",
    2: "EE",
    3: "EL",
    4: "ES",
    5: "FI",
    6: "FR",
    7: "HU",
    8: "IT",
    9: "LU",
    10: "NL",
    11: "NO",
    12: "PL",
    13: "RO",
    14: "RS",
    15: "UK"
}

# Preserve original numeric code if needed later
clustering_data["COUNTRY_CODE"] = clustering_data["COUNTRY"]

# Replace codes with short country names
clustering_data["COUNTRY"] = clustering_data["COUNTRY"].map(country_map)


# Extract energy time series columns and categorical features
energy_columns = [col for col in clustering_data.columns if col.startswith("Household_Energy_W_")]
categorical_features = [
    "DDV1", "DDV3", "DDV6", "COUNTRY", "INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
    "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"
]

print(f"Total households: {len(clustering_data)}")
print(f"Energy time series features: {len(energy_columns)}")
print(f"Categorical features: {len(categorical_features)}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def determine_day_type(row):
    """
    Determine if a day is working or non-working based on DDV7 and DDV1.
    
    Rules:
    1. If DDV7 is 1 or 2 → Working day
    2. If DDV7 is 3, 4, 5, or 6 → Non-working day
    3. If DDV7 is missing or negative (-1 to -9, or -99 after Stage 04) → Use DDV1:
       - DDV1 = 2-6 → Working day
       - DDV1 = 1 or 7 → Non-working day
    4. If still undetermined → None (exclude from analysis)
    """
    ddv7 = row.get('DDV7', -9)  # Default to -9 if not present
    ddv1 = row.get('DDV1', -9)
    
    # Check DDV7 first
    if ddv7 in [1, 2]:
        return "working"
    elif ddv7 in [3, 4, 5, 6]:
        return "non_working"
    elif pd.isna(ddv7) or ddv7 < 0:  # Missing or negative code: fall back to DDV1
        if ddv1 in [2, 3, 4, 5, 6]:
            return "working"
        elif ddv1 in [1, 7]:
            return "non_working"
        else:
            return None  # Cannot determine
    else:
        return None  # Cannot determine

def perform_clustering_analysis(data, analysis_name, output_dir, energy_cols, cat_features):
    """
    Perform clustering analysis on a subset of data.
    
    Args:
        data: DataFrame with household data
        analysis_name: Name for this analysis (for output files)
        output_dir: Directory to save outputs
        energy_cols: List of energy time series columns
        cat_features: List of categorical features
    """
    print(f"\n{'='*80}")
    print(f"PERFORMING CLUSTERING ANALYSIS: {analysis_name}")
    print(f"{'='*80}")
    
    if len(data) < 10:
        print(f"  Not enough data for clustering (only {len(data)} households)")
        return None
    
    print(f"  Number of households: {len(data)}")
    
    # Prepare preprocessing: power slots only, unless CLUSTER_ON_SOCIO is set
    transformers = [("energy", StandardScaler(), energy_cols)]
    if CLUSTER_ON_SOCIO:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             [f for f in cat_features if f in data.columns]))
    preprocessor = ColumnTransformer(transformers=transformers)
    
    try:
        # Transform data
        X_processed = preprocessor.fit_transform(data)
        
        # Apply PCA, keeping the same share of variance as the global solution.
        # (Earlier versions kept min(0.95, 144 / number of columns), that is
        # about 59 to 71% of the variance in the stratified solutions.)
        pca = PCA(n_components=PCA_VARIANCE)
        X_pca = pca.fit_transform(X_processed)
        print(f"  PCA: {X_processed.shape[1]} columns reduced to {X_pca.shape[1]} components")
        
        # Determine optimal k using elbow method
        wcss = []
        max_k = min(10, len(data)//10)  # Don't test more clusters than reasonable
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_pca)
            wcss.append(kmeans.inertia_)
        
        # Find elbow point. The log records the WCSS curve and whether the
        # knee was detected or the fallback value was used, since the two
        # cannot be told apart from the number of clusters alone.
        print("  WCSS: " + ", ".join(f"K={k}: {w:.1f}" for k, w in zip(k_range, wcss)))
        if len(wcss) > 1:
            kl = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
            optimal_k = kl.elbow if kl.elbow else min(4, max_k)
            k_note = "" if kl.elbow else " (no knee detected, fallback value)"
        else:
            optimal_k = 2
            k_note = " (only one K tested, fallback value)"
        
        print(f"  Optimal number of clusters: {optimal_k}{k_note}")
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        data_with_clusters = data.copy()
        data_with_clusters["cluster"] = kmeans.fit_predict(X_pca)
        collect_cluster_profiles(data_with_clusters, energy_cols, analysis_name)
        
        # Create energy profiles
        create_energy_profiles(data_with_clusters, energy_cols, optimal_k, 
                              analysis_name, output_dir)
        
        # Create cluster summary
        create_cluster_summary(data_with_clusters, energy_cols, cat_features, 
                              analysis_name, output_dir)
        
        # Save results
        safe_name = analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        cluster_results_path = f"{output_dir}/{safe_name}_clustering_results.csv"
        data_with_clusters.to_csv(cluster_results_path, index=False)
        print(f"  Clustering results saved to: {cluster_results_path}")
        
        return data_with_clusters
        
    except Exception as e:
        print(f"  Error in clustering: {e}")
        return None

def create_energy_profiles(data_with_clusters, energy_cols, n_clusters, 
                          analysis_name, output_dir):
    """
    Create 10-minute energy profiles for each cluster.
    """
    plt.figure(figsize=(16, 10))
    colors = [SERIES_COLOURS[i % len(SERIES_COLOURS)] for i in range(n_clusters)]
    
    for cluster_id in range(n_clusters):
        cluster_mask = data_with_clusters["cluster"] == cluster_id
        cluster_energy = data_with_clusters.loc[cluster_mask, energy_cols].mean(axis=0).values
        
        # Use all 144 (24*6) 10-minute data points
        time_points = np.arange(144) / 6  # hours since 04:00 (diary order)
        
        # Plot line
        plt.plot(time_points, cluster_energy, color=colors[cluster_id], 
                 linewidth=1.5, alpha=0.8)
        # Plot individual points
        plt.scatter(time_points, cluster_energy, color=colors[cluster_id], s=20, 
                   alpha=0.6, label=f'Cluster {cluster_id} (n={cluster_mask.sum()})')
    
    plt.title(f'Mean 10-minute power demand by cluster\n{analysis_name}',
              fontsize=18, fontweight='bold')
    plt.xlabel('Time of day', fontsize=14)
    plt.ylabel('Mean power demand (W)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    set_clock_axis(plt.gca(), 2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save plot
    safe_name = analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    plot_path = f"{output_dir}/{safe_name}_energy_profiles.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Energy profiles saved to: {plot_path}")
    
    # Also create individual cluster profiles
    create_individual_cluster_profiles(data_with_clusters, energy_cols, 
                                      n_clusters, analysis_name, output_dir)

def create_individual_cluster_profiles(data_with_clusters, energy_cols, n_clusters, 
                                      analysis_name, output_dir):
    """
    Create separate energy profiles for each cluster with statistics.
    """
    cluster_stats = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = data_with_clusters["cluster"] == cluster_id
        cluster_data = data_with_clusters.loc[cluster_mask, energy_cols]
        
        if len(cluster_data) == 0:
            continue
        
        # Calculate statistics
        mean_profile = cluster_data.mean(axis=0).values
        std_profile = cluster_data.std(axis=0).values
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        time_points = np.arange(144) / 6
        
        # Plot mean with confidence interval
        ax.plot(time_points, mean_profile, color='blue', linewidth=2, 
                label='Mean')
        ax.fill_between(time_points, mean_profile - std_profile, 
                        mean_profile + std_profile, alpha=0.3, 
                        color='blue', label='±1 Std Dev')
        
        # Calculate daily statistics
        daily_energy = cluster_data.sum(axis=1) * INTERVAL_HOURS
        peak_power = cluster_data.max(axis=1)
        
        stats_text = (f"Cluster {cluster_id} (n={len(cluster_data)})\n"
                     f"Avg Daily Energy: {daily_energy.mean():.0f} ± {daily_energy.std():.0f} Wh\n"
                     f"Avg Peak Power: {peak_power.mean():.0f} ± {peak_power.std():.0f} W\n"
                     f"Peak Time: {slot_to_clock_str(np.argmax(mean_profile))}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(f'Energy Profile - Cluster {cluster_id}\n{analysis_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time of day', fontsize=12)
        ax.set_ylabel('Energy Demand (W)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        set_clock_axis(ax, 2)
        
        plt.tight_layout()
        safe_name = analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        plot_path = f"{output_dir}/{safe_name}_cluster_{cluster_id}_profile.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store statistics
        cluster_stats.append({
            'analysis': analysis_name,
            'cluster': cluster_id,
            'n_households': len(cluster_data),
            'avg_daily_energy_wh': daily_energy.mean(),
            'std_daily_energy_wh': daily_energy.std(),
            'avg_peak_power_w': peak_power.mean(),
            'std_peak_power_w': peak_power.std(),
            # Clock time of the peak of the cluster's mean profile
            'avg_peak_hour': int(slot_to_clock_str(np.argmax(mean_profile))[:2]),
            'avg_peak_minute': int(slot_to_clock_str(np.argmax(mean_profile))[3:])
        })
    
    # Save statistics
    if cluster_stats:
        stats_df = pd.DataFrame(cluster_stats)
        safe_name = analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        stats_path = f"{output_dir}/{safe_name}_cluster_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"  Cluster statistics saved to: {stats_path}")

def create_cluster_summary(data_with_clusters, energy_cols, cat_features, 
                          analysis_name, output_dir):
    """
    Create detailed summary of cluster characteristics.
    """
    summary_data = []
    n_clusters = data_with_clusters["cluster"].nunique()
    
    for cluster_id in range(n_clusters):
        cluster_data = data_with_clusters[data_with_clusters["cluster"] == cluster_id]
        energy_data = cluster_data[energy_cols].values
        
        # Basic statistics
        summary = {
            'analysis': analysis_name,
            'cluster': cluster_id,
            'size': len(cluster_data),
            'size_percentage': len(cluster_data) / len(data_with_clusters) * 100,
            'avg_daily_energy_wh': (np.sum(energy_data, axis=1) * INTERVAL_HOURS).mean(),
            'std_daily_energy_wh': (np.sum(energy_data, axis=1) * INTERVAL_HOURS).std(),
            'avg_peak_energy_w': np.max(energy_data, axis=1).mean(),
            'std_peak_energy_w': np.max(energy_data, axis=1).std(),
            # Clock time of the linear mean of the household-day peak slots
            # (see Stage 08 for the circular mean)
            'avg_peak_hour': int(slot_to_clock_str(np.argmax(energy_data, axis=1).mean())[:2]),
            'avg_peak_minute': int(slot_to_clock_str(np.argmax(energy_data, axis=1).mean())[3:])
        }
        
        # Add mode for categorical features
        for col in [f for f in cat_features if f in cluster_data.columns]:
            if len(cluster_data[col].dropna()) > 0:
                mode_values = cluster_data[col].mode()
                if not mode_values.empty:
                    summary[f'mode_{col}'] = mode_values.iloc[0]
                    summary[f'mode_freq_{col}'] = (cluster_data[col] == mode_values.iloc[0]).mean()
        
        summary_data.append(summary)
    
    # Create and save summary
    summary_df = pd.DataFrame(summary_data)
    safe_name = analysis_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    summary_path = f"{output_dir}/{safe_name}_cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Cluster summary saved to: {summary_path}")

# ============================================================================
# MEAN PROFILES, COINCIDENT PEAKS AND EXPORTS (October 2026)
# ============================================================================

def profile_stats(energy_data):
    """Summary statistics of a set of household-day power series (rows, W).

    avg_peak_power_w is the mean of each household-day's own daily maximum
    (the non-coincident peak). mean_profile_peak_w is the maximum of the mean
    profile, the coincident peak per household (after-diversity maximum
    demand), which is what a feeder serving many such households would see.
    """
    mean_profile = energy_data.mean(axis=0)
    k = int(np.argmax(mean_profile))
    summary = {
        'n': int(energy_data.shape[0]),
        'avg_daily_energy_wh': float((energy_data.sum(axis=1) * INTERVAL_HOURS).mean()),
        'avg_peak_power_w': float(energy_data.max(axis=1).mean()),
        'mean_profile_peak_w': float(mean_profile[k]),
        'mean_profile_peak_time': slot_to_clock_str(k),
    }
    return summary, mean_profile


def profile_row(meta, summary, mean_profile):
    """One export row: metadata, statistics and the 144 slot means (W).
    Disclosure control is applied to the whole table before it is written
    (see write_profile_exports and export_daytype_mean_profiles)."""
    row = {**meta, **summary}
    for t in range(len(mean_profile)):
        row[f'P_{t:03d}'] = float(mean_profile[t])
    return row


def collect_cluster_profiles(data_with_clusters, energy_cols, analysis_name):
    """Keep the mean profile of every cluster of a stratified solution for
    stratified_cluster_mean_profiles.csv, written at the end of the stage.

    Clusters of fewer than MIN_N_PUBLISH household-days are withheld from the
    export (disclosure.py). Since the mean profile of every stratum is
    published (mean_profiles_country_daytype.csv), the profile of a withheld
    cluster could be recovered by subtracting the published clusters from it,
    so the household-days of the withheld clusters are reported together with
    the largest cluster of the solution, whose row is marked includes_withheld.
    """
    n_total = len(data_with_clusters)
    sizes = data_with_clusters["cluster"].value_counts()
    # Rank of each cluster by size within the stratum (1 = largest), kept so
    # that figures can order clusters after the sizes have been rounded
    ranks = sizes.rank(ascending=False, method="first").astype(int)
    largest = sizes.idxmax()
    withheld = sorted(sizes.index[sizes < MIN_N_PUBLISH]) if sizes[largest] >= MIN_N_PUBLISH else []
    for cluster_id in sorted(data_with_clusters["cluster"].unique()):
        members = [cluster_id] + (withheld if cluster_id == largest else [])
        energy_data = data_with_clusters.loc[
            data_with_clusters["cluster"].isin(members), energy_cols].values
        summary, mean_profile = profile_stats(energy_data)
        meta = {'stratum': analysis_name, 'stratum_n': n_total,
                'cluster': int(cluster_id), 'size_rank': int(ranks[cluster_id]),
                'includes_withheld': bool(cluster_id == largest and withheld),
                'share_pct': 100 * summary['n'] / n_total}
        PROFILE_EXPORT.append(profile_row(meta, summary, mean_profile))


def write_profile_exports():
    """Write the cluster mean profiles of all stratified solutions."""
    if PROFILE_EXPORT:
        path = PROCESSED_DIR / "stratified_cluster_mean_profiles.csv"
        protect(pd.DataFrame(PROFILE_EXPORT), n_col='n', stat_cols=PROFILE_STAT_COLS,
                group_col='stratum', share_col='share_pct',
                total_col='stratum_n').to_csv(path, index=False)
        print(f"Cluster mean profiles of the stratified solutions saved to: {path}")


def export_daytype_mean_profiles(data, energy_cols):
    """Mean profile and peak statistics for every combination of country
    and day type, for each country over both day types (day_type = 'all'),
    for each day type over all countries (country = 'ALL') and for the whole
    sample. Written to mean_profiles_country_daytype.csv, and the coincident
    peaks of the two day types are compared country by country."""
    groups = [((c, d), g) for (c, d), g in data.groupby(['COUNTRY', 'day_type'])]
    groups += [((c, 'all'), g) for c, g in data.groupby('COUNTRY')]
    groups += [(('ALL', d), g) for d, g in data.groupby('day_type')]
    groups += [(('ALL', 'all'), data)]
    rows = []
    for (country, day_type), g in groups:
        summary, mean_profile = profile_stats(g[energy_cols].values)
        rows.append(profile_row({'country': country, 'day_type': day_type},
                                summary, mean_profile))
    # Within a country, the 'all' group is the sum of the two day types, so
    # rounding is applied country by country if a group is withheld or flagged
    profiles = protect(pd.DataFrame(rows), n_col='n', stat_cols=PROFILE_STAT_COLS,
                       group_col='country')
    path = COUNTRY_DAYTYPE_DIR / "mean_profiles_country_daytype.csv"
    profiles.to_csv(path, index=False)
    print(f"  Mean profiles by country and day type saved to: {path}")

    # Console comparison of the two kinds of peak, country by country
    print("\n  Coincident peak (maximum of the mean profile) and mean daily peak, "
          "working vs non-working days:")
    print(f"  {'':4s} {'coincident W (time)':>34s} {'change':>8s}   "
          f"{'mean daily peak W':>20s} {'change':>8s}")
    higher = 0
    countries = sorted(c for c in profiles['country'].unique() if c != 'ALL')
    for c in countries + ['ALL']:
        w = profiles[(profiles['country'] == c) & (profiles['day_type'] == 'working')]
        nw = profiles[(profiles['country'] == c) & (profiles['day_type'] == 'non_working')]
        if w.empty or nw.empty:
            continue
        w, nw = w.iloc[0], nw.iloc[0]
        d_coinc = 100 * (nw['mean_profile_peak_w'] / w['mean_profile_peak_w'] - 1)
        d_indiv = 100 * (nw['avg_peak_power_w'] / w['avg_peak_power_w'] - 1)
        if c != 'ALL' and d_coinc > 0:
            higher += 1
        print(f"  {c:4s} {w['mean_profile_peak_w']:7.0f} ({w['mean_profile_peak_time']}) -> "
              f"{nw['mean_profile_peak_w']:7.0f} ({nw['mean_profile_peak_time']}) "
              f"{d_coinc:+7.1f}%   {w['avg_peak_power_w']:8.0f} -> {nw['avg_peak_power_w']:8.0f} "
              f"{d_indiv:+7.1f}%")
    print(f"  Coincident peak higher on non-working days in {higher} of {len(countries)} countries")
    return profiles


def plot_daytype_small_multiples(profiles, output_dir=COUNTRY_DAYTYPE_DIR):
    """Working-day and non-working-day mean profiles of every country, one
    panel per country. Dots mark the maximum of each mean profile, the
    coincident peak per household. Each panel has its own y scale, since the
    single-diarist countries draw much less power than the others."""
    prof = profiles[(profiles['country'] != 'ALL')
                    & profiles['day_type'].isin(['working', 'non_working'])]
    countries = sorted(prof['country'].unique())
    if not countries:
        return
    pcols = [f'P_{t:03d}' for t in range(144)]
    hours = np.arange(144) / 6  # hours since 04:00 (diary order)
    n_cols = 3
    n_rows = int(np.ceil(len(countries) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.25 * n_rows + 0.9),
                             sharex=True, squeeze=False)
    styles = (('working', WORK_COLOUR, 'Working days'),
              ('non_working', NONWORK_COLOUR, 'Non-working days'))
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        if idx >= len(countries):
            ax.set_visible(False)
            if idx - n_cols >= 0:
                axes[(idx - n_cols) // n_cols][idx % n_cols].xaxis.set_tick_params(labelbottom=True)
            continue
        country = countries[idx]
        ns = {}
        for day_type, colour, label in styles:
            r = prof[(prof['country'] == country) & (prof['day_type'] == day_type)]
            if r.empty or bool(r['withheld'].iloc[0]):
                continue
            y = r[pcols].values[0].astype(float)
            n = r['n'].iloc[0]
            # A flagged group (20 to 49) is shown by its size class only
            ns[day_type] = f"{int(n):,}" if pd.notna(n) else str(r['n_class'].iloc[0])
            # Groups of 20 to 49 household-days are flagged by a dashed line
            ax.plot(hours, y, color=colour, linewidth=1.6, label=label,
                    linestyle='--' if r['n_class'].iloc[0] else '-')
            k = int(np.nanargmax(y))
            ax.plot(hours[k], y[k], 'o', color=colour, markersize=5.5,
                    markeredgecolor='white', markeredgewidth=0.9, zorder=3)
        ax.set_title(f"{country}  (n = {ns.get('working', '0')} / "
                     f"{ns.get('non_working', '0')})", fontsize=11, color=INK, loc='left')
        ax.grid(True, axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.set_xlim(0, 24)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xticklabels([f"{(DIARY_START_HOUR + t) % 24:02d}:00" for t in range(0, 25, 6)])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=12, labelcolor=INK)
    fig.supxlabel('Time of day', fontsize=13, color=INK)
    fig.supylabel('Mean power demand (W)', fontsize=13, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = f"{output_dir}/country_daytype_profiles_small_multiples.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Small-multiple figure of working and non-working days saved to: {path}")


# ============================================================================
# COUNTRY COMPARISON SUMMARY FUNCTIONS
# ============================================================================

def create_country_comparison_summary(country_results):
    """
    Create a comprehensive comparison of all countries analyzed.
    """
    if not country_results:
        print("No country results to summarize")
        return
    
    print(f"\n{'='*80}")
    print("CREATING COUNTRY COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_data = []
    
    for country, country_df in country_results.items():
        # Calculate country-level statistics
        energy_data = country_df[energy_columns].values
        daily_energy = energy_data.sum(axis=1) * INTERVAL_HOURS
        peak_power = energy_data.max(axis=1)
        
        # Find peak hour
        avg_profile = energy_data.mean(axis=0)
        peak_clock = slot_to_clock_str(np.argmax(avg_profile))
        
        # Calculate day type distribution if available
        if "day_type" in country_df.columns:
            day_type_counts = country_df["day_type"].value_counts(normalize=True) * 100
            working_pct = day_type_counts.get("working", 0)
            non_working_pct = day_type_counts.get("non_working", 0)
        else:
            working_pct = non_working_pct = np.nan
        
        # Get number of clusters
        n_clusters = country_df["cluster"].nunique() if "cluster" in country_df.columns else 0
        
        comparison_data.append({
            'country': country,
            'n_households': len(country_df),
            'n_clusters': n_clusters,
            'avg_daily_energy_wh': daily_energy.mean(),
            'std_daily_energy_wh': daily_energy.std(),
            'avg_peak_power_w': peak_power.mean(),
            'std_peak_power_w': peak_power.std(),
            'peak_time': peak_clock,
            # Coincident peak per household: maximum of the mean profile
            'mean_profile_peak_w': avg_profile.max(),
            'working_days_pct': working_pct,
            'non_working_days_pct': non_working_pct
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by number of households
    comparison_df = comparison_df.sort_values('n_households', ascending=False)
    
    # Save comparison data
    comparison_df.to_csv(COUNTRY_DIR / "country_comparison_summary.csv", index=False)
    print(f"Country comparison summary saved to: {COUNTRY_DIR}/country_comparison_summary.csv")
    
    # Create visualization
    create_country_comparison_visualization(comparison_df)

def create_country_comparison_visualization(comparison_df):
    """
    Create visualization comparing countries.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Country Comparison Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of households per country
    axes[0, 0].bar(range(len(comparison_df)), comparison_df['n_households'], 
                   color='lightblue', edgecolor='black')
    axes[0, 0].set_xlabel('Country')
    axes[0, 0].set_ylabel('Number of Households')
    axes[0, 0].set_title('Households per Country')
    axes[0, 0].set_xticks(range(len(comparison_df)))
    axes[0, 0].set_xticklabels(comparison_df['country'], rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, count in enumerate(comparison_df['n_households']):
        axes[0, 0].text(i, count + max(comparison_df['n_households'])*0.01, 
                       str(count), ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Average daily energy
    axes[0, 1].bar(range(len(comparison_df)), comparison_df['avg_daily_energy_wh'], 
                   yerr=comparison_df['std_daily_energy_wh'], capsize=5,
                   color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Country')
    axes[0, 1].set_ylabel('Daily Energy (Wh)')
    axes[0, 1].set_title('Average Daily Energy Consumption')
    axes[0, 1].set_xticks(range(len(comparison_df)))
    axes[0, 1].set_xticklabels(comparison_df['country'], rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Peak power
    axes[1, 0].bar(range(len(comparison_df)), comparison_df['avg_peak_power_w'], 
                   yerr=comparison_df['std_peak_power_w'], capsize=5,
                   color='salmon', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Country')
    axes[1, 0].set_ylabel('Peak Power (W)')
    axes[1, 0].set_title('Average Peak Power Demand')
    axes[1, 0].set_xticks(range(len(comparison_df)))
    axes[1, 0].set_xticklabels(comparison_df['country'], rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Number of clusters
    axes[1, 1].bar(range(len(comparison_df)), comparison_df['n_clusters'], 
                   color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Country')
    axes[1, 1].set_ylabel('Number of Clusters')
    axes[1, 1].set_title('Clusters Found per Country')
    axes[1, 1].set_xticks(range(len(comparison_df)))
    axes[1, 1].set_xticklabels(comparison_df['country'], rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(COUNTRY_DIR / "country_comparison_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Country comparison visualization saved to: {COUNTRY_DIR}/country_comparison_summary.png")

# ============================================================================
# COUNTRY & DAY TYPE COMBINED ANALYSIS
# ============================================================================

def perform_country_daytype_combined_analysis(clustering_data, min_households=20):
    """
    Perform clustering analysis for each combination of country and day type.
    This is the most granular analysis: clusters specific to each (country, day_type).
    """
    print("\n" + "="*80)
    print("ANALYSIS: COUNTRY & DAY TYPE COMBINED")
    print("="*80)
    
    # Check if we have both country and day type
    if "COUNTRY" not in clustering_data.columns or "day_type" not in clustering_data.columns:
        print("  Need both COUNTRY and day_type columns for combined analysis")
        return {}
    
    # Get unique combinations
    combinations = clustering_data.groupby(['COUNTRY', 'day_type']).size().reset_index()
    combinations.columns = ['country', 'day_type', 'count']
    
    print(f"Found {len(combinations)} country-day type combinations")
    
    # Filter to combinations with sufficient data
    valid_combinations = combinations[combinations['count'] >= min_households]
    print(f"{len(valid_combinations)} combinations have at least {min_households} households")
    
    if len(valid_combinations) == 0:
        print("  No combinations have sufficient data for clustering")
        return {}
    
    combined_results = {}
    
    for _, combo in valid_combinations.iterrows():
        country = combo['country']
        day_type = combo['day_type']
        count = combo['count']
        
        # Filter data for this combination
        combo_data = clustering_data[
            (clustering_data['COUNTRY'] == country) & 
            (clustering_data['day_type'] == day_type)
        ].copy()
        
        analysis_name = f"{country}_{day_type}_days"
        print(f"\n  Analyzing: {analysis_name} ({count} households)")
        
        # Exclude country and day_type from categorical features for this analysis
        combo_cat_features = [f for f in categorical_features 
                            if f not in ['COUNTRY', 'day_type']]
        
        # Perform clustering
        result = perform_clustering_analysis(
            data=combo_data,
            analysis_name=analysis_name,
            output_dir=COUNTRY_DAYTYPE_DIR,
            energy_cols=energy_columns,
            cat_features=combo_cat_features
        )
        
        if result is not None:
            combined_results[(country, day_type)] = result
    
    print(f"\nCompleted combined analysis for {len(combined_results)} country-day type combinations")
    
    # Create comprehensive comparison
    if combined_results:
        create_combined_analysis_summary(combined_results)
    
    return combined_results

def create_combined_analysis_summary(combined_results):
    """
    Create a comprehensive summary of country-day type combined analysis.
    """
    print("\n" + "="*80)
    print("CREATING COUNTRY-DAY TYPE COMBINED ANALYSIS SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for (country, day_type), data in combined_results.items():
        # Calculate statistics for this combination
        energy_data = data[energy_columns].values
        daily_energy = energy_data.sum(axis=1) * INTERVAL_HOURS
        peak_power = energy_data.max(axis=1)
        
        # Average profile
        avg_profile = energy_data.mean(axis=0)
        peak_clock = slot_to_clock_str(np.argmax(avg_profile))
        
        # Number of clusters found
        n_clusters = data['cluster'].nunique() if 'cluster' in data.columns else 0
        
        # Cluster sizes. Where a cluster of the solution has fewer than 50
        # household-days, its size is withheld or given only as a class in the
        # cluster exports, and the other sizes are rounded (disclosure.py). The
        # largest cluster is then reported in the same way, rounded to the
        # nearest ten with its share computed from the rounded size, since its
        # exact size would let the small cluster's size be derived.
        if 'cluster' in data.columns:
            cluster_sizes = data['cluster'].value_counts()
            largest_cluster = cluster_sizes.iloc[0] if len(cluster_sizes) > 0 else 0
            largest_cluster_pct = (largest_cluster / len(data)) * 100
            if (cluster_sizes < MIN_N_UNFLAGGED).any():
                largest_cluster = int(round_size(largest_cluster))
                largest_cluster_pct = round(100 * largest_cluster / len(data))
        else:
            largest_cluster = 0
            largest_cluster_pct = 0
        
        summary_data.append({
            'country': country,
            'day_type': day_type,
            'n_households': len(data),
            'n_clusters': n_clusters,
            'largest_cluster_size': largest_cluster,
            'largest_cluster_pct': largest_cluster_pct,
            'avg_daily_energy_wh': daily_energy.mean(),
            'std_daily_energy_wh': daily_energy.std(),
            'avg_peak_power_w': peak_power.mean(),
            'std_peak_power_w': peak_power.std(),
            'peak_time': peak_clock,
            # Coincident peak per household: maximum of the mean profile
            'mean_profile_peak_w': avg_profile.max(),
            'clustering_performed': 'Yes'
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by country then day type
    summary_df = summary_df.sort_values(['country', 'day_type'])
    
    # Save summary
    summary_df.to_csv(COUNTRY_DAYTYPE_DIR / "country_daytype_combined_summary.csv", index=False)
    print(f"Combined analysis summary saved to: {COUNTRY_DAYTYPE_DIR}/country_daytype_combined_summary.csv")
    
    # Create visualizations
    create_combined_analysis_visualizations(summary_df)

def create_combined_analysis_visualizations(summary_df):
    """
    Create visualizations for country-day type combined analysis.
    """
    if len(summary_df) == 0:
        return
    
    # 1. Heatmap of number of households by country and day type
    pivot_table = summary_df.pivot_table(
        values='n_households',
        index='country',
        columns='day_type',
        fill_value=0
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Country & Day Type Combined Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Heatmap of household counts
    im1 = axes[0, 0].imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_title('Number of Households\n(Country × Day Type)', fontsize=12)
    axes[0, 0].set_xlabel('Day Type')
    axes[0, 0].set_ylabel('Country')
    axes[0, 0].set_xticks(range(len(pivot_table.columns)))
    axes[0, 0].set_xticklabels(pivot_table.columns)
    axes[0, 0].set_yticks(range(len(pivot_table.index)))
    axes[0, 0].set_yticklabels(pivot_table.index)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            axes[0, 0].text(j, i, f"{pivot_table.iloc[i, j]:.0f}",
                          ha="center", va="center", color="black", fontsize=8)
    
    # Panels 2 to 4 use side-by-side bars with one fixed colour per day type,
    # so that working days are blue and non-working days orange in every
    # panel. (Earlier versions overlaid the bars of the two day types in
    # panels 2 and 3, with the colours reversed relative to panel 4.)
    day_type_style = [('working', 'Working Days', WORK_COLOUR),
                      ('non_working', 'Non-Working Days', NONWORK_COLOUR)]
    width = 0.35

    def paired_bars(ax, value_col):
        pivot = summary_df.pivot_table(values=value_col, index='country',
                                       columns='day_type')
        x = np.arange(len(pivot.index))
        for i, (day_type, label, colour) in enumerate(day_type_style):
            if day_type in pivot.columns:
                offset = (i - 0.5) * width
                ax.bar(x + offset, pivot[day_type], width, label=label,
                       color=colour, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45)

    # Plot 2: Daily energy comparison
    paired_bars(axes[0, 1], 'avg_daily_energy_wh')
    axes[0, 1].set_title('Average Daily Energy Consumption', fontsize=12)
    axes[0, 1].set_xlabel('Country')
    axes[0, 1].set_ylabel('Daily Energy (Wh)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Peak power comparison
    paired_bars(axes[1, 0], 'avg_peak_power_w')
    axes[1, 0].set_title('Average Peak Power Demand', fontsize=12)
    axes[1, 0].set_xlabel('Country')
    axes[1, 0].set_ylabel('Peak Power (W)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Number of clusters found
    pivot_clusters = summary_df.pivot_table(
        values='n_clusters',
        index='country',
        columns='day_type',
        fill_value=0
    )
    
    x = np.arange(len(pivot_clusters.index))

    if 'working' in pivot_clusters.columns and 'non_working' in pivot_clusters.columns:
        axes[1, 1].bar(x - width/2, pivot_clusters['working'], width,
                      label='Working Days', color=WORK_COLOUR, alpha=0.8)
        axes[1, 1].bar(x + width/2, pivot_clusters['non_working'], width,
                      label='Non-Working Days', color=NONWORK_COLOUR, alpha=0.8)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(pivot_clusters.index, rotation=45)
    else:
        for i, day_type in enumerate(pivot_clusters.columns):
            axes[1, 1].bar(x + i*width - width*(len(pivot_clusters.columns)-1)/2, 
                          pivot_clusters[day_type], width, 
                          label=day_type, alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(pivot_clusters.index, rotation=45)
    
    axes[1, 1].set_title('Number of Clusters Found', fontsize=12)
    axes[1, 1].set_xlabel('Country')
    axes[1, 1].set_ylabel('Number of Clusters')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(COUNTRY_DAYTYPE_DIR / "country_daytype_combined_summary.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined analysis visualizations saved to: {COUNTRY_DAYTYPE_DIR}/")

# ============================================================================
# ADD THESE FUNCTION CALLS TO YOUR MAIN ANALYSIS PIPELINE
# ============================================================================
# ============================================================================
# CORRECTED TIME SERIES VISUALIZATION FUNCTIONS
# ============================================================================

def plot_country_time_series_profiles(clustering_data, energy_columns, output_dir=COUNTRY_DIR):
    """
    Create time series plots of 10-minute energy profiles for each country.
    Shows the actual 144 data points (24 hours × 6 measurements per hour).
    """
    if "COUNTRY" not in clustering_data.columns:
        print("  COUNTRY column not found for country time series plots")
        return
    
    countries = sorted(clustering_data["COUNTRY"].unique())
    
    # Create two types of plots:
    # 1. Individual country plots (subplots)
    # 2. Combined comparison plot
    
    # 1. Individual country plots in subplots
    n_countries = len(countries)
    n_cols = 3
    n_rows = (n_countries + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows), sharey=True)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    time_points = np.arange(144) / 6  # hours since 04:00 (diary order)
    
    for idx, country in enumerate(countries):
        country_data = clustering_data[clustering_data["COUNTRY"] == country]
        
        if len(country_data) == 0:
            axes[idx].set_visible(False)
            continue
        
        # Calculate mean and std for 10-minute intervals
        mean_profile = country_data[energy_columns].mean(axis=0).values
        std_profile = country_data[energy_columns].std(axis=0).values
        
        # Plot time series
        ax = axes[idx]
        ax.plot(time_points, mean_profile, 'b-', linewidth=1.5, alpha=0.8)
        ax.fill_between(time_points, mean_profile - std_profile, 
                        mean_profile + std_profile, alpha=0.2, color='blue')
        
        # Find peak and off-peak times
        peak_idx = np.argmax(mean_profile)
        peak_time = peak_idx / 6  # position on the diary-order axis
        peak_value = mean_profile[peak_idx]
        
        # Mark peak time with vertical line
        ax.axvline(x=peak_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Calculate area under curve (daily energy) and peak-to-average ratio
        daily_energy = np.sum(mean_profile) / 6  # Sum of 10-min intervals, divide by 6 for Wh
        avg_power = np.mean(mean_profile)
        peak_to_avg = peak_value / avg_power if avg_power > 0 else 0
        
        # Customize plot
        ax.set_title(f'{country}\n(n={len(country_data)} households)', fontsize=10)
        ax.set_xlabel('Time of day', fontsize=9)
        ax.set_ylabel('Power (W)', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        set_clock_axis(ax, 6)
        
        # Add statistics annotation
        stats_text = (f"Peak: {peak_value:.0f} W @ {slot_to_clock_str(peak_idx)}\n"
                     f"Daily: {daily_energy:.0f} Wh\n"
                     f"Peak/Avg: {peak_to_avg:.2f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(countries), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('10-Minute Energy Time Series Profiles by Country', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/country_timeseries_profiles_subplots.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Country time series profiles (subplots) saved to: {output_dir}/country_timeseries_profiles_subplots.png")
    
    # 2. Combined comparison plot
    fig_combined, ax_combined = plt.subplots(figsize=(14, 8))
    
    # Create color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
    
    for idx, country in enumerate(countries):
        country_data = clustering_data[clustering_data["COUNTRY"] == country]
        
        if len(country_data) == 0:
            continue
        
        mean_profile = country_data[energy_columns].mean(axis=0).values
        
        # Plot with unique color and label
        ax_combined.plot(time_points, mean_profile, color=colors[idx], 
                        linewidth=1.5, alpha=0.7, label=country)
    
    ax_combined.set_title('10-Minute Energy Time Series Profiles: Country Comparison', 
                         fontsize=16, fontweight='bold')
    ax_combined.set_xlabel('Time of day', fontsize=14)
    ax_combined.set_ylabel('Average Power (W)', fontsize=14)
    ax_combined.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax_combined.grid(True, alpha=0.3, linestyle=':')
    set_clock_axis(ax_combined, 2)
    
    # Add shaded areas for typical periods
    # Periods in clock time, drawn on the diary-order axis
    clock_span(ax_combined, 0, 6, alpha=0.1, color='blue', label='Night')
    clock_span(ax_combined, 6, 9, alpha=0.1, color='green', label='Morning')
    clock_span(ax_combined, 9, 17, alpha=0.1, color='yellow', label='Day')
    clock_span(ax_combined, 17, 20, alpha=0.1, color='orange', label='Evening')
    clock_span(ax_combined, 20, 24, alpha=0.1, color='purple', label='Late evening')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/country_timeseries_profiles_combined.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Country time series profiles (combined) saved to: {output_dir}/country_timeseries_profiles_combined.png")

def plot_daytype_time_series_profiles(clustering_data, energy_columns, output_dir=DAYTYPE_DIR):
    """
    Create time series plots comparing working vs non-working day energy profiles.
    Shows actual 10-minute intervals.
    """
    if "day_type" not in clustering_data.columns:
        print("  day_type column not found for day type time series plots")
        return
    
    # Prepare data
    working_data = clustering_data[clustering_data["day_type"] == "working"]
    non_working_data = clustering_data[clustering_data["day_type"] == "non_working"]
    
    if len(working_data) == 0 and len(non_working_data) == 0:
        print("  No data available for day type time series plots")
        return
    
    time_points = np.arange(144) / 6
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot working days if available
    if len(working_data) > 0:
        working_mean = working_data[energy_columns].mean(axis=0).values
        working_std = working_data[energy_columns].std(axis=0).values
        
        ax.plot(time_points, working_mean, 'b-', linewidth=2, alpha=0.8, label='Working Days')
        ax.fill_between(time_points, working_mean - working_std, 
                        working_mean + working_std, alpha=0.2, color='blue')
        
        # Mark peak time for working days
        peak_idx_work = np.argmax(working_mean)
        peak_time_work = peak_idx_work / 6
        ax.axvline(x=peak_time_work, color='blue', linestyle='--', alpha=0.5, 
                  label=f'Working Peak: {slot_to_clock_str(peak_idx_work)}')
    
    # Plot non-working days if available
    if len(non_working_data) > 0:
        non_working_mean = non_working_data[energy_columns].mean(axis=0).values
        non_working_std = non_working_data[energy_columns].std(axis=0).values
        
        ax.plot(time_points, non_working_mean, 'r-', linewidth=2, alpha=0.8, label='Non-Working Days')
        ax.fill_between(time_points, non_working_mean - non_working_std, 
                        non_working_mean + non_working_std, alpha=0.2, color='red')
        
        # Mark peak time for non-working days
        peak_idx_non = np.argmax(non_working_mean)
        peak_time_non = peak_idx_non / 6
        ax.axvline(x=peak_time_non, color='red', linestyle='--', alpha=0.5, 
                  label=f'Non-Working Peak: {slot_to_clock_str(peak_idx_non)}')
    
    # Customize plot
    ax.set_title('10-Minute Energy Time Series: Working Days vs Non-Working Days', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Time of day', fontsize=14)
    ax.set_ylabel('Average Power (W)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    set_clock_axis(ax, 2)
    
    # Add time period labels
    # Periods in clock time, converted to positions on the diary-order axis
    # (hours since 04:00). Night runs 00:00-06:00, which on this axis is the
    # last four hours plus the first two.
    time_periods = [(20, 24, 'Night'), (0, 2, 'Night'), (2, 5, 'Morning'),
                    (5, 13, 'Day'), (13, 16, 'Evening'), (16, 20, 'Late evening')]
    
    for start, end, label in time_periods:
        ax.axvspan(start, end, alpha=0.05, color='gray')
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.95, label, 
               ha='center', va='top', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daytype_timeseries_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Day type time series comparison saved to: {output_dir}/daytype_timeseries_comparison.png")
    
    # Create individual plots for each day type
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    day_types = [("working", "Working Days", 'blue'), 
                ("non_working", "Non-Working Days", 'red')]
    
    for idx, (day_type, label, color) in enumerate(day_types):
        day_data = clustering_data[clustering_data["day_type"] == day_type]
        
        if len(day_data) == 0:
            axes2[idx].set_visible(False)
            continue
        
        mean_profile = day_data[energy_columns].mean(axis=0).values
        std_profile = day_data[energy_columns].std(axis=0).values
        
        # Plot individual time series
        ax = axes2[idx]
        ax.plot(time_points, mean_profile, f'{color[0]}-', linewidth=2, alpha=0.8)
        ax.fill_between(time_points, mean_profile - std_profile, 
                        mean_profile + std_profile, alpha=0.2, color=color)
        
        # Calculate statistics
        daily_energy = np.sum(mean_profile) / 6
        peak_power = np.max(mean_profile)
        avg_power = np.mean(mean_profile)
        peak_time = np.argmax(mean_profile) / 6
        
        ax.set_title(f'{label}\n(n={len(day_data)} households)', fontsize=14)
        ax.set_xlabel('Time of day', fontsize=12)
        ax.set_ylabel('Average Power (W)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':')
        set_clock_axis(ax, 3)
        
        # Add statistics
        stats_text = (f"Daily Energy: {daily_energy:.0f} Wh\n"
                     f"Peak Power: {peak_power:.0f} W\n"
                     f"Avg Power: {avg_power:.0f} W\n"
                     f"Peak Time: {slot_to_clock_str(np.argmax(mean_profile))}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig2.suptitle('10-Minute Energy Time Series by Day Type', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daytype_timeseries_individual.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Day type time series (individual) saved to: {output_dir}/daytype_timeseries_individual.png")

def plot_country_timeseries_by_daytype(clustering_data, energy_columns, output_dir=COUNTRY_DAYTYPE_DIR):
    """
    Create two separate time series plots:
    1. Country energy profiles on working days
    2. Country energy profiles on non-working days
    (Day types follow DDV7 where available and the weekday otherwise.)
    Each shows 10-minute time series for each country.
    """
    if "COUNTRY" not in clustering_data.columns or "day_type" not in clustering_data.columns:
        print("  Need both COUNTRY and day_type columns for country-day type time series plots")
        return
    
    # Prepare data
    working_data = clustering_data[clustering_data["day_type"] == "working"]
    non_working_data = clustering_data[clustering_data["day_type"] == "non_working"]
    
    time_points = np.arange(144) / 6
    
    # Plot 1: Working days (weekdays)
    if len(working_data) > 0:
        countries_working = sorted(working_data["COUNTRY"].unique())
        
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        # Color palette for working days (blues)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(countries_working)))
        
        for idx, country in enumerate(countries_working):
            country_data = working_data[working_data["COUNTRY"] == country]
            
            if len(country_data) == 0:
                continue
            
            mean_profile = country_data[energy_columns].mean(axis=0).values
            
            ax1.plot(time_points, mean_profile, color=colors[idx], 
                    linewidth=1.5, alpha=0.7, label=f'{country} (n={len(country_data)})')
        
        ax1.set_title('10-Minute Energy Time Series by Country: Working Days', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time of day', fontsize=14)
        ax1.set_ylabel('Average Power (W)', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':')
        set_clock_axis(ax1, 2)
        
        # Highlight typical working hours (9-17)
        clock_span(ax1, 9, 17, alpha=0.1, color='yellow', label='Typical Working Hours')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/country_timeseries_working_days.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Country time series on working days saved to: {output_dir}/country_timeseries_working_days.png")
    
    # Plot 2: Non-working days (weekends)
    if len(non_working_data) > 0:
        countries_non_working = sorted(non_working_data["COUNTRY"].unique())
        
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        # Color palette for non-working days (reds)
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(countries_non_working)))
        
        for idx, country in enumerate(countries_non_working):
            country_data = non_working_data[non_working_data["COUNTRY"] == country]
            
            if len(country_data) == 0:
                continue
            
            mean_profile = country_data[energy_columns].mean(axis=0).values
            
            ax2.plot(time_points, mean_profile, color=colors[idx], 
                    linewidth=1.5, alpha=0.7, label=f'{country} (n={len(country_data)})')
        
        ax2.set_title('10-Minute Energy Time Series by Country: Non-Working Days',
                     fontsize=16, fontweight='bold')
        ax2.set_xlabel('Time of day', fontsize=14)
        ax2.set_ylabel('Average Power (W)', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle=':')
        set_clock_axis(ax2, 2)
        
        # Highlight typical leisure hours
        clock_span(ax2, 10, 12, alpha=0.1, color='green', label='Morning Leisure')
        clock_span(ax2, 18, 22, alpha=0.1, color='orange', label='Evening Leisure')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/country_timeseries_non_working_days.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Country time series on non-working days saved to: {output_dir}/country_timeseries_non_working_days.png")
    
    # Plot 3: Comparison for countries with both day types
    common_countries = set(working_data["COUNTRY"].unique()) & set(non_working_data["COUNTRY"].unique())
    
    if common_countries:
        # Create one plot per country showing both day types
        for country in sorted(common_countries):
            country_working = working_data[working_data["COUNTRY"] == country]
            country_non_working = non_working_data[non_working_data["COUNTRY"] == country]
            
            if len(country_working) > 0 and len(country_non_working) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                working_profile = country_working[energy_columns].mean(axis=0).values
                non_working_profile = country_non_working[energy_columns].mean(axis=0).values
                
                ax.plot(time_points, working_profile, 'b-', linewidth=2, 
                       alpha=0.8, label=f'Working Days (n={len(country_working)})')
                ax.plot(time_points, non_working_profile, 'r-', linewidth=2, 
                       alpha=0.8, label=f'Non-Working Days (n={len(country_non_working)})')
                
                # Calculate difference
                diff_profile = non_working_profile - working_profile
                ax.fill_between(time_points, 0, diff_profile, 
                               where=diff_profile>0, alpha=0.3, color='red', 
                               label='Higher on Non-Working')
                ax.fill_between(time_points, 0, diff_profile, 
                               where=diff_profile<0, alpha=0.3, color='blue', 
                               label='Higher on Working')
                
                ax.set_title(f'{country}: Working vs Non-Working Day Energy Profiles\n10-Minute Time Series', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Time of day', fontsize=12)
                ax.set_ylabel('Average Power (W)', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle=':')
                set_clock_axis(ax, 3)
                
                plt.tight_layout()
                safe_country = str(country).replace(" ", "_").replace("/", "_")
                plt.savefig(f"{output_dir}/country_{safe_country}_daytype_comparison_timeseries.png", 
                          dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"  Individual country day-type comparison time series saved for {len(common_countries)} countries")
        
# ============================================================================
# MISSING FUNCTION: CROSS-COUNTRY COMPARISONS
# ============================================================================

def create_cross_country_comparisons(combined_results):
    """
    Compare the same day type across different countries.
    """
    print("\n" + "="*80)
    print("CROSS-COUNTRY COMPARISONS")
    print("="*80)
    
    # Group by day type
    working_day_results = {k: v for k, v in combined_results.items() if k[1] == 'working'}
    non_working_day_results = {k: v for k, v in combined_results.items() if k[1] == 'non_working'}
    
    # Compare working days across countries
    if working_day_results:
        print("\nWorking Days - Cross-Country Comparison:")
        comparison_data = []
        
        for (country, day_type), data in working_day_results.items():
            energy_data = data[energy_columns].values
            daily_energy = energy_data.sum(axis=1) * INTERVAL_HOURS
            peak_power = energy_data.max(axis=1)
            
            comparison_data.append({
                'country': country,
                'day_type': day_type,
                'avg_daily_energy': daily_energy.mean(),
                'avg_peak_power': peak_power.mean(),
                'n_households': len(data),
                'n_clusters': data['cluster'].nunique() if 'cluster' in data.columns else 0
            })
        
        working_comparison = pd.DataFrame(comparison_data)
        working_comparison = working_comparison.sort_values('avg_daily_energy', ascending=False)
        
        print("\nCountries ranked by average daily energy on working days:")
        for i, row in working_comparison.iterrows():
            print(f"  {row['country']}: {row['avg_daily_energy']:.0f} Wh/day "
                  f"({row['n_households']} households, {row['n_clusters']} clusters)")
        
        working_comparison.to_csv(COUNTRY_DAYTYPE_DIR / "working_days_cross_country.csv", index=False)
        print(f"  Saved to: {COUNTRY_DAYTYPE_DIR}/working_days_cross_country.csv")
    
    # Compare non-working days across countries
    if non_working_day_results:
        print("\nNon-Working Days - Cross-Country Comparison:")
        comparison_data = []
        
        for (country, day_type), data in non_working_day_results.items():
            energy_data = data[energy_columns].values
            daily_energy = energy_data.sum(axis=1) * INTERVAL_HOURS
            peak_power = energy_data.max(axis=1)
            
            comparison_data.append({
                'country': country,
                'day_type': day_type,
                'avg_daily_energy': daily_energy.mean(),
                'avg_peak_power': peak_power.mean(),
                'n_households': len(data),
                'n_clusters': data['cluster'].nunique() if 'cluster' in data.columns else 0
            })
        
        non_working_comparison = pd.DataFrame(comparison_data)
        non_working_comparison = non_working_comparison.sort_values('avg_daily_energy', ascending=False)
        
        print("\nCountries ranked by average daily energy on non-working days:")
        for i, row in non_working_comparison.iterrows():
            print(f"  {row['country']}: {row['avg_daily_energy']:.0f} Wh/day "
                  f"({row['n_households']} households, {row['n_clusters']} clusters)")
        
        non_working_comparison.to_csv(COUNTRY_DAYTYPE_DIR / "non_working_days_cross_country.csv", index=False)
        print(f"  Saved to: {COUNTRY_DAYTYPE_DIR}/non_working_days_cross_country.csv")
    
    # Create comparison visualization
    if working_day_results and non_working_day_results:
        create_cross_country_comparison_visualization(working_day_results, non_working_day_results)

def create_cross_country_comparison_visualization(working_results, non_working_results):
    """
    Create visualization comparing same day types across countries.
    """
    # Prepare data
    working_data = []
    non_working_data = []
    
    for (country, _), data in working_results.items():
        daily_energy = (data[energy_columns].sum(axis=1) * INTERVAL_HOURS).mean()
        working_data.append({'country': country, 'avg_daily_energy': daily_energy})
    
    for (country, _), data in non_working_results.items():
        daily_energy = (data[energy_columns].sum(axis=1) * INTERVAL_HOURS).mean()
        non_working_data.append({'country': country, 'avg_daily_energy': daily_energy})
    
    working_df = pd.DataFrame(working_data).sort_values('avg_daily_energy', ascending=False)
    non_working_df = pd.DataFrame(non_working_data).sort_values('avg_daily_energy', ascending=False)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Cross-Country Energy Consumption Comparison', fontsize=16, fontweight='bold')
    
    # Plot working days
    axes[0].bar(range(len(working_df)), working_df['avg_daily_energy'], 
                color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Working Days', fontsize=14)
    axes[0].set_xlabel('Country')
    axes[0].set_ylabel('Average Daily Energy (Wh)')
    axes[0].set_xticks(range(len(working_df)))
    axes[0].set_xticklabels(working_df['country'], rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels for working days
    for i, energy in enumerate(working_df['avg_daily_energy']):
        axes[0].text(i, energy + max(working_df['avg_daily_energy'])*0.01, 
                    f'{energy:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot non-working days
    axes[1].bar(range(len(non_working_df)), non_working_df['avg_daily_energy'], 
                color='red', alpha=0.7, edgecolor='black')
    axes[1].set_title('Non-Working Days', fontsize=14)
    axes[1].set_xlabel('Country')
    axes[1].set_ylabel('Average Daily Energy (Wh)')
    axes[1].set_xticks(range(len(non_working_df)))
    axes[1].set_xticklabels(non_working_df['country'], rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels for non-working days
    for i, energy in enumerate(non_working_df['avg_daily_energy']):
        axes[1].text(i, energy + max(non_working_df['avg_daily_energy'])*0.01, 
                    f'{energy:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(COUNTRY_DAYTYPE_DIR / "cross_country_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Cross-country comparison visualization saved to: {COUNTRY_DAYTYPE_DIR}/cross_country_comparison.png")
    
# ============================================================================
# NEW FUNCTION: COMPARE PEAK POWER BETWEEN DAY TYPES
# ============================================================================

def compare_peak_power_day_types(combined_results):
    """
    Compare peak power between working and non-working days for each country.
    Calculate the percentage of countries where non-working day peak power 
    is NOT higher than working day peak power.
    """
    print("\n" + "="*80)
    print("PEAK POWER COMPARISON: WORKING vs NON-WORKING DAYS")
    print("="*80)
    
    # Group results by country
    country_peak_data = {}
    
    for (country, day_type), data in combined_results.items():
        # Calculate average peak power for this country and day type
        peak_power = data[energy_columns].max(axis=1).mean()
        # Coincident peak per household: maximum of the mean profile
        coincident_peak = data[energy_columns].mean(axis=0).max()

        if country not in country_peak_data:
            country_peak_data[country] = {}

        country_peak_data[country][day_type] = {
            'avg_peak_power': peak_power,
            'mean_profile_peak': coincident_peak,
            'n_households': len(data)
        }
    
    # Analyze comparison
    comparison_results = []
    countries_with_both_types = 0
    countries_non_working_not_higher = 0
    
    for country, day_data in country_peak_data.items():
        if 'working' in day_data and 'non_working' in day_data:
            countries_with_both_types += 1
            
            working_peak = day_data['working']['avg_peak_power']
            non_working_peak = day_data['non_working']['avg_peak_power']
            
            # Calculate difference and percentage difference
            diff = non_working_peak - working_peak
            pct_diff = (diff / working_peak) * 100 if working_peak > 0 else 0
            
            # Determine if non-working peak is NOT higher than working peak
            non_working_not_higher = non_working_peak <= working_peak
            
            if non_working_not_higher:
                countries_non_working_not_higher += 1
            
            w_coinc = day_data['working']['mean_profile_peak']
            nw_coinc = day_data['non_working']['mean_profile_peak']
            comparison_results.append({
                'country': country,
                'working_peak_power_w': working_peak,
                'non_working_peak_power_w': non_working_peak,
                'difference_w': diff,
                'percent_difference': pct_diff,
                'working_mean_profile_peak_w': w_coinc,
                'non_working_mean_profile_peak_w': nw_coinc,
                'mean_profile_peak_percent_difference':
                    (nw_coinc / w_coinc - 1) * 100 if w_coinc > 0 else 0,
                'non_working_not_higher': non_working_not_higher,
                'n_households_working': day_data['working']['n_households'],
                'n_households_non_working': day_data['non_working']['n_households']
            })
    
    # Calculate percentage
    if countries_with_both_types > 0:
        percentage_non_working_not_higher = (
            countries_non_working_not_higher / countries_with_both_types
        ) * 100
    else:
        percentage_non_working_not_higher = 0
    
    # Create DataFrame and sort by difference
    comparison_df = pd.DataFrame(comparison_results)
    
    # Sort by whether non-working is not higher, then by difference
    comparison_df = comparison_df.sort_values(
        ['non_working_not_higher', 'difference_w'], 
        ascending=[False, True]
    )
    
    # Save results
    comparison_df.to_csv(
        COUNTRY_DAYTYPE_DIR / "peak_power_daytype_comparison.csv", 
        index=False
    )
    
    # Print summary
    print(f"\nCountries with both working and non-working day data: {countries_with_both_types}")
    print(f"Countries where non-working day peak power is NOT higher than working day: "
          f"{countries_non_working_not_higher}")
    print(f"Percentage: {percentage_non_working_not_higher:.1f}%")
    
    # Detailed breakdown
    print("\nCountries where non-working peak ≤ working peak:")
    non_higher_countries = comparison_df[comparison_df['non_working_not_higher'] == True]
    for _, row in non_higher_countries.iterrows():
        print(f"  {row['country']}: {row['non_working_peak_power_w']:.0f}W (non-work) "
              f"≤ {row['working_peak_power_w']:.0f}W (work) "
              f"diff={row['difference_w']:.0f}W ({row['percent_difference']:.1f}%)")
    
    print("\nCountries where non-working peak > working peak:")
    higher_countries = comparison_df[comparison_df['non_working_not_higher'] == False]
    for _, row in higher_countries.iterrows():
        print(f"  {row['country']}: {row['non_working_peak_power_w']:.0f}W (non-work) "
              f"> {row['working_peak_power_w']:.0f}W (work) "
              f"diff=+{abs(row['difference_w']):.0f}W (+{abs(row['percent_difference']):.1f}%)")
    
    # Create visualization
    create_peak_power_comparison_visualization(comparison_df, percentage_non_working_not_higher)
    
    return comparison_df, percentage_non_working_not_higher

def create_peak_power_comparison_visualization(comparison_df, percentage):
    """
    Create visualization of peak power comparison between day types.
    """
    if len(comparison_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Peak Power Comparison: Working vs Non-Working Days\n'
                f'Non-Working Peak ≤ Working Peak: {percentage:.1f}% of Countries', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Bar chart comparing peak power for each country
    x = np.arange(len(comparison_df))
    width = 0.35
    
    # Color bars based on whether non-working is higher
    working_colors = ['lightblue'] * len(comparison_df)
    non_working_colors = []
    
    for idx, row in comparison_df.iterrows():
        if row['non_working_not_higher']:
            non_working_colors.append('lightcoral')  # Red for not higher
        else:
            non_working_colors.append('lightgreen')  # Green for higher
    
    axes[0, 0].bar(x - width/2, comparison_df['working_peak_power_w'], 
                   width, label='Working Days', color=working_colors, edgecolor='black')
    axes[0, 0].bar(x + width/2, comparison_df['non_working_peak_power_w'], 
                   width, label='Non-Working Days', color=non_working_colors, edgecolor='black')
    
    axes[0, 0].set_xlabel('Country')
    axes[0, 0].set_ylabel('Average Peak Power (W)')
    axes[0, 0].set_title('Peak Power by Country and Day Type')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(comparison_df['country'], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Difference in peak power
    axes[0, 1].bar(x, comparison_df['difference_w'], 
                   color=comparison_df['non_working_not_higher'].map(
                       {True: 'red', False: 'green'}),
                   edgecolor='black')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Country')
    axes[0, 1].set_ylabel('Difference (Non-Working - Working) (W)')
    axes[0, 1].set_title('Peak Power Difference (Negative = Non-Working ≤ Working)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(comparison_df['country'], rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, diff in enumerate(comparison_df['difference_w']):
        axes[0, 1].text(i, diff + (max(comparison_df['difference_w'])*0.01 if diff >= 0 
                                  else min(comparison_df['difference_w'])*0.01), 
                       f'{diff:.0f}', ha='center', va='bottom' if diff >= 0 else 'top', 
                       fontsize=8)
    
    # Plot 3: Percentage difference
    axes[1, 0].bar(x, comparison_df['percent_difference'], 
                   color=comparison_df['non_working_not_higher'].map(
                       {True: 'red', False: 'green'}),
                   edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Country')
    axes[1, 0].set_ylabel('Percentage Difference (%)')
    axes[1, 0].set_title('Percentage Difference in Peak Power')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(comparison_df['country'], rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    summary_stats = {
        'Countries with both day types': len(comparison_df),
        'Non-working peak ≤ working peak': f"{percentage:.1f}%",
        'Average working peak (W)': f"{comparison_df['working_peak_power_w'].mean():.0f}",
        'Average non-working peak (W)': f"{comparison_df['non_working_peak_power_w'].mean():.0f}",
        'Average difference (W)': f"{comparison_df['difference_w'].mean():.0f}",
        'Median difference (W)': f"{comparison_df['difference_w'].median():.0f}"
    }
    
    axes[1, 1].axis('off')
    summary_text = "\n".join([f"{key}: {value}" for key, value in summary_stats.items()])
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(
        COUNTRY_DAYTYPE_DIR / "peak_power_daytype_comparison.png", 
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f"\nPeak power comparison visualization saved to: "
          f"{COUNTRY_DAYTYPE_DIR}/peak_power_daytype_comparison.png")

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

# Step 1: Determine day type for all households
# ============================================================================
# MODIFIED MAIN ANALYSIS PIPELINE
# ============================================================================

# 1. After loading data (plot for all countries)
print("\n" + "="*80)
print("CREATING AVERAGE ENERGY PROFILE VISUALIZATIONS")
print("="*80)

# Plot average energy profiles for all countries
print("\n1. Country time series profiles...")
plot_country_time_series_profiles(clustering_data, energy_columns, output_dir=COUNTRY_DIR)

# Plot day type energy profiles (if day_type column exists)
# This will be called later after day_type is determined

print("Starting clustering analysis...")

# FIRST: Run country analysis on ALL data (before day type filtering)
print("\n" + "="*80)
print("STEP 1: COUNTRY-LEVEL ANALYSIS (ON ALL DATA)")
print("="*80)

country_results_all = {}
if not RUN_ALL_DATA_PASS:
    print("Skipped: it repeats the country solutions of Step 2 exactly "
          "(set HETUS_RUN_ALL_DATA_PASS=1 to run it).")
elif "COUNTRY" in clustering_data.columns:
    countries = clustering_data["COUNTRY"].unique()
    
    for country in countries:
        country_data = clustering_data[clustering_data["COUNTRY"] == country]
        
        if len(country_data) < 20:  # Or your threshold
            print(f"Skipping {country}: only {len(country_data)} households")
            continue
        
        print(f"\nAnalyzing {country} ({len(country_data)} households)...")
        
        # Use categorical features without COUNTRY for this analysis
        country_cat_features = [f for f in categorical_features if f != "COUNTRY"]
        
        result = perform_clustering_analysis(
            data=country_data,
            analysis_name=f"{country}_ALL_DATA",
            output_dir=COUNTRY_DIR,
            energy_cols=energy_columns,
            cat_features=country_cat_features
        )
        
        if result is not None:
            country_results_all[country] = result

# SECOND: Now determine day types and do day-specific analyses
print("\n" + "="*80)
print("STEP 2: DETERMINING DAY TYPES FOR DAY-SPECIFIC ANALYSES")
print("="*80)

# Your existing day type determination code...
if "DDV7" in clustering_data.columns:
    print("Using DDV7 and DDV1 for day type determination...")
    clustering_data["day_type"] = clustering_data.apply(determine_day_type, axis=1)
else:
    print("DDV7 not found, using DDV1 only...")
    def determine_day_type_ddv1_only(row):
        ddv1 = row.get('DDV1', -9)
        if ddv1 in [2, 3, 4, 5, 6]:
            return "working"
        elif ddv1 in [1, 7]:
            return "non_working"
        else:
            return None
    clustering_data["day_type"] = clustering_data.apply(determine_day_type_ddv1_only, axis=1)

# Remove households with undetermined day type
original_count = len(clustering_data)
clustering_data_with_daytypes = clustering_data[clustering_data["day_type"].notna()].copy()
print(f"\nRemoved {original_count - len(clustering_data_with_daytypes)} households with undetermined day type")
print(f"Remaining households for day-specific analyses: {len(clustering_data_with_daytypes)}")

# THIRD: Run day type and combined analyses (only on households with day types)
print("\n" + "="*80)
print("STEP 3: DAY TYPE AND COMBINED ANALYSES")
print("="*80)

# Step 2: Country analysis (if COUNTRY column exists)
country_results = {}
if "COUNTRY" in clustering_data.columns:
    print("\n" + "="*80)
    print("STEP 2: COUNTRY-LEVEL ANALYSIS")
    print("="*80)
    
    countries = clustering_data["COUNTRY"].unique()
    
    for country in countries:
        country_data = clustering_data[clustering_data["COUNTRY"] == country]
        
        if len(country_data) < 20:
            continue
        
        print(f"\nAnalyzing {country} ({len(country_data)} households)...")
        
        country_cat_features = [f for f in categorical_features if f != "COUNTRY"]
        result = perform_clustering_analysis(
            data=country_data,
            analysis_name=f"{country}",
            output_dir=COUNTRY_DIR,
            energy_cols=energy_columns,
            cat_features=country_cat_features
        )
        
        if result is not None:
            country_results[country] = result
        

# Step 3: Day type analysis
print("\n" + "="*80)
print("STEP 3: DAY TYPE ANALYSIS")
print("="*80)

working_data = clustering_data[clustering_data["day_type"] == "working"]
non_working_data = clustering_data[clustering_data["day_type"] == "non_working"]

print(f"\nWorking days: {len(working_data)} households")
print(f"Non-working days: {len(non_working_data)} households")

print("\n2. Day type time series profiles...")
plot_daytype_time_series_profiles(clustering_data_with_daytypes, energy_columns, output_dir=DAYTYPE_DIR)

# Clustering for each day type
if len(working_data) >= 20:
    print(f"\nClustering analysis for WORKING DAYS...")
    perform_clustering_analysis(
        data=working_data,
        analysis_name="Working_Days",
        output_dir=DAYTYPE_DIR,
        energy_cols=energy_columns,
        cat_features=categorical_features
    )

if len(non_working_data) >= 20:
    print(f"\nClustering analysis for NON-WORKING DAYS...")
    perform_clustering_analysis(
        data=non_working_data,
        analysis_name="Non_Working_Days",
        output_dir=DAYTYPE_DIR,
        energy_cols=energy_columns,
        cat_features=categorical_features
    )



# Step 4: Country & Day Type Combined Analysis
print("\n" + "="*80)
print("STEP 4: COUNTRY & DAY TYPE COMBINED ANALYSIS")
print("="*80)
print("This is the most granular analysis: clusters specific to each (country, day_type) combination")

combined_results = perform_country_daytype_combined_analysis(clustering_data, min_households=15)

print("\n3. Country time series by day type...")
plot_country_timeseries_by_daytype(clustering_data_with_daytypes, energy_columns, output_dir=COUNTRY_DAYTYPE_DIR)

print("\n4. Mean profiles and coincident peaks by country and day type...")
daytype_profiles = export_daytype_mean_profiles(clustering_data_with_daytypes, energy_columns)
plot_daytype_small_multiples(daytype_profiles, output_dir=COUNTRY_DAYTYPE_DIR)

# Step 5: Create comprehensive summaries
print("\n" + "="*80)
print("STEP 5: CREATING COMPREHENSIVE SUMMARIES")
print("="*80)

# Create country comparison summary (if we have country results)
if country_results:
    create_country_comparison_summary(country_results)

# Create combined analysis comparison (if we have combined results)
if combined_results:
    # Create cross-country comparison of similar day types
    print("\nCreating cross-country comparisons...")
    create_cross_country_comparisons(combined_results)
    
    # NEW: Compare peak power between day types
    print("\nComparing peak power between working and non-working days...")
    peak_comparison_df, percentage = compare_peak_power_day_types(combined_results)
    
    # Add this result to the final summary
    print("\n" + "="*80)
    print("KEY FINDING: PEAK POWER COMPARISON")
    print("="*80)
    print(f"Percentage of countries where non-working day peak power is NOT higher "
          f"than working day peak power: {percentage:.1f}%")
    print("\nInterpretation:")
    if percentage > 50:
        print(f"  - In the majority of countries ({percentage:.1f}%), households do NOT have higher")
        print("    peak power demands on non-working days compared to working days.")
    else:
        print(f"  - In only {percentage:.1f}% of countries, households do NOT have higher")
        print("    peak power demands on non-working days.")

# Cluster mean profiles of all stratified solutions (country, day type, and
# country by day type), for figures and checks outside the pipeline
write_profile_exports()
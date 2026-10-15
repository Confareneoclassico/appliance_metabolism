import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from scipy import stats
from itertools import combinations
import warnings
import os
from paths import PROCESSED_DIR, FIGURES_DIR
from disclosure import protect, MIN_N_PUBLISH
warnings.filterwarnings("ignore")

# --- Configuration of the clustering (October 2026 re-run) ------------------
# The clusters are formed on the 144 standardised power slots only. The 16
# background variables stay in the results file, where they describe the
# clusters and are tested against them (here and in Stage 07), but they no
# longer enter the clustering, so the tests are not run on variables that
# helped to form the clusters. HETUS_CLUSTER_ON_SOCIO=1 restores the earlier
# joint feature space, for comparison.
CLUSTER_ON_SOCIO = os.environ.get("HETUS_CLUSTER_ON_SOCIO", "0") == "1"
# Optional stability diagnostics of the final solution (silhouette width on a
# subsample, and adjusted Rand index and cluster-wise Jaccard similarity over
# repeated half-samples). Off by default because they add several minutes.
RUN_STABILITY = os.environ.get("HETUS_STABILITY", "0") == "1"
STABILITY_REPS = int(os.environ.get("HETUS_STABILITY_REPS", "10"))

# Load the prepared clustering data
clustering_data = pd.read_csv(PROCESSED_DIR / "prepared_clustering_data.csv")

# Output folders are created by paths.py when it is imported.

# Extract energy time series columns and categorical features
energy_columns = [col for col in clustering_data.columns if col.startswith("Household_Energy_W_")]
categorical_features = [
    "DDV1", "DDV3", "DDV6","COUNTRY","INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
    "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"
]

print(f"Energy time series features: {len(energy_columns)}")
print(f"Categorical features: {len(categorical_features)}")

# Create a column transformer for preprocessing: power slots only, unless
# CLUSTER_ON_SOCIO is set
transformers = [("energy", StandardScaler(), energy_columns)]
if CLUSTER_ON_SOCIO:
    transformers.append(
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
preprocessor = ColumnTransformer(transformers=transformers)
print("Clustering features: "
      + ("power slots and one-hot background variables" if CLUSTER_ON_SOCIO else "power slots only"))

# Fit the preprocessor and transform the data
X_processed = preprocessor.fit_transform(clustering_data)

# Apply PCA to reduce dimensionality for better clustering
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_processed)
print(f"Reduced dimensionality from {X_processed.shape[1]} to {X_pca.shape[1]} with PCA")

# --- PCA Explained Variance Analysis ---
explained_variance_ratio = pca.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance_ratio)

print("\n--- PCA Explained Variance ---")
print(f"Explained variance ratio of each component: {explained_variance_ratio}")
print(f"Cumulative explained variance: {cum_explained_variance}")

# Determine number of components for specific variance thresholds
for threshold in [0.80, 0.90, 0.95, 0.99]:
    try:
        n_components = np.where(cum_explained_variance >= threshold)[0][0] + 1
        print(f"Number of components to explain {threshold*100:.0f}% variance: {n_components}")
    except IndexError:
        print(f"PCA does not explain up to {threshold*100:.0f}% variance with the current number of components.")

# --- Interpretation of PCA Components (Loadings) ---
# Get feature names after one-hot encoding
feature_names = list(energy_columns)
if CLUSTER_ON_SOCIO:
    feature_names += list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))

print("\n--- Top Features for First Few Principal Components ---")
num_components_to_show = min(5, pca.n_components_)
num_top_features = 10

for i in range(num_components_to_show):
    component_loadings = pca.components_[i]
    # Get absolute loadings and sort to find most influential features
    abs_loadings = np.abs(component_loadings)
    sorted_indices = np.argsort(abs_loadings)[::-1]
    
    print(f"\nPrincipal Component {i+1} (Explained Variance: {explained_variance_ratio[i]*100:.2f}%):")
    print("  Top positive contributing features:")
    for idx in sorted_indices[:num_top_features]:
        if component_loadings[idx] > 0:
            print(f"    - {feature_names[idx]}: {component_loadings[idx]:.4f}")
    print("  Top negative contributing features:")
    for idx in sorted_indices[:num_top_features]:
        if component_loadings[idx] < 0:
            print(f"    - {feature_names[idx]}: {component_loadings[idx]:.4f}")

# Elbow method to find optimal k
wcss = []
k_range = range(2, 16)  # Test more clusters to find the best

for i in k_range:
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    print(f"K={i}, WCSS={kmeans.inertia_:.2f}")

# Automatic elbow detection using KneeLocator
kl = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
optimal_k = kl.elbow

if optimal_k is None:
    print("Warning: Could not automatically detect elbow. Using k=8 as default based on previous analysis.")
    optimal_k = 8 # Default if elbow not found, based on previous analysis
else:
    print(f"Automatically detected optimal k: {optimal_k}")

# --- Improved Elbow Method Plot ---
plt.figure(figsize=(10, 7))
plt.plot(k_range, wcss, marker='o', linestyle='-' , color='#1f77b4', linewidth=2)
plt.axvline(x=optimal_k, color='#d62728', linestyle='--', alpha=0.8, label=f'Optimal K = {optimal_k}' , linewidth=1.5)
plt.title('Elbow Method for Optimal K', fontsize=16, fontweight='bold')
plt.xlabel('Number of Clusters (K)' , fontsize=14)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)' , fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "elbow_method_improved.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Improved Elbow method plot saved to {FIGURES_DIR}/elbow_method_improved.png")

# Perform final KMeans clustering with optimal k
print(f"\nPerforming clustering with optimal k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clustering_data["cluster"] = kmeans.fit_predict(X_pca)

print(f"Clustering completed with {optimal_k} clusters.")
print("Cluster sizes:")
cluster_counts = clustering_data["cluster"].value_counts().sort_index()
print(cluster_counts)

# --- Optional stability diagnostics (HETUS_STABILITY=1) ----------------------
# Silhouette width on a random subsample of 10,000 household-days, and, over
# STABILITY_REPS half-samples, (a) the adjusted Rand index between K-means
# refitted on the half-sample and the full-data solution restricted to it, and
# (b) for each full-data cluster, the Jaccard similarity with its best-matching
# cluster in the refit (cluster-wise stability in the sense of Hennig, 2007:
# above about 0.75 stable, below 0.5 dissolved).
if RUN_STABILITY:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    print("\n--- Stability diagnostics ---")
    labels_full = clustering_data["cluster"].values
    sil = silhouette_score(X_pca, labels_full, sample_size=min(10000, len(X_pca)),
                           random_state=42)
    print(f"Silhouette width (10,000 household-days): {sil:.3f}")
    rng = np.random.default_rng(42)
    aris, jaccard = [], np.full((STABILITY_REPS, optimal_k), np.nan)
    for b in range(STABILITY_REPS):
        idx = rng.choice(len(X_pca), size=len(X_pca) // 2, replace=False)
        refit = KMeans(n_clusters=optimal_k, random_state=100 + b, n_init=5).fit(X_pca[idx])
        lab_full, lab_refit = labels_full[idx], refit.labels_
        aris.append(adjusted_rand_score(lab_full, lab_refit))
        for k in range(optimal_k):
            in_k = lab_full == k
            if not in_k.any():
                continue
            best = 0.0
            for m in range(optimal_k):
                in_m = lab_refit == m
                union = np.sum(in_k | in_m)
                if union:
                    best = max(best, np.sum(in_k & in_m) / union)
            jaccard[b, k] = best
        print(f"  half-sample {b + 1}/{STABILITY_REPS}: ARI = {aris[-1]:.3f}")
    stability = pd.DataFrame({
        'cluster': range(optimal_k),
        'size': [int((labels_full == k).sum()) for k in range(optimal_k)],
        'jaccard_mean': np.nanmean(jaccard, axis=0),
        'jaccard_min': np.nanmin(jaccard, axis=0),
    })
    stability['ari_mean'] = float(np.mean(aris))
    stability['ari_min'] = float(np.min(aris))
    stability['silhouette'] = float(sil)
    # Disclosure control (disclosure.py): no cluster-wise result for a cluster
    # of fewer than 20 household-days, a flag for one of 20 to 49
    protect(stability, n_col='size', stat_cols=['jaccard_mean', 'jaccard_min']).to_csv(
        PROCESSED_DIR / "cluster_stability.csv", index=False)
    print(stability.round(3).to_string(index=False))
    print(f"Stability diagnostics saved to {PROCESSED_DIR}/cluster_stability.csv")

# ============================================================================
# STATISTICAL SIGNIFICANCE ANALYSIS OF CATEGORICAL FEATURES
# ============================================================================

print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE ANALYSIS OF CATEGORICAL FEATURES")
print("="*80)

def chi_square_test_categorical(feature_name, cluster_data, alpha=0.05):
    """
    Perform Chi-square test for independence on a categorical feature across clusters.
    Returns test results and effect size (Cramer's V).
    """
    # Create contingency table
    contingency_table = pd.crosstab(cluster_data["cluster"], cluster_data[feature_name])
    
    # Perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calculate Cramer's V effect size
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0
    
    # Interpret Cramer's V
    if cramers_v < 0.1:
        effect_strength = "Negligible"
    elif cramers_v < 0.2:
        effect_strength = "Weak"
    elif cramers_v < 0.3:
        effect_strength = "Moderate"
    elif cramers_v < 0.4:
        effect_strength = "Relatively Strong"
    else:
        effect_strength = "Strong"
    
    return {
        'feature': feature_name,
        'chi2': chi2,
        'p_value': p,
        'dof': dof,
        'significant': p < alpha,
        'cramers_v': cramers_v,
        'effect_strength': effect_strength,
        'n_clusters': contingency_table.shape[0],
        'n_categories': contingency_table.shape[1]
    }

def interpret_cramers_v(v):
    """Interpret Cramer's V effect size"""
    if v < 0.1:
        return "Negligible"
    elif v < 0.2:
        return "Weak"
    elif v < 0.3:
        return "Moderate"
    elif v < 0.4:
        return "Relatively Strong"
    else:
        return "Strong"

def analyze_feature_distributions(clustering_data, categorical_features, alpha=0.05):
    """
    Comprehensive analysis of categorical feature distributions across clusters
    """
    results = []
    significant_features = []
    
    for feature in categorical_features:
        print(f"\n{'='*60}")
        print(f"ANALYSIS FOR: {feature}")
        print(f"{'='*60}")
        
        # Get contingency table
        contingency_table = pd.crosstab(clustering_data["cluster"], clustering_data[feature], normalize='index')
        print(f"Distribution across clusters:")
        print(contingency_table.round(3))
        
        # Perform statistical test
        test_result = chi_square_test_categorical(feature, clustering_data, alpha)
        
        # Add to results
        results.append(test_result)
        
        if test_result['significant']:
            significant_features.append(feature)
        
        # Print test results
        print(f"\nChi-square test results:")
        print(f"  Chi2 statistic: {test_result['chi2']:.2f}")
        print(f"  Degrees of freedom: {test_result['dof']}")
        print(f"  P-value: {test_result['p_value']:.6f}")
        print(f"  Significant at alpha={alpha}: {test_result['significant']}")
        print(f"  Cramer's V (effect size): {test_result['cramers_v']:.4f}")
        print(f"  Effect strength: {test_result['effect_strength']}")
        
        # Identify clusters with most different distributions
        identify_outlier_clusters(feature, clustering_data)
        
        # Create visualization
        visualize_categorical_distribution(feature, clustering_data, test_result['significant'])
    
    return pd.DataFrame(results), significant_features

def identify_outlier_clusters(feature, clustering_data, top_n=3):
    """
    Identify clusters with distributions most different from the average
    """
    # Calculate normalized distribution per cluster
    cluster_dist = pd.crosstab(clustering_data["cluster"], clustering_data[feature], normalize='index')
    
    # Calculate average distribution across all clusters
    avg_distribution = cluster_dist.mean()
    
    # Calculate Euclidean distance of each cluster from average
    distances = {}
    for cluster_idx in cluster_dist.index:
        cluster_props = cluster_dist.loc[cluster_idx]
        distance = np.sqrt(((cluster_props - avg_distribution) ** 2).sum())
        distances[cluster_idx] = distance
    
    # Sort by distance (most different first)
    sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nClusters with most distinctive distributions (top {top_n}):")
    for cluster_idx, distance in sorted_distances[:top_n]:
        print(f"  Cluster {cluster_idx}: Distance from average = {distance:.4f}")
        
        # Identify which categories contribute most to the difference
        cluster_props = cluster_dist.loc[cluster_idx]
        differences = (cluster_props - avg_distribution).abs()
        top_diff_categories = differences.nlargest(3).index.tolist()
        
        for cat in top_diff_categories:
            cluster_val = cluster_props[cat]
            avg_val = avg_distribution[cat]
            diff_pct = (cluster_val - avg_val) * 100
            print(f"    Category {cat}: Cluster={cluster_val:.3f}, Average={avg_val:.3f}, "
                  f"Difference={diff_pct:+.1f}%")

def visualize_categorical_distribution(feature, clustering_data, is_significant):
    """
    Create visualization of categorical feature distribution across clusters
    """
    # Get normalized distribution
    dist_df = pd.crosstab(clustering_data["cluster"], clustering_data[feature], normalize='index')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get number of categories
    n_categories = dist_df.shape[1]
    categories = dist_df.columns
    
    # Set up bar positions
    x = np.arange(len(dist_df))
    width = 0.8 / n_categories
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_categories))
    
    # Plot bars for each category
    for i, category in enumerate(categories):
        proportions = dist_df[category].values
        ax.bar(x + i * width - width * (n_categories - 1) / 2, 
               proportions, width, label=str(category), color=colors[i], alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    significance_text = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
    ax.set_title(f'{feature} Distribution Across Clusters\n({significance_text}, α=0.05)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {i}' for i in dist_df.index])
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{feature}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Distribution plot saved to {FIGURES_DIR}/{feature}_distribution.png")

def perform_pairwise_comparisons(clustering_data, categorical_features):
    """
    Perform pairwise comparisons between clusters for each feature
    """
    print("\n" + "="*80)
    print("PAIRWISE CLUSTER COMPARISONS")
    print("="*80)
    
    pairwise_results = {}
    
    for feature in categorical_features:
        print(f"\n{feature}:")
        
        # Get all unique cluster pairs
        cluster_pairs = list(combinations(sorted(clustering_data["cluster"].unique()), 2))
        significant_pairs = []
        
        for cluster1, cluster2 in cluster_pairs:
            # Get data for the two clusters
            data1 = clustering_data[clustering_data["cluster"] == cluster1][feature]
            data2 = clustering_data[clustering_data["cluster"] == cluster2][feature]
            
            # Create contingency table for this pair
            table = pd.crosstab(
                pd.concat([pd.Series([0]*len(data1), index=data1.index), 
                          pd.Series([1]*len(data2), index=data2.index)]),
                pd.concat([data1, data2])
            )
            
            # Chi-square test for this pair
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                chi2, p, dof, expected = stats.chi2_contingency(table)
                
                # Calculate effect size for this pair
                n = table.sum().sum()
                k = min(table.shape[0] - 1, table.shape[1] - 1)
                cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0
                
                if p < 0.05 and cramers_v > 0.2:  # Moderate or stronger effect
                    significant_pairs.append({
                        'cluster1': cluster1,
                        'cluster2': cluster2,
                        'p_value': p,
                        'cramers_v': cramers_v,
                        'effect_strength': interpret_cramers_v(cramers_v)
                    })
        
        if significant_pairs:
            pairwise_results[feature] = significant_pairs
            print(f"  Significantly different cluster pairs:")
            for pair in significant_pairs[:5]:  # Show top 5
                print(f"    Cluster {pair['cluster1']} vs Cluster {pair['cluster2']}: "
                      f"p={pair['p_value']:.4f}, V={pair['cramers_v']:.3f} ({pair['effect_strength']})")
        else:
            print(f"  No significantly different cluster pairs found")
    
    return pairwise_results

def create_cluster_profiles(clustering_data, categorical_features, significant_features):
    """
    Create profiles for each cluster based on categorical features
    """
    print("\n" + "="*80)
    print("CLUSTER PROFILES")
    print("="*80)
    
    if not significant_features:
        print("No features show statistically significant differences across clusters.")
        return
    
    print(f"\nUsing {len(significant_features)} significant features for profiling:")
    print(", ".join(significant_features))
    
    # Create profile for each cluster
    for cluster_id in sorted(clustering_data["cluster"].unique()):
        print(f"\n{'='*40}")
        print(f"PROFILE: CLUSTER {cluster_id}")
        print(f"Size: {sum(clustering_data['cluster'] == cluster_id)} households")
        print(f"{'='*40}")
        
        cluster_data = clustering_data[clustering_data["cluster"] == cluster_id]
        
        for feature in significant_features[:5]:  # Show top 5 most significant
            # Calculate distribution for this cluster
            cluster_dist = cluster_data[feature].value_counts(normalize=True)
            
            # Calculate overall distribution
            overall_dist = clustering_data[feature].value_counts(normalize=True)
            
            # Find most characteristic category for this cluster
            most_common = cluster_dist.idxmax() if not cluster_dist.empty else None
            if most_common is not None:
                cluster_freq = cluster_dist[most_common]
                overall_freq = overall_dist.get(most_common, 0)
                diff_pct = (cluster_freq - overall_freq) * 100
                
                print(f"{feature}:")
                print(f"  Most common: {most_common} ({cluster_freq:.1%} in cluster vs {overall_freq:.1%} overall)")
                print(f"  Difference: {diff_pct:+.1f} percentage points")
        
        # Calculate distinctiveness score
        distinctiveness_score = calculate_distinctiveness_score(cluster_id, clustering_data, significant_features)
        print(f"\nOverall distinctiveness score: {distinctiveness_score:.4f}")
        print(f"Interpretation: {'Highly distinct' if distinctiveness_score > 0.15 else 'Moderately distinct' if distinctiveness_score > 0.1 else 'Similar to average'}")

def calculate_distinctiveness_score(cluster_id, clustering_data, features):
    """
    Calculate how distinct a cluster is from the overall population
    """
    scores = []
    cluster_data = clustering_data[clustering_data["cluster"] == cluster_id]
    
    for feature in features:
        # Calculate distribution for this cluster
        cluster_dist = cluster_data[feature].value_counts(normalize=True)
        
        # Calculate overall distribution
        overall_dist = clustering_data[feature].value_counts(normalize=True)
        
        # Calculate Euclidean distance between distributions
        all_categories = set(cluster_dist.index) | set(overall_dist.index)
        distance = 0
        for cat in all_categories:
            cluster_val = cluster_dist.get(cat, 0)
            overall_val = overall_dist.get(cat, 0)
            distance += (cluster_val - overall_val) ** 2
        distance = np.sqrt(distance)
        
        scores.append(distance)
    
    return np.mean(scores) if scores else 0

def create_summary_report(stat_results, pairwise_results, optimal_k):
    """
    Create a comprehensive summary report
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*80)
    
    # Count significant features
    significant_features = stat_results[stat_results['significant']]
    
    print(f"\nTotal features analyzed: {len(stat_results)}")
    print(f"Features with statistically significant differences: {len(significant_features)}")
    print(f"Percentage of significant features: {len(significant_features)/len(stat_results)*100:.1f}%")
    
    # Sort features by effect size
    sorted_features = significant_features.sort_values('cramers_v', ascending=False)
    
    print("\nTop 5 features with strongest differences (by Cramer's V):")
    for i, (_, row) in enumerate(sorted_features.head(5).iterrows()):
        print(f"{i+1}. {row['feature']}: Cramer's V = {row['cramers_v']:.4f} ({row['effect_strength']})")
    
    # Calculate average effect size
    avg_effect_size = significant_features['cramers_v'].mean()
    print(f"\nAverage effect size (Cramer's V) for significant features: {avg_effect_size:.4f}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if len(significant_features) > len(stat_results) * 0.5:
        print("✓ STRONG CLUSTER SEPARATION: Most features show significant differences.")
        print("  The clustering appears to capture meaningful patterns in the data.")
    else:
        print("⚠ MODERATE CLUSTER SEPARATION: Only some features show significant differences.")
        print("  Consider: 1) Reviewing clustering parameters, 2) Feature selection")
    
    print(f"\n✓ ACTIONABLE INSIGHTS:")
    print(f"  1. Focus on {len(significant_features)} significant features for interpretation")
    print(f"  2. Use cluster profiles for targeted interventions/analysis")
    print(f"  3. Validate with domain knowledge")
    print("  4. Consider feature engineering for better separation")

# ============================================================================
# RUN THE STATISTICAL ANALYSIS
# ============================================================================

# Analyze categorical feature distributions
stat_results_df, significant_features = analyze_feature_distributions(clustering_data, categorical_features)

# Perform pairwise comparisons
pairwise_results = perform_pairwise_comparisons(clustering_data, categorical_features)

# Create cluster profiles
create_cluster_profiles(clustering_data, categorical_features, significant_features)

# Create summary report
create_summary_report(stat_results_df, pairwise_results, optimal_k)

# Save statistical results
stat_results_df.to_csv(PROCESSED_DIR / "categorical_feature_statistics.csv", index=False)
print(f"\nStatistical analysis saved to {PROCESSED_DIR}/categorical_feature_statistics.csv")

# ============================================================================
# CONTINUE WITH ORIGINAL CLUSTERING VISUALIZATION AND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CONTINUING WITH CLUSTER ANALYSIS AND VISUALIZATION")
print("="*80)

# --- Figure 2: mean 10-minute demand profile of each global cluster ---
# Sized for a full-width journal figure (about 190 mm wide), so that the text
# stays legible once the image is scaled down in print. The eight categorical
# colours follow a fixed order checked for colour-vision deficiency; a ninth
# or later cluster reuses a colour with a dashed line instead of a new hue.
# The values plotted are mean power in W per 10-minute interval (not Wh).
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# HETUS diary days run from 04:00 to 03:59 the next day, and slot 0 of each
# series covers 04:00-04:09 (SUF variable Mact001). The x axis keeps the diary
# order (hours since 04:00) and is labelled with clock time.
DIARY_START_HOUR = 4
time_points = np.arange(144) / 6  # hours since 04:00


def clock_labels(ticks):
    return [f"{(DIARY_START_HOUR + t) % 24:02d}:00" for t in ticks]


def slot_clock(slot):
    """HH:MM clock time of a 10-minute slot (slot 0 = 04:00)."""
    minutes = (DIARY_START_HOUR * 60 + 10 * slot) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
cluster_sizes = clustering_data["cluster"].value_counts().reindex(range(optimal_k), fill_value=0)
mean_profiles = {
    k: clustering_data.loc[clustering_data["cluster"] == k, energy_columns].mean(axis=0).values
    for k in range(optimal_k)
}


def style_profile_axes(ax, tick_size):
    """Recessive solid grid, no top/right frame, y axis anchored at zero."""
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=tick_size)
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)


# Single panel with all clusters (the figure used in the article)
fig, ax = plt.subplots(figsize=(10, 6))
for k in range(optimal_k):
    ax.plot(time_points, mean_profiles[k], color=SERIES_COLOURS[k % 8],
            linestyle="-" if k < 8 else "--", linewidth=2.0,
            label=f"Cluster {k} (n = {cluster_sizes[k]:,})")
ax.set_xticks(range(0, 25, 3))
ax.set_xticklabels(clock_labels(range(0, 25, 3)))
ax.set_xlabel("Time of day", fontsize=16, color=INK)
ax.set_ylabel("Mean power demand (W)", fontsize=16, color=INK)
style_profile_axes(ax, tick_size=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
          fontsize=13, handlelength=2.2, columnspacing=1.6, labelcolor=INK)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "cluster_energy_profiles_10min_combined.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Combined 10-minute plot saved to cluster_energy_profiles_10min_combined.png")

# Small multiples: one cluster per panel, the other clusters in grey for
# context. An alternative to the single panel when the lines overlap.
n_cols = 3
n_rows = int(np.ceil(optimal_k / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.6 * n_rows + 0.9),
                         sharex=True, sharey=True, squeeze=False)
for idx in range(n_rows * n_cols):
    ax = axes[idx // n_cols][idx % n_cols]
    if idx >= optimal_k:
        ax.set_visible(False)
        # Show x tick labels on the panel above an empty slot
        if idx - n_cols >= 0:
            axes[(idx - n_cols) // n_cols][idx % n_cols].xaxis.set_tick_params(labelbottom=True)
        continue
    for j in range(optimal_k):
        if j != idx:
            ax.plot(time_points, mean_profiles[j], color=AXIS, linewidth=0.9)
    ax.plot(time_points, mean_profiles[idx], color=SERIES_COLOURS[0], linewidth=2.0)
    share = 100 * cluster_sizes[idx] / len(clustering_data)
    share_txt = f"{share:.1f}%" if share >= 0.1 else "<0.1%"
    ax.set_title(f"Cluster {idx} (n = {cluster_sizes[idx]:,}, {share_txt})",
                 fontsize=13, color=INK, loc="left")
    ax.set_xticks(range(0, 25, 6))
    ax.set_xticklabels(clock_labels(range(0, 25, 6)))
    style_profile_axes(ax, tick_size=12)
fig.supxlabel("Time of day", fontsize=15, color=INK)
fig.supylabel("Mean power demand (W)", fontsize=15, color=INK)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "cluster_energy_profiles_10min_panels.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Small-multiple version saved to cluster_energy_profiles_10min_panels.png")

# Analyze categorical features per cluster (original analysis)
print("\nCategorical Feature Distribution per Cluster:")
categorical_analysis = {}

for col in categorical_features:
    print(f"\n--- {col} ---")
    cross_tab = pd.crosstab(clustering_data["cluster"], clustering_data[col], normalize='index')
    print(cross_tab.round(3))
    categorical_analysis[col] = cross_tab

# --- Improved PCA Scatter Plot ---
plt.figure(figsize=(12, 9))

# Plot each cluster with a different color and label
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for cluster_id in range(optimal_k):
    mask = clustering_data["cluster"] == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=[colors[cluster_id]], label=f'Cluster {cluster_id}', alpha=0.7)

# Calculate and plot cluster centroids
centroids = []
for cluster_id in range(optimal_k):
    centroid = np.mean(X_pca[clustering_data["cluster"] == cluster_id], axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, 
            color='red', edgecolor='black', linewidth=1.5, label='Cluster Centroids')

plt.title('Household Clusters in PCA Space', fontsize=18, fontweight='bold')
plt.xlabel('First Principal Component', fontsize=14)
plt.ylabel('Second Principal Component', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clusters_pca_improved.png", dpi=300, bbox_inches='tight')
plt.close()
print("Improved Cluster PCA plot saved to clusters_pca_improved.png")

# Save clustering results
clustering_data.to_csv(PROCESSED_DIR / "clustering_results.csv", index=False)
print(f"Clustering results saved to {PROCESSED_DIR}/clustering_results.csv")

# Create a detailed summary of cluster characteristics
cluster_summary = []
for cluster_id in range(optimal_k):
    cluster_data = clustering_data[clustering_data["cluster"] == cluster_id]
    energy_data = cluster_data[energy_columns].values
    
    summary = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        # Daily energy (Wh) = sum of the 144 mean-power values (W) x 10/60 h.
        # The peak is the highest 10-minute mean power of each household-day,
        # in W (this column was previously mislabelled avg_peak_energy_wh).
        'avg_daily_energy_wh': (np.sum(energy_data, axis=1) * 10 / 60).mean(),
        'avg_peak_power_w': np.max(energy_data, axis=1).mean(),
        # Clock hour of the linear mean of the household-day peak slots
        # (slot 0 = 04:00). Stage 08 gives the circular mean.
        'avg_peak_time': int((DIARY_START_HOUR + np.argmax(energy_data, axis=1).mean() / 6) % 24),
        'energy_std': np.std(energy_data),
        # Maximum of the cluster's mean profile (the coincident peak per
        # household, plotted in Figure 2) and its clock time
        'mean_profile_peak_w': mean_profiles[cluster_id].max(),
        'mean_profile_peak_time': slot_clock(int(np.argmax(mean_profiles[cluster_id]))),
    }
    
    # Add mode for categorical features
    for col in categorical_features:
        mode_values = cluster_data[col].mode()
        summary[f'mode_{col}'] = mode_values.iloc[0] if not mode_values.empty else None
        summary[f'mode_freq_{col}'] = (cluster_data[col] == mode_values.iloc[0]).mean() if not mode_values.empty else 0
    
    cluster_summary.append(summary)

cluster_summary_df = pd.DataFrame(cluster_summary)
# The export follows Eurostat's publication rules for HETUS (disclosure.py).
# No global cluster has fewer than MIN_N_PUBLISH household-days (the smallest
# has 50). If one had, its profile could be recovered from the published mean
# profile of the whole sample, and its household-days would have to be reported
# with the largest cluster, as Stage 06 does for the stratified solutions.
if (np.asarray(cluster_sizes) < MIN_N_PUBLISH).any():
    print(f"WARNING: a global cluster has fewer than {MIN_N_PUBLISH} household-days. "
          "Report it with the largest cluster before sharing the exports "
          "(see collect_cluster_profiles in Stage 06).")
summary_stat_cols = [c for c in cluster_summary_df.columns if c not in ('cluster', 'size')]
protect(cluster_summary_df, n_col='size', stat_cols=summary_stat_cols).to_csv(
    PROCESSED_DIR / "cluster_summary.csv", index=False)
print(f"Cluster summary saved to {PROCESSED_DIR}/cluster_summary.csv")

# Mean profile of each global cluster (W in each 10-minute slot, slot 0 =
# 04:00), so that Figure 2 can be redrawn and checked without the microdata
profile_rows = []
for cluster_id in range(optimal_k):
    row = {'cluster': cluster_id, 'n': int(cluster_sizes[cluster_id])}
    row.update({f'P_{t:03d}': float(mean_profiles[cluster_id][t]) for t in range(len(energy_columns))})
    profile_rows.append(row)
protect(pd.DataFrame(profile_rows), n_col='n').to_csv(
    PROCESSED_DIR / "cluster_mean_profiles.csv", index=False)
print(f"Cluster mean profiles saved to {PROCESSED_DIR}/cluster_mean_profiles.csv")

# Create comprehensive statistical summary
print("\n" + "="*80)
print("FINAL STATISTICAL SUMMARY")
print("="*80)

print(f"\nClustering Analysis Completed Successfully!")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Total households clustered: {len(clustering_data)}")
print(f"Significant categorical features: {len(significant_features)}/{len(categorical_features)}")

# Show most distinctive features
if len(significant_features) > 0:
    print("\nMost distinctive categorical features (highest Cramer's V):")
    top_features = stat_results_df.sort_values('cramers_v', ascending=False).head(3)
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: Cramer's V = {row['cramers_v']:.4f} ({row['effect_strength']})")

print(f"\nAnalysis complete! Figures are in {FIGURES_DIR}")
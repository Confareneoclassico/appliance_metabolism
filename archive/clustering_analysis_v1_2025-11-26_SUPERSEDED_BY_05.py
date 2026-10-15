
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
warnings.filterwarnings("ignore")

# Load the prepared clustering data
clustering_data = pd.read_csv("prepared_clustering_data.csv")

# Extract energy time series columns and categorical features
energy_columns = [col for col in clustering_data.columns if col.startswith("Household_Energy_W_")]
categorical_features = [
    "DDV1", "DDV3", "DDV6","COUNTRY","INC4_1", "IND6_1", "HHC1", "HHC3", "HHC4", "HHC5",
    "HHQ5", "HHQ9_1", "INC1", "INC2", "IND22_1", "IND44"
]

print(f"Energy time series features: {len(energy_columns)}")
print(f"Categorical features: {len(categorical_features)}")

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("energy", StandardScaler(), energy_columns),
                                    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)


    ])

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
# To interpret the components, we look at the `pca.components_` attribute.
# Each row in `pca.components_` corresponds to a principal component,
# and each column corresponds to an original feature (after preprocessing).
# The values (loadings) indicate how much each original feature contributes to the component.
# Higher absolute values mean a stronger contribution.

# Get feature names after one-hot encoding
feature_names = energy_columns + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))

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
plt.savefig("./Figures/elbow_method_improved.png", dpi=300, bbox_inches='tight')
plt.close()
print("Improved Elbow method plot saved to elbow_method_improved.png")

# Perform final KMeans clustering with optimal k
print(f"\nPerforming clustering with optimal k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clustering_data["cluster"] = kmeans.fit_predict(X_pca)

print(f"Clustering completed with {optimal_k} clusters.")
print("Cluster sizes:")
cluster_counts = clustering_data["cluster"].value_counts().sort_index()
print(cluster_counts)

# Analyze cluster characteristics
print("\nCluster Analysis:")

# Calculate average energy profiles for each cluster
cluster_profiles = []
# --- Improved Cluster Energy Profiles Plot ---
plt.figure(figsize=(16, 10))

# Define a color palette for better distinction
colors = sns.color_palette("tab10", n_colors=optimal_k)

for cluster_id in range(optimal_k):
    cluster_mask = clustering_data["cluster"] == cluster_id
    cluster_energy = clustering_data.loc[cluster_mask, energy_columns].mean(axis=0).values
    
    # Use all 144 (24*6) 10-minute data points
    time_points = np.arange(144) / 6  # Convert to hours for x-axis (0-24)
    
    # Plot line
    plt.plot(time_points, cluster_energy, color=colors[cluster_id], linewidth=1.5, alpha=0.8)
    # Plot individual points
    plt.scatter(time_points, cluster_energy, color=colors[cluster_id], s=20, alpha=0.6,
                label=f'Cluster {cluster_id} (n={cluster_mask.sum()})')

plt.title('10-minute Energy Demand Profiles with Individual Points', fontsize=18, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Energy Demand (Wh)', fontsize=14)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(0, 25, 2), fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 24)
plt.tight_layout()
plt.savefig("./Figures/cluster_energy_profiles_10min_combined.png", dpi=300, bbox_inches='tight')
plt.close()
print("Combined 10-minute plot saved to cluster_energy_profiles_10min_combined.png")

# Analyze categorical features per cluster
print("\nCategorical Feature Distribution per Cluster:")
categorical_analysis = {}

for col in categorical_features:
    print(f"\n--- {col} ---")
    cross_tab = pd.crosstab(clustering_data["cluster"], clustering_data[col], normalize='index')
    print(cross_tab.round(3))
    categorical_analysis[col] = cross_tab

# --- Improved PCA Scatter Plot ---
plt.figure(figsize=(12, 9))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clustering_data["cluster"], 
                     cmap="viridis", alpha=0.7, s=40, edgecolor='w', linewidth=0.5)

# Calculate and plot cluster centroids
# Calculate and plot cluster centroids
centroids = []
for cluster_id in range(optimal_k):
    centroid = np.mean(X_pca[clustering_data["cluster"] == cluster_id], axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)

# Plot each cluster with a different color and label
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for cluster_id in range(optimal_k):
    mask = clustering_data["cluster"] == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=[colors[cluster_id]], label=f'Cluster {cluster_id}', alpha=0.7)

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
plt.savefig("./Figures/clusters_pca_improved.png", dpi=300, bbox_inches='tight')
plt.close()
print("Improved Cluster PCA plot saved to clusters_pca_improved.png")

# Save clustering results
clustering_data.to_csv("clustering_results.csv", index=False)
print("Clustering results saved to clustering_results.csv")

# Create a detailed summary of cluster characteristics
cluster_summary = []
for cluster_id in range(optimal_k):
    cluster_data = clustering_data[clustering_data["cluster"] == cluster_id]
    energy_data = cluster_data[energy_columns].values
    
    summary = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_daily_energy_wh': np.sum(energy_data, axis=1).mean(),
        'avg_peak_energy_wh': np.max(energy_data, axis=1).mean(),
        'avg_peak_time': np.argmax(energy_data, axis=1).mean() // 6,  # Convert to hour
        'energy_std': np.std(energy_data),
    }
    
    # Add mode for categorical features
    for col in categorical_features:
        mode_values = cluster_data[col].mode()
        summary[f'mode_{col}'] = mode_values.iloc[0] if not mode_values.empty else None
        summary[f'mode_freq_{col}'] = (cluster_data[col] == mode_values.iloc[0]).mean() if not mode_values.empty else 0
    
    cluster_summary.append(summary)

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df.to_csv("cluster_summary.csv", index=False)
print("Cluster summary saved to cluster_summary.csv")

print(f"\nClustering analysis completed successfully!")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Total households clustered: {len(clustering_data)}")


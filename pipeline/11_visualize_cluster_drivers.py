import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from paths import DAYTYPE_DIR, COUNTRY_DIR, DRIVER_FIGURES_DIR

# --- Configuration ---
# Top features identified by Chi-squared test
TOP_FEATURES = ["HHC5", "HHC1", "INC2", "DDV6"]

# Files representing granular clustering solutions
# Country files are named by the two-letter country code (Stage 06)
FILES_TO_PROCESS = {
    "Working_Days": DAYTYPE_DIR / "Working_Days_clustering_results.csv",
    "Non_Working_Days": DAYTYPE_DIR / "Non_Working_Days_clustering_results.csv",
    "PL": COUNTRY_DIR / "PL_clustering_results.csv"
}

OUTPUT_DIR = DRIVER_FIGURES_DIR
# --- End Configuration ---

# Mapping for better plot labels (based on HETUS codes)
FEATURE_LABELS = {
    "HHC5": "Number of Persons Aged 18+",
    "HHC1": "Household Size",
    "INC2": "Age Group",
    "DDV6": "Employment/Student Status",
    "HHQ5": "Home Ownership Status"
}

def visualize_driver_distribution(file_path, analysis_name, feature):
    """
    Calculates and plots the distribution of a single feature across all clusters
    in a given clustering result file.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping visualization for {analysis_name} - {feature}.")
        return

    if feature not in df.columns or 'cluster' not in df.columns:
        print(f"Missing required columns in {file_path}. Skipping.")
        return

    # 1. Prepare Data: Calculate the percentage distribution
    # Drop missing values (-99) for cleaner visualization
    temp_df = df[df[feature] != -99].copy()
    
    if temp_df.empty:
        print(f"No valid data for feature {feature} in {analysis_name}. Skipping.")
        return

    # Calculate the count of each feature value within each cluster
    contingency_table = pd.crosstab(temp_df['cluster'], temp_df[feature])
    
    # Calculate the percentage of each feature value within each cluster (row-wise normalization)
    # This shows the composition of each cluster in terms of the feature
    distribution_pct = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
    
    # Transpose for plotting (clusters on x-axis, feature values stacked)
    plot_data = distribution_pct.T
    
    # 2. Plotting
    plt.figure(figsize=(12, 7))
    
    # Use a stacked bar plot
    plot_data.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
    
    # 3. Customization
    plt.title(f'Distribution of {FEATURE_LABELS.get(feature, feature)} Across Clusters\n({analysis_name} Analysis)', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Percentage of Cluster Households (%)', fontsize=12)
    plt.xticks(rotation=0)
    
    # Move legend outside the plot
    plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

    # 4. Save Plot
    safe_name = f"{analysis_name}_{feature}_distribution.png"
    save_path = os.path.join(OUTPUT_DIR, safe_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {save_path}")


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting visualization of cluster driver distributions...")
    
    for analysis_name, file_path in FILES_TO_PROCESS.items():
        for feature in TOP_FEATURES:
            visualize_driver_distribution(file_path, analysis_name, feature)
            
    print(f"\nVisualization complete. Plots are saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

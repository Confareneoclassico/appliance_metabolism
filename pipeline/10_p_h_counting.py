import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from paths import PROCESSED_DIR, FIGURES_DIR

# Load the data
df = pd.read_csv(PROCESSED_DIR / 'person_daily_metabolic_totals.csv')

print("Data loaded successfully!")
print(f"Total rows: {len(df)}")
print(f"Total unique households (HID): {df['HID'].nunique()}")
print(f"Total unique diarists (PID): {df['PID'].nunique()}")

# Count diarists per household
print("\n" + "="*60)
print("DIARISTS PER HOUSEHOLD ANALYSIS")
print("="*60)

# Count unique diarists (PID) per household (HID)
diarists_per_household = df.groupby('HID')['PID'].nunique().reset_index()
diarists_per_household.columns = ['HID', 'n_diarists']

# Extract country from HID
diarists_per_household['country'] = diarists_per_household['HID'].apply(
    lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else 'Unknown'
)

# Basic statistics
print(f"\nBASIC STATISTICS:")
print(f"Total households: {len(diarists_per_household)}")
print(f"Mean diarists per household: {diarists_per_household['n_diarists'].mean():.2f}")
print(f"Median diarists per household: {diarists_per_household['n_diarists'].median():.2f}")
print(f"Standard deviation: {diarists_per_household['n_diarists'].std():.2f}")
print(f"Minimum diarists: {diarists_per_household['n_diarists'].min()}")
print(f"Maximum diarists: {diarists_per_household['n_diarists'].max()}")

# Distribution analysis
distribution = diarists_per_household['n_diarists'].value_counts().sort_index()
print(f"\nDISTRIBUTION BREAKDOWN:")
for n_diarists, count in distribution.items():
    percentage = (count / len(diarists_per_household)) * 100
    print(f"  {n_diarists} diarist(s): {count} households ({percentage:.1f}%)")

# Country-level analysis
print(f"\nCOUNTRY-LEVEL ANALYSIS:")
country_stats = diarists_per_household.groupby('country').agg({
    'n_diarists': ['count', 'mean', 'median', 'std', 'min', 'max']
}).round(2)

# Flatten column names
country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns.values]
print(country_stats)

# Create visualizations
print(f"\nCreating visualizations...")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Distribution of Diarists per Household', fontsize=16, fontweight='bold')

# Plot 1: Histogram of diarists per household
counts, bins, patches = axes[0, 0].hist(diarists_per_household['n_diarists'], 
                                       bins=range(1, diarists_per_household['n_diarists'].max() + 2),
                                       edgecolor='black', alpha=0.7, align='left')
axes[0, 0].set_xlabel('Number of Diarists per Household')
axes[0, 0].set_ylabel('Number of Households')
axes[0, 0].set_title('Histogram: Diarists per Household')
axes[0, 0].grid(True, alpha=0.3)

# Add value labels on bars
for count, patch in zip(counts, patches):
    if count > 0:
        axes[0, 0].text(patch.get_x() + patch.get_width()/2, count + max(counts)*0.01, 
                       f'{int(count)}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Box plot by country
country_data = diarists_per_household.copy()
# Limit to countries with sufficient data for better visualization
country_counts = country_data['country'].value_counts()
min_households = 5  # Minimum households to show separately
major_countries = country_counts[country_counts >= min_households].index
country_data['country_plot'] = country_data['country'].apply(
    lambda x: x if x in major_countries else 'Other'
)

sns.boxplot(data=country_data, x='country_plot', y='n_diarists', ax=axes[0, 1])
axes[0, 1].set_xlabel('Country')
axes[0, 1].set_ylabel('Number of Diarists')
axes[0, 1].set_title('Box Plot: Diarists per Household by Country')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Pie chart of distribution
dist_counts = diarists_per_household['n_diarists'].value_counts().sort_index()
axes[1, 0].pie(dist_counts.values, labels=dist_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Pie Chart: Distribution of Diarists per Household')

# Plot 4: Cumulative distribution
sorted_counts = np.sort(diarists_per_household['n_diarists'])
y_vals = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
axes[1, 1].plot(sorted_counts, y_vals, marker='.', linestyle='-', linewidth=2)
axes[1, 1].set_xlabel('Number of Diarists per Household')
axes[1, 1].set_ylabel('Cumulative Proportion')
axes[1, 1].set_title('Cumulative Distribution Function')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "diarists_per_household_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Create additional detailed visualization
plt.figure(figsize=(12, 8))

# Bar plot with percentages
dist_pct = (diarists_per_household['n_diarists'].value_counts().sort_index() / len(diarists_per_household) * 100)
bars = plt.bar(dist_pct.index, dist_pct.values, color='skyblue', edgecolor='black', alpha=0.7)

plt.xlabel('Number of Diarists per Household', fontsize=12)
plt.ylabel('Percentage of Households (%)', fontsize=12)
plt.title('Detailed Distribution: Diarists per Household', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, dist_pct.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "diarists_per_household_detailed.png", dpi=300, bbox_inches='tight')
plt.show()

# Create summary table for display
summary_table = pd.DataFrame({
    'Number of Diarists': distribution.index,
    'Household Count': distribution.values,
    'Percentage': (distribution.values / len(diarists_per_household) * 100).round(1)
})

print(f"\n" + "="*50)
print("SUMMARY TABLE")
print("="*50)
print(summary_table.to_string(index=False))

# Additional insights
print(f"\n" + "="*50)
print("ADDITIONAL INSIGHTS")
print("="*50)

# Most common household size
most_common = distribution.idxmax()
most_common_count = distribution.max()
most_common_pct = (most_common_count / len(diarists_per_household)) * 100
print(f"Most common household size: {most_common} diarist(s) ({most_common_pct:.1f}% of households)")

# Single diarist households
single_diarist = distribution.get(1, 0)
single_pct = (single_diarist / len(diarists_per_household)) * 100
print(f"Single-diarist households: {single_diarist} ({single_pct:.1f}%)")

# Multi-diarist households (2+)
multi_diarist = len(diarists_per_household) - single_diarist
multi_pct = (multi_diarist / len(diarists_per_household)) * 100
print(f"Multi-diarist households (2+): {multi_diarist} ({multi_pct:.1f}%)")

# Save the diarists per household data to CSV
diarists_per_household.to_csv(PROCESSED_DIR / 'diarists_per_household_summary.csv', index=False)
print(f"\nDetailed summary saved to {PROCESSED_DIR}/diarists_per_household_summary.csv")

print(f"\nVisualizations saved in {FIGURES_DIR}:")
print(f"  - diarists_per_household_distribution.png")
print(f"  - diarists_per_household_detailed.png")
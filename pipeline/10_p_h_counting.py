"""Stage 10: number of diarists per household-day (console and figures).

Diarists are counted as distinct persons (HID, PID) within each household-day
(HID, DIARY). Versions up to 1.0.1 counted PID without HID, although PID is
the person number within the household (BE_1 is the first person of every
Belgian household), and counted the diarists of a household over both diary
days (audit of November 2026, L14).

The console and the figures follow the publication rules (disclosure.py): the
largest numbers of diarists are pooled into one class so that every class
holds at least 50 household-days, and no minimum or maximum is given. The
file diarists_per_household_summary.csv has one row per household-day and
stays on the machine.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from paths import PROCESSED_DIR, FIGURES_DIR
from disclosure import pool_tail, size_class, MIN_N_UNFLAGGED

# Load the data
df = pd.read_csv(PROCESSED_DIR / 'person_daily_metabolic_totals.csv')

print("Data loaded successfully!")
print(f"Total rows: {len(df)}")
print(f"Total unique households (HID): {df['HID'].nunique()}")
print(f"Total unique diarists (HID, PID): {len(df[['HID', 'PID']].drop_duplicates())}")
print(f"Total household-days (HID, DIARY): {len(df[['HID', 'DIARY']].drop_duplicates())}")

# Count diarists per household-day
print("\n" + "="*60)
print("DIARISTS PER HOUSEHOLD-DAY ANALYSIS")
print("="*60)

# Distinct persons per household-day (within a household, PID identifies the
# person)
diarists_per_household = df.groupby(['HID', 'DIARY'])['PID'].nunique().reset_index()
diarists_per_household.columns = ['HID', 'DIARY', 'n_diarists']

# Extract country from HID
diarists_per_household['country'] = diarists_per_household['HID'].apply(
    lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else 'Unknown'
)

n_diarists = diarists_per_household['n_diarists']
total = len(diarists_per_household)

# Classes under the publication rules: the largest numbers of diarists are
# pooled into one class so that every class holds at least MIN_N_UNFLAGGED
# household-days, and no minimum or maximum is printed, since each describes a
# single household-day (November 2026; before, the console and the figures
# showed every number of diarists, down to classes of a few household-days,
# and the largest by country)
distribution, tail = pool_tail(n_diarists.value_counts())
labels = list(distribution.index)

print(f"\nBASIC STATISTICS:")
print(f"Total household-days: {total}")
print(f"Mean diarists per household-day: {n_diarists.mean():.2f}")
print(f"Median diarists per household-day: {n_diarists.median():.2f}")
print(f"Standard deviation: {n_diarists.std():.2f}")

print(f"\nDISTRIBUTION BREAKDOWN:")
for label, count in distribution.items():
    shown = count if count >= MIN_N_UNFLAGGED else size_class(count)
    print(f"  {label} diarist(s): {shown} household-days ({100 * count / total:.1f}%)")

# Country-level analysis (no minimum or maximum)
print(f"\nCOUNTRY-LEVEL ANALYSIS:")
country_stats = diarists_per_household.groupby('country').agg({
    'n_diarists': ['count', 'mean', 'median', 'std']
}).round(2)
country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns.values]
print(country_stats.to_string())

# Create visualizations
print(f"\nCreating visualizations...")
plt.style.use('default')
sns.set_palette("husl")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Distribution of Diarists per Household-Day', fontsize=16, fontweight='bold')

# Plot 1: household-days by number of diarists, pooled classes
x = np.arange(len(labels))
axes[0, 0].bar(x, distribution.values, edgecolor='black', alpha=0.7)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].set_xlabel('Number of Diarists per Household-Day')
axes[0, 0].set_ylabel('Number of Household-Days')
axes[0, 0].set_title('Household-Days by Number of Diarists')
axes[0, 0].grid(True, alpha=0.3)
for xi, count in zip(x, distribution.values):
    axes[0, 0].text(xi, count + distribution.max() * 0.01, f'{int(count)}',
                    ha='center', va='bottom', fontweight='bold')

# Plot 2: box plot by country, whiskers at the 5th and 95th percentiles and no
# single household-days drawn; countries with fewer than MIN_N_UNFLAGGED
# household-days are drawn together
country_data = diarists_per_household.copy()
country_counts = country_data['country'].value_counts()
major_countries = country_counts[country_counts >= MIN_N_UNFLAGGED].index
country_data['country_plot'] = country_data['country'].apply(
    lambda c: c if c in major_countries else 'Other'
)
sns.boxplot(data=country_data, x='country_plot', y='n_diarists', ax=axes[0, 1],
            whis=(5, 95), showfliers=False)
axes[0, 1].set_xlabel('Country')
axes[0, 1].set_ylabel('Number of Diarists')
axes[0, 1].set_title('Diarists per Household-Day by Country (5th to 95th percentile)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: shares of the pooled classes
axes[1, 0].pie(distribution.values, labels=labels, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Shares of Household-Days by Number of Diarists')

# Plot 4: cumulative distribution over the pooled classes
cdf = distribution.cumsum() / total
axes[1, 1].step(x, cdf.values, where='post', marker='o', linewidth=2)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_xlabel('Number of Diarists per Household-Day')
axes[1, 1].set_ylabel('Cumulative Proportion')
axes[1, 1].set_title('Cumulative Distribution')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "diarists_per_household_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Percentages of the pooled classes
plt.figure(figsize=(12, 8))
dist_pct = distribution / total * 100
bars = plt.bar(x, dist_pct.values, color='skyblue', edgecolor='black', alpha=0.7)
plt.xticks(x, labels)
plt.xlabel('Number of Diarists per Household-Day', fontsize=12)
plt.ylabel('Percentage of Household-Days (%)', fontsize=12)
plt.title('Detailed Distribution: Diarists per Household-Day', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, value in zip(bars, dist_pct.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "diarists_per_household_detailed.png", dpi=300, bbox_inches='tight')
plt.close()

summary_table = pd.DataFrame({
    'Number of Diarists': labels,
    'Household-Day Count': [c if c >= MIN_N_UNFLAGGED else size_class(c) for c in distribution.values],
    'Percentage': (distribution.values / total * 100).round(1)
})

print(f"\n" + "="*50)
print("SUMMARY TABLE")
print("="*50)
print(summary_table.to_string(index=False))

# Additional insights
print(f"\n" + "="*50)
print("ADDITIONAL INSIGHTS")
print("="*50)

most_common = distribution.idxmax()
most_common_pct = distribution.max() / total * 100
print(f"Most common number of diarists: {most_common} ({most_common_pct:.1f}% of household-days)")

single_diarist = int((n_diarists == 1).sum())
single_pct = single_diarist / total * 100
print(f"Single-diarist household-days: {single_diarist} ({single_pct:.1f}%)")

multi_diarist = total - single_diarist
multi_pct = multi_diarist / total * 100
print(f"Multi-diarist household-days (2+): {multi_diarist} ({multi_pct:.1f}%)")

# Save the diarists per household data to CSV
diarists_per_household.to_csv(PROCESSED_DIR / 'diarists_per_household_summary.csv', index=False)
print(f"\nDetailed summary saved to {PROCESSED_DIR}/diarists_per_household_summary.csv")

print(f"\nVisualizations saved in {FIGURES_DIR}:")
print(f"  - diarists_per_household_distribution.png")
print(f"  - diarists_per_household_detailed.png")
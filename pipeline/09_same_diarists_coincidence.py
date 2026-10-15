import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from paths import PROCESSED_DIR, FIGURES_DIR

# ============================================================================
# FIX (applied during pipeline consolidation): the original version of this
# script treated -99 (and other HETUS missing/non-response codes) as an
# ordinary activity value. Two diarists who both had no recorded activity
# at the same interval were therefore counted as "sharing an activity",
# which inflates every reported coincidence percentage by an unknown
# amount. MISSING_CODES below are excluded from the activity sets before
# any intersection is computed, consistent with the -99-exclusion
# convention used in 07_statistical_validation.py.
# ============================================================================
MISSING_CODES = {-99, -9, -8, -7, -6, -1}

def analyze_same_activity_diarists(lu_data):
    """
    Analyze whether diarists from the same household record the same activities
    at the same time intervals (either both as main, both as secondary, or one as each).

    Missing/non-response activity codes (MISSING_CODES) are excluded from the
    comparison: two diarists both lacking a recorded activity at the same
    interval are NOT counted as sharing an activity.
    """
    print("Starting analysis of same activities among household diarists...")
    
    # Get unique household-day combinations
    household_days = lu_data[['HID', 'DIARY']].drop_duplicates()
    print(f"Found {len(household_days)} unique household-day combinations")
    
    # Initialize results storage
    results = {
        'same_activity_counts': [],
        'household_summary': [],
        'interval_details': []
    }
    
    # Process each household-day
    for _, (hid, diary) in household_days.iterrows():
        household_data = lu_data[(lu_data['HID'] == hid) & (lu_data['DIARY'] == diary)]
        
        if len(household_data) < 2:
            continue  # Skip households with only one diarist
            
        diarists = household_data['PID'].unique()
        n_diarists = len(diarists)
        
        # Track same activities per interval
        same_activity_intervals = 0
        total_comparable_intervals = 0
        
        household_interval_details = []
        
        # Check each 10-minute interval
        for interval in range(1, 145):  # Mact1 to Mact144, Sactn1 to Sactn144
            mact_col = f'Mact{interval}'
            sactn_col = f'Sactn{interval}'
            
            # Get activities for each diarist in this interval
            diarist_activities = {}
            for diarist in diarists:
                diarist_data = household_data[household_data['PID'] == diarist].iloc[0]
                mact = diarist_data[mact_col] if not pd.isna(diarist_data[mact_col]) else None
                sactn = diarist_data[sactn_col] if not pd.isna(diarist_data[sactn_col]) else None
                # FIX: treat missing/non-response codes as "no activity recorded",
                # matching the None branch below, instead of a comparable value.
                if mact is not None and mact in MISSING_CODES:
                    mact = None
                if sactn is not None and sactn in MISSING_CODES:
                    sactn = None
                diarist_activities[diarist] = {
                    'mact': mact,
                    'sactn': sactn,
                    'all_activities': set()
                }
                if mact is not None:
                    diarist_activities[diarist]['all_activities'].add(mact)
                if sactn is not None:
                    diarist_activities[diarist]['all_activities'].add(sactn)
            
            # Check for same activities between all pairs of diarists
            interval_has_same_activity = False
            activity_pairs = []
            
            for i, diarist1 in enumerate(diarists):
                for j, diarist2 in enumerate(diarists[i+1:], i+1):
                    activities1 = diarist_activities[diarist1]['all_activities']
                    activities2 = diarist_activities[diarist2]['all_activities']
                    
                    # Skip if no activities recorded
                    if not activities1 or not activities2:
                        continue
                    
                    # Find common activities
                    common_activities = activities1.intersection(activities2)
                    
                    if common_activities:
                        interval_has_same_activity = True
                        
                        # Determine activity type combination
                        for activity in common_activities:
                            type1 = []
                            type2 = []
                            
                            if diarist_activities[diarist1]['mact'] == activity:
                                type1.append('Mact')
                            if diarist_activities[diarist1]['sactn'] == activity:
                                type1.append('Sactn')
                            if diarist_activities[diarist2]['mact'] == activity:
                                type2.append('Mact')
                            if diarist_activities[diarist2]['sactn'] == activity:
                                type2.append('Sactn')
                            
                            for t1 in type1:
                                for t2 in type2:
                                    activity_pairs.append({
                                        'diarist1': diarist1,
                                        'diarist2': diarist2,
                                        'activity': activity,
                                        'type1': t1,
                                        'type2': t2,
                                        'combination': f"{t1}-{t2}"
                                    })
            
            if activity_pairs:
                same_activity_intervals += 1
                household_interval_details.extend([{
                    'HID': hid,
                    'DIARY': diary,
                    'interval': interval,
                    **pair
                } for pair in activity_pairs])
            
            total_comparable_intervals += 1
        
        # Calculate percentages
        if total_comparable_intervals > 0:
            same_activity_percentage = (same_activity_intervals / total_comparable_intervals) * 100
        else:
            same_activity_percentage = 0
        
        # Store household summary
        results['household_summary'].append({
            'HID': hid,
            'DIARY': diary,
            'n_diarists': n_diarists,
            'same_activity_intervals': same_activity_intervals,
            'total_intervals': total_comparable_intervals,
            'same_activity_percentage': same_activity_percentage
        })
        
        # Store interval details
        results['interval_details'].extend(household_interval_details)
    
    return results

def analyze_activity_combinations(interval_details):
    """Analyze the types of activity combinations (Mact-Mact, Mact-Sactn, etc.)"""
    if not interval_details:
        return {}
    
    details_df = pd.DataFrame(interval_details)
    
    # Count combination types
    combination_counts = details_df['combination'].value_counts()
    
    # Count unique activities that are shared
    shared_activities = details_df.groupby('activity').size().sort_values(ascending=False)
    
    return {
        'combination_counts': combination_counts,
        'shared_activities': shared_activities,
        'details_df': details_df
    }

def create_visualizations(results, output_dir=FIGURES_DIR):
    """Create visualizations for the same activity analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    household_summary = pd.DataFrame(results['household_summary'])
    
    # Plot 1: Distribution of same activity percentages across households
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(household_summary['same_activity_percentage'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Percentage of Intervals with Same Activities')
    plt.ylabel('Number of Households')
    plt.title('Distribution of Same Activity Coincidence')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Same activity percentage vs number of diarists
    plt.subplot(2, 2, 2)
    sns.boxplot(data=household_summary, x='n_diarists', y='same_activity_percentage')
    plt.xlabel('Number of Diarists')
    plt.ylabel('Same Activity Percentage')
    plt.title('Same Activities by Household Size')
    
    # Plot 3: Cumulative distribution of same activity percentages
    plt.subplot(2, 2, 3)
    sorted_percentages = np.sort(household_summary['same_activity_percentage'])
    y_vals = np.arange(1, len(sorted_percentages) + 1) / len(sorted_percentages)
    plt.plot(sorted_percentages, y_vals, linewidth=2)
    plt.xlabel('Same Activity Percentage')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Activity combination types (if available)
    plt.subplot(2, 2, 4)
    if results['interval_details']:
        combo_analysis = analyze_activity_combinations(results['interval_details'])
        combo_counts = combo_analysis['combination_counts']
        if not combo_counts.empty:
            plt.pie(combo_counts.values, labels=combo_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Activity Type Combinations')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/same_activity_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed plot: Top shared activities
    if results['interval_details']:
        combo_analysis = analyze_activity_combinations(results['interval_details'])
        shared_activities = combo_analysis['shared_activities']
        
        plt.figure(figsize=(12, 6))
        top_activities = shared_activities.head(15)
        plt.bar(range(len(top_activities)), top_activities.values)
        plt.xlabel('Activity Code')
        plt.ylabel('Frequency')
        plt.title('Top 15 Most Frequently Shared Activities')
        plt.xticks(range(len(top_activities)), top_activities.index, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_shared_activities.png", dpi=300, bbox_inches='tight')
        plt.show()

def print_detailed_analysis(results):
    """Print detailed analysis results"""
    household_summary = pd.DataFrame(results['household_summary'])
    
    print("\n" + "="*60)
    print("SAME ACTIVITY ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"Total multi-diarist households analyzed: {len(household_summary)}")
    print(f"Mean same activity percentage: {household_summary['same_activity_percentage'].mean():.2f}%")
    print(f"Median same activity percentage: {household_summary['same_activity_percentage'].median():.2f}%")
    print(f"Standard deviation: {household_summary['same_activity_percentage'].std():.2f}%")
    
    print(f"\nDistribution by number of diarists:")
    for n_diarists in sorted(household_summary['n_diarists'].unique()):
        subset = household_summary[household_summary['n_diarists'] == n_diarists]
        print(f"  {n_diarists} diarists: {len(subset)} households, "
              f"mean {subset['same_activity_percentage'].mean():.2f}% same activities")
    
    # Percentile analysis
    print(f"\nPercentile Analysis:")
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(household_summary['same_activity_percentage'], p)
        print(f"  {p}th percentile: {value:.2f}%")
    
    # Extreme cases
    print(f"\nExtreme Cases:")
    max_idx = household_summary['same_activity_percentage'].idxmax()
    min_idx = household_summary['same_activity_percentage'].idxmin()
    print(f"  Highest coincidence: {household_summary.loc[max_idx, 'same_activity_percentage']:.2f}% "
          f"(HID: {household_summary.loc[max_idx, 'HID']}, DIARY: {household_summary.loc[max_idx, 'DIARY']})")
    print(f"  Lowest coincidence: {household_summary.loc[min_idx, 'same_activity_percentage']:.2f}% "
          f"(HID: {household_summary.loc[min_idx, 'HID']}, DIARY: {household_summary.loc[min_idx, 'DIARY']})")
    
    # Activity combination analysis
    if results['interval_details']:
        combo_analysis = analyze_activity_combinations(results['interval_details'])
        print(f"\nActivity Combination Types:")
        for combo, count in combo_analysis['combination_counts'].items():
            percentage = (count / len(results['interval_details'])) * 100
            print(f"  {combo}: {count} occurrences ({percentage:.1f}%)")
        
        print(f"\nTop 5 Most Shared Activities:")
        for activity, count in combo_analysis['shared_activities'].head(5).items():
            print(f"  Activity {activity}: {count} occurrences")

def main():
    """Main function to run the same activity analysis"""
    print("Loading data...")
    
    # Load the data
    try:
        lu_data = pd.read_csv(PROCESSED_DIR / "lu_data_at_home_with_dominant_activity.csv")
        print(f"Loaded {len(lu_data)} rows from lu_data_at_home_with_dominant_activity.csv")
    except FileNotFoundError:
        print(f"Error: {PROCESSED_DIR}/lu_data_at_home_with_dominant_activity.csv not found!")
        return
    
    # Basic data info
    print(f"Total diarists (unique PID): {lu_data['PID'].nunique()}")
    print(f"Total households (unique HID): {lu_data['HID'].nunique()}")
    
    # Count diarists per household for context
    diarists_per_household = lu_data.groupby(['HID', 'DIARY'])['PID'].nunique()
    multi_diarist_households = diarists_per_household[diarists_per_household > 1]
    print(f"Multi-diarist household-days: {len(multi_diarist_households)}")
    
    # Run the analysis
    results = analyze_same_activity_diarists(lu_data)
    
    # Print results
    print_detailed_analysis(results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results)
    
    # Save detailed results to CSV
    print("\nSaving detailed results to CSV files...")
    
    household_summary_df = pd.DataFrame(results['household_summary'])
    household_summary_df.to_csv(PROCESSED_DIR / "same_activity_household_summary.csv", index=False)
    print(f"Saved household summary to same_activity_household_summary.csv")
    
    if results['interval_details']:
        interval_details_df = pd.DataFrame(results['interval_details'])
        interval_details_df.to_csv(PROCESSED_DIR / "same_activity_interval_details.csv", index=False)
        print(f"Saved interval details to same_activity_interval_details.csv")
        
        # Also save combination analysis
        combo_analysis = analyze_activity_combinations(results['interval_details'])
        combo_analysis['combination_counts'].to_csv(PROCESSED_DIR / "activity_combination_counts.csv", header=['count'])
        combo_analysis['shared_activities'].to_csv(PROCESSED_DIR / "shared_activities_frequency.csv", header=['frequency'])
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
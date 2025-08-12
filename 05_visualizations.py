import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data: paste your CSV data here as string, or load from file
balance_before_csv = pd.read_csv('data/balance_before_matching.csv')
balance_after_csv = pd.read_csv('data/balance_after_matching.csv')

# Compute absolute SMD for before and after
balance_before_csv['abs_smd_before'] = balance_before_csv['smd'].abs()
balance_after_csv['abs_smd_after'] = balance_after_csv['smd'].abs()

# Merge by variable
df = pd.merge(balance_before_csv[['variable', 'abs_smd_before']],
              balance_after_csv[['variable', 'abs_smd_after']],
              on='variable')

# Sort variables by abs_smd_before descending for better visualization order
df = df.sort_values('abs_smd_before', ascending=False)

# Plot
plt.figure(figsize=(12, 10))
sns.set_style('whitegrid')

bar_width = 0.4
indices = np.arange(len(df))

plt.barh(indices - bar_width/2, df['abs_smd_before'], height=bar_width, color='salmon', label='Before Matching')
plt.barh(indices + bar_width/2, df['abs_smd_after'], height=bar_width, color='seagreen', label='After Matching')

plt.yticks(indices, df['variable'])
plt.xlabel('Absolute Standardized Mean Difference (|SMD|)')
plt.title('Covariate Absolute SMD Improvement Before and After Matching')

plt.legend()
plt.gca().invert_yaxis()  # Highest imbalance on top
plt.tight_layout()
plt.savefig('visualizations/SMD_balance_improvement.png', dpi=300, bbox_inches='tight')
plt.show()


# Calculate overall balance improvement
overall_before = df['abs_smd_before'].mean()
overall_after = df['abs_smd_after'].mean()
improvement = overall_before - overall_after

# Plot overall balance improvement
plt.figure(figsize=(8, 6))
plt.bar(['Before Matching', 'After Matching'], [overall_before, overall_after], color=['salmon', 'seagreen'])
plt.ylabel('Mean Absolute SMD')
plt.title('Overall Balance Improvement Before and After Matching')
plt.ylim(0, max(overall_before, overall_after) * 1.1)
plt.tight_layout()
plt.savefig('visualizations/overall_balance_improvement.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize treatment effect 
treated = 0.5083
control = 0.4457
effect = 0.0626
ci_lower = 0.0286
ci_upper = 0.0966

# X positions for bars
x = [0, 1]
outcomes = [control, treated]

plt.figure(figsize=(6,5))

# Plot bars for control and treated
plt.bar(x, outcomes, color=['salmon', 'seagreen'], width=0.6)
plt.xticks(x, ['Control Outcome', 'Treated Outcome'])

plt.ylabel('Mental Health Score')
plt.title('Treatment Effect on Mental Health')
plt.tight_layout()
plt.savefig('visualizations/treatment_effect.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize IPW dose-response curve
dose_response = pd.read_csv('data/ipw_dose_response_results.csv')

# For visualization, use mid-point of each exercise bin as the x-axis
def get_bin_midpoint(bin_str):
    # Assumes bins like '0.0-59.0'
    start, end = bin_str.split('-')
    return (float(start) + float(end)) / 2

dose_response['bin_midpoint'] = dose_response['EXERCISE_MINUTES_BINS'].apply(get_bin_midpoint)

# Filter the data - Between 30 minutes and 500 minutes
dose_response = dose_response[(dose_response['bin_midpoint'] >= 30) & (dose_response['bin_midpoint'] <= 500)]

# Plot mean with confidence intervals
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(dose_response['bin_midpoint'], dose_response['mean'],
            fmt='o', ecolor='gray', elinewidth=2, capsize=5, color='seagreen')

# Join the points with a line
ax.plot(dose_response['bin_midpoint'], dose_response['mean'], color='seagreen', linestyle='-', alpha=0.8)
ax.set_xlabel('Exercise Minutes/week')
ax.set_ylabel('Average Mental Health Score')
ax.set_title('Exercise Minutes/week vs. Mental Health Score (Up to 475 minutes)')
ax.set_xlim([50, 475])
plt.tight_layout()
plt.savefig('visualizations/ipw_dose_response.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot heterogeneity of treatment effect
het_df = pd.read_csv('data/ipw_heterogeneous_effects_results.csv')

het_df = het_df[het_df['subgroup'] == 'Marital Status']

# Plot
plt.figure(figsize=(10,6))
sns.barplot(
    data=het_df,
    y="category", x="effect",
    palette="Set2",
    dodge=False
)

plt.axvline(0, color="black", linewidth=1)
plt.title("Treatment Effect by Marital Status", fontsize=14)
plt.xlabel("Estimated Effect")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig('visualizations/ipw_heterogeneous_effects.png', dpi=300, bbox_inches='tight')
plt.show()
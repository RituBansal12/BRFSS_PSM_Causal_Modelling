import pandas as pd
import numpy as np
from psmpy import PsmPy
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('data/cleaned_llcp2023.csv')

# Add unique identifier
if '_SEQNO' not in df.columns:
    df['_SEQNO'] = range(len(df))

# Define treatment and outcome
W_cols = ['_AGEG5YR', '_SEX', '_EDUCAG', '_INCOMG1', 'MARITAL', '_RACEGR3', '_RFHLTH', '_BMI5CAT', 'ADDEPEV3', 'CHCCOPD3', 'DIABETE4', 'HAVARTH4', 'SDHSTRE1', 'EMTSUPRT', 'SDLONELY', 'FOODSTMP', 'SDHBILLS', 'SDHUTILS', 'SDHTRNSP']
treatment_var = '_EXERCISE_BINARY'
outcome_var = '_MENTAL_HEALTH_BINARY'

# Filter for complete cases
print(f"Rows before filtering for complete cases: {df.shape[0]}")
df_filtered_psm = df.dropna(subset=[treatment_var, outcome_var] + W_cols).copy()
print(f"Rows after filtering for complete cases: {df_filtered_psm.shape[0]}")

# Identify columns causing most drops
missing_counts = df[W_cols].isna().sum().sort_values(ascending=False)
print("Missing value counts for W_cols:")
print(missing_counts)

# Ensure all covariates are numeric
df_filtered_psm[W_cols] = df_filtered_psm[W_cols].apply(pd.to_numeric, errors='coerce')

# Initialize PsmPy
psm = PsmPy(df_filtered_psm, treatment=treatment_var, indx='_SEQNO', exclude=[outcome_var])
psm.logistic_ps(balance=True)

print("Covariate balance BEFORE matching (Cohen's D):")
print(psm.effect_size.head(10))

plt.figure(figsize=(10, 6))
psm.plot_ps(cmap='viridis')
plt.title('Propensity Score Distribution Before Matching')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.show()

psm.knn_matched(matcher='propensity_logit', how_many=1, replacement=False, caliper=0.2)

print("Covariate balance AFTER matching (Cohen's D):")
print(psm.effect_size.head(10))

plt.figure(figsize=(10, 6))
psm.plot_ps(cmap='viridis')
plt.title('Propensity Score Distribution After Matching')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.show()

df_matched = psm.df_matched.copy()
df_for_causal_model = df_filtered_psm.merge(df_matched, on='_SEQNO', how='inner')
print(f"DataFrame prepared for causal modeling (after PSM): {df_for_causal_model.shape}")
df_for_causal_model.to_csv('data/matched_llcp2023.csv', index=False)

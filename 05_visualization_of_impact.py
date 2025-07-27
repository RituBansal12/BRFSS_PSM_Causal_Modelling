import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df_results = pd.read_csv('data/results_with_cate.csv')

plt.figure(figsize=(10, 6))
sns.histplot(df_results['CATE_Exercise_MentalHealth'], kde=True, bins=50)
plt.title('Distribution of Estimated Causal Effects of Exercise on Mental Health')
plt.xlabel('Estimated Causal Effect (CATE)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='_AGEG5YR', y='CATE_Exercise_MentalHealth', data=df_results)
plt.title('Causal Effect of Exercise on Mental Health by Age Group')
plt.xlabel('Age Group (Categorical Code)')
plt.ylabel('Estimated Causal Effect (CATE)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='_INCOMG1', y='CATE_Exercise_MentalHealth', data=df_results)
plt.title('Causal Effect of Exercise on Mental Health by Income Level')
plt.xlabel('Income Level (Categorical Code)')
plt.ylabel('Estimated Causal Effect (CATE)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

correlation_vars_for_heatmap = ['_AGEG5YR', '_SEX', '_EDUCAG', '_INCOMG1', 'SDHSTRE1', 'EMTSUPRT', 'SDLONELY', 'CATE_Exercise_MentalHealth']
df_heatmap_data = df_results[correlation_vars_for_heatmap].copy()
for col in df_heatmap_data.columns:
    if pd.api.types.is_categorical_dtype(df_heatmap_data[col]):
        df_heatmap_data[col] = df_heatmap_data[col].cat.codes
correlation_matrix = df_heatmap_data.corr(numeric_only=True)
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Covariates and Causal Effect Estimates')
plt.show()

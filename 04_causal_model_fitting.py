import pandas as pd
import numpy as np
from econml.orf import DMLOrthoForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load matched data
df_for_causal_model = pd.read_csv('data/matched_llcp2023.csv')

Y = df_for_causal_model['_MENTAL_HEALTH_BINARY']
T = df_for_causal_model['_EXERCISE_BINARY']
W_cols = ['_AGEG5YR', '_SEX', '_EDUCAG', '_INCOMG1', 'MARITAL', '_RACEGR3', '_RFHLTH', '_BMI5CAT', 'ADDEPEV3', 'CHCCOPD3', 'DIABETE4', 'HAVARTH4', 'SDHSTRE1', 'EMTSUPRT', 'SDLONELY', 'FOODSTMP', 'SDHBILLS', 'SDHUTILS', 'SDHTRNSP']
W = pd.get_dummies(df_for_causal_model[W_cols], dummy_na=False)
X = W.copy()

est = DMLOrthoForest(
    model_Y=RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42),
    model_T=RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42),
    n_estimators=1000,
    min_samples_leaf=5,
    max_depth=10,
    inference='bootstrap',
    random_state=42
)

print("Fitting Causal Forest model...")
est.fit(Y, T, X=X, W=W)
print("Causal Forest model fitted.")

cate_estimates = est.effect(X)
cate_lower, cate_upper = est.effect_interval(X, alpha=0.05)

df_for_causal_model['CATE_Exercise_MentalHealth'] = cate_estimates
df_for_causal_model['CATE_Lower_Bound'] = cate_lower
df_for_causal_model['CATE_Upper_Bound'] = cate_upper

print(df_for_causal_model[['CATE_Exercise_MentalHealth', 'CATE_Lower_Bound', 'CATE_Upper_Bound']].head())
df_for_causal_model.to_csv('data/results_with_cate.csv', index=False)

#!/usr/bin/env python3
"""
Phase 4: SHAP explainability for NightSafe
Computes SHAP values and generates plain-language explanations for every street
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED = 'Data/processed'
REASONS_CSV = DATA_PROCESSED + '/shap_reasons_by_street.csv'

feature_columns = [
    'length_km',
    'lighting_score',
    'foot_traffic_score',
    'pole_density',
    'distance_to_help_km',
    'control_density',
    'traffic_volume_log',
    'has_traffic_data',
]

friendly = {
    'length_km': 'street length',
    'lighting_score': 'lighting level',
    'foot_traffic_score': 'foot traffic',
    'pole_density': 'street lighting density',
    'distance_to_help_km': 'distance from help',
    'control_density': 'intersection controls',
    'traffic_volume_log': 'vehicle traffic volume',
    'has_traffic_data': 'monitored-road status',
}

print("Loading model and data")
rf = joblib.load(DATA_PROCESSED + '/model_random_forest.pkl')
df = pd.read_csv(DATA_PROCESSED + '/corridor_forecast.csv')

X = df[feature_columns].copy()
for col in feature_columns:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

try:
    import shap
except ImportError:
    raise SystemExit("SHAP not installed. Run: pip install shap")

print("Computing SHAP values")
explainer = shap.TreeExplainer(rf)

# Global importance
sample = X.sample(min(2000, len(X)), random_state=42)
shap_sample = explainer.shap_values(sample)
global_imp = pd.DataFrame({
    'feature': feature_columns,
    'mean_abs_shap': np.abs(shap_sample).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nGlobal SHAP importance:")
print(global_imp.to_string(index=False))
global_imp.to_csv(DATA_PROCESSED + '/shap_global_importance.csv', index=False)

# Per-segment explanations
print("\nComputing per-segment SHAP reasons")
all_shap = explainer.shap_values(X)

def top_reasons_for_row(shap_row):
    pairs = sorted(zip(feature_columns, shap_row), key=lambda p: abs(p[1]), reverse=True)
    parts = []
    for feat, val in pairs[:3]:
        direction = "raises" if val > 0 else "lowers"
        parts.append(f"{friendly.get(feat, feat)} {direction} risk")
    return "; ".join(parts)

df_reasons = df[['name', 'risk_index', 'severity']].copy()
df_reasons['reason'] = [top_reasons_for_row(all_shap[i]) for i in range(len(all_shap))]
df_reasons['name'] = (df_reasons['name'].astype(str)
                      .str.replace(r"^\['?|'?\]$", "", regex=True)
                      .str.replace("'", "").str.strip())

streets = df_reasons.drop_duplicates('name').reset_index(drop=True)

def plain_fallback(reason_text):
    if not reason_text or str(reason_text) == 'nan':
        return "This corridor's risk reflects its overall street characteristics."
    return "This corridor scored as it did because " + str(reason_text) + "."

# Reuse previous explanations if available
prev_expl = {}
if os.path.exists(REASONS_CSV):
    prev = pd.read_csv(REASONS_CSV)
    if 'explanation' in prev.columns:
        prev_expl = dict(zip(prev['name'].astype(str), prev['explanation'].astype(str)))

explanation_col = []
for _, row in streets.iterrows():
    nm = str(row['name'])
    if nm in prev_expl and str(prev_expl[nm]) != 'nan':
        explanation_col.append(str(prev_expl[nm]))
    else:
        explanation_col.append(plain_fallback(row['reason']))

streets['explanation'] = pd.Series(explanation_col, dtype='object')
streets[['name', 'reason', 'explanation']].to_csv(REASONS_CSV, index=False)

print(f"\nSaved explanations for {len(streets):,} streets")
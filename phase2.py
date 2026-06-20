#!/usr/bin/env python3
"""
Phase 2: Risk model for NightSafe
Predicts future crime risk from structural street features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED = 'Data/processed'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
MIN_LENGTH_KM = 0.02

# Load the feature table
df = pd.read_csv(DATA_PROCESSED + '/feature_table_temporal.csv')
print(f"Loaded {len(df):,} road segments")

# Only structural features - no incident data
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

# Sanity check: make sure no target features are in the inputs
for f in feature_columns:
    if 'incident' in f.lower():
        raise ValueError(f"Feature {f} contains target information")

# Remove tiny segments that distort per-km rates
before = len(df)
df = df[df['length_km'] >= MIN_LENGTH_KM].copy()
print(f"Dropped {before - len(df):,} segments under {MIN_LENGTH_KM*1000:.0f} meters")

# Build target: average of 2025 and 2026 incidents per km
def density(year_col):
    d = df[year_col] / df['length_km']
    return d.clip(upper=d.quantile(0.99))

df['target_density'] = (density('incidents_2025') + density('incidents_2026')) / 2.0
df['target_log'] = np.log1p(df['target_density'])

# Prepare features and target
X = df[feature_columns].copy()
for col in feature_columns:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].median())
y = df['target_log']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train):,}  Test set: {len(X_test):,}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline: median by road type
train_df = df.loc[X_train.index]
highway_medians = train_df.groupby('highway')['target_log'].median()
global_median = train_df['target_log'].median()
baseline_pred = df.loc[X_test.index, 'highway'].map(highway_medians).fillna(global_median).values

# Train models
models = {}

models['Random Forest'] = RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_split=10,
    min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
)

models['XGBoost'] = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE
)

models['LightGBM'] = lgb.LGBMRegressor(
    n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
    random_state=RANDOM_STATE, verbose=-1
)

models['Neural Network'] = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
    alpha=0.001, batch_size=256, max_iter=300, early_stopping=True,
    validation_fraction=0.1, random_state=RANDOM_STATE
)

preds = {}
for name, model in models.items():
    if name == 'Neural Network':
        model.fit(X_train_scaled, y_train)
        preds[name] = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
    print(f"Trained {name}")

# Ensemble
ensemble_weights = {'Random Forest': 0.35, 'XGBoost': 0.25, 'LightGBM': 0.30, 'Neural Network': 0.10}
ensemble_pred = sum(ensemble_weights[n] * preds[n] for n in models)

# Evaluation
def recall_at_top(y_true, y_pred, k=0.10):
    n = int(len(y_true) * k)
    true_top = set(np.argsort(y_true.values)[-n:])
    pred_top = set(np.argsort(y_pred)[-n:])
    return len(true_top & pred_top) / n

def metrics_row(name, y_true, y_pred):
    return {
        'Model': name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Recall@10%': recall_at_top(y_true, y_pred, 0.10),
    }

rows = [metrics_row('Baseline', y_test, baseline_pred)]
for name in models:
    rows.append(metrics_row(name, y_test, preds[name]))
rows.append(metrics_row('Ensemble', y_test, ensemble_pred))
results = pd.DataFrame(rows)

print("\nModel performance:")
print(results.to_string(index=False))

# Refit on all data and forecast forward
full_preds = {}
for name, model in models.items():
    if name == 'Neural Network':
        model.fit(scaler.fit_transform(X), y)
        full_preds[name] = model.predict(scaler.transform(X))
    else:
        model.fit(X, y)
        full_preds[name] = model.predict(X)

stack = np.vstack([full_preds[n] for n in models])
forecast_mean = stack.mean(axis=0)
forecast_std = stack.std(axis=0)

df['risk_forecast'] = np.expm1(forecast_mean)
df['risk_low'] = np.expm1(forecast_mean - forecast_std)
df['risk_high'] = np.expm1(forecast_mean + forecast_std)

# Convert to 0-100 risk index
df['risk_index'] = (df['risk_forecast'].rank(pct=True) * 100).round(1)

def severity(idx):
    if idx >= 90:
        return 'severe'
    if idx >= 70:
        return 'high'
    if idx >= 40:
        return 'moderate'
    return 'low'

df['severity'] = df['risk_index'].apply(severity)
print("\nRisk severity distribution:")
print(df['severity'].value_counts().to_string())

# Fairness audit by income quartile
df['actual_high'] = (df['target_density'] >= df['target_density'].quantile(0.90)).astype(int)
df['flagged_high'] = df['severity'].isin(['severe', 'high']).astype(int)

audit = df.groupby('income_quartile').agg(
    segments=('edge_id', 'count'),
    actual_high_rate=('actual_high', 'mean'),
    flagged_high_rate=('flagged_high', 'mean'),
)
audit['parity_gap'] = audit['flagged_high_rate'] - audit['actual_high_rate']

print("\nFairness audit by income quartile:")
print(audit.round(3))

max_gap = audit['parity_gap'].abs().max()
if max_gap > 0.10:
    print(f"\nWarning: Parity gap of {max_gap:.3f} exceeds threshold")
else:
    print(f"\nParity gaps within acceptable range ({max_gap:.3f})")

# Flag for human review
df['needs_human_review'] = (
    df['severity'].isin(['severe', 'high']) & (df['income_quartile'] == 1)
).astype(int)
review_count = int(df['needs_human_review'].sum())
print(f"Human review needed for {review_count:,} high-risk segments in lowest income quartile")

# Feature importance
rf_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': models['Random Forest'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance (Random Forest):")
print(rf_importance.to_string(index=False))

# Save outputs
df.to_csv(DATA_PROCESSED + '/corridor_forecast.csv', index=False)
results.to_csv(DATA_PROCESSED + '/model_metrics.csv', index=False)
audit.to_csv(DATA_PROCESSED + '/fairness_audit.csv')
rf_importance.to_csv(DATA_PROCESSED + '/feature_importance.csv', index=False)

for name, model in models.items():
    safe = name.lower().replace(' ', '_')
    joblib.dump(model, DATA_PROCESSED + f'/model_{safe}.pkl')
joblib.dump(scaler, DATA_PROCESSED + '/scaler.pkl')
joblib.dump(ensemble_weights, DATA_PROCESSED + '/ensemble_weights.pkl')

print("\nAll outputs saved to Data/processed/")
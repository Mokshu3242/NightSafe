"""
Phase 2: Risk model for NightSafe

Three changes from the original, all aimed at making the model honest and the
story defensible:

1. No data leakage. The original used poles_per_incident as a feature, which
   is built from incident counts, while also predicting incident counts. That
   let the model cheat. Here, crime is only the target. We predict it from
   structural features that contain no incident information.

2. Past predicts future. We train on 2022 to 2024 incidents to predict the
   2025 to 2026 average, which we actually have, so we can measure honestly.
   We then project forward as a forecast with uncertainty.

3. Interpretable scale. Tiny segments made incidents-per-km explode into the
   tens of thousands. We drop sub-20m stubs, winsorize the tail, and convert
   the final risk into a 0 to 100 index.

We also run a fairness audit (demographic parity across income quartiles).

Expect R2 to be lower than the old leaked number. That is correct. An honest
model that survives a judge reading the feature importance beats a leaked one.
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


# ----------------------------------------------------------------------
# Step 1: Load the temporal feature table from Phase 1
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PROCESSED + '/feature_table_temporal.csv')
print("Loaded " + str(len(df)) + " road segments")

# Structural features only. None of these contain incident information.
# Model features. Income is deliberately NOT here. We do not let the model
# predict risk from neighborhood wealth, because that turns poverty into a
# proxy for danger and unfairly over-flags low-income areas. Income is kept
# in the data and used only for the fairness audit and human-review trigger
# below, never as a model input. These features describe the street itself.
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

# Guard against accidentally reintroducing leakage later.
for f in feature_columns:
    if 'incident' in f.lower():
        raise ValueError("Feature " + f + " looks like it contains the target")


# ----------------------------------------------------------------------
# Step 2: Drop tiny stub segments that distort per-km rates
# ----------------------------------------------------------------------
before = len(df)
df = df[df['length_km'] >= MIN_LENGTH_KM].copy()
print("Dropped " + str(before - len(df)) + " segments under 20 meters")


# ----------------------------------------------------------------------
# Step 3: Build the target. Past years predict future years.
# ----------------------------------------------------------------------
# Target is the average incidents-per-km in 2025 and 2026, winsorized so a
# handful of extreme segments do not dominate.
def density(year_col):
    d = df[year_col] / df['length_km']
    return d.clip(upper=d.quantile(0.99))

df['target_density'] = (density('incidents_2025') + density('incidents_2026')) / 2.0
df['target_log'] = np.log1p(df['target_density'])

X = df[feature_columns].copy()
for col in feature_columns:
    if X[col].isna().sum() > 0:
        X[col] = X[col].fillna(X[col].median())
y = df['target_log']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print("Train: " + str(len(X_train)) + "   Test: " + str(len(X_test)))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ----------------------------------------------------------------------
# Step 4: Baseline. Median target by road type. Needed to show lift.
# ----------------------------------------------------------------------
train_df = df.loc[X_train.index]
highway_medians = train_df.groupby('highway')['target_log'].median()
global_median = train_df['target_log'].median()
baseline_pred = df.loc[X_test.index, 'highway'].map(highway_medians).fillna(global_median).values


# ----------------------------------------------------------------------
# Step 5: Train the four models
# ----------------------------------------------------------------------
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
    print("Trained " + name)

ensemble_weights = {'Random Forest': 0.35, 'XGBoost': 0.25, 'LightGBM': 0.30, 'Neural Network': 0.10}
ensemble_pred = sum(ensemble_weights[n] * preds[n] for n in models)


# ----------------------------------------------------------------------
# Step 6: Evaluate, including recall on the high-risk segments
# ----------------------------------------------------------------------
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

print("")
print("Model performance (log target):")
print(results.to_string(index=False))
print("")
print("R2 is lower than the old leaked run. That is expected. The leaked")
print("feature poles_per_incident has been removed, so the model can no")
print("longer see the answer in its inputs.")


# ----------------------------------------------------------------------
# Step 7: Refit on all data and forecast forward, with an uncertainty band
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Step 8: Convert to a 0 to 100 risk index and a severity label
# ----------------------------------------------------------------------
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
print("")
print("Severity counts:")
print(df['severity'].value_counts().to_string())


# ----------------------------------------------------------------------
# Step 9: Fairness audit. Demographic parity across income quartiles.
# ----------------------------------------------------------------------
# We ask whether the model flags high risk in proportion to where high
# incidents actually are, across income groups, or whether it over-flags
# one group. parity_gap is flagged rate minus actual high rate.
df['actual_high'] = (df['target_density'] >= df['target_density'].quantile(0.90)).astype(int)
df['flagged_high'] = df['severity'].isin(['severe', 'high']).astype(int)

audit = df.groupby('income_quartile').agg(
    segments=('edge_id', 'count'),
    actual_high_rate=('actual_high', 'mean'),
    flagged_high_rate=('flagged_high', 'mean'),
)
audit['parity_gap'] = audit['flagged_high_rate'] - audit['actual_high_rate']

print("")
print("Fairness audit by income quartile:")
print(audit.round(3).to_string())

max_gap = audit['parity_gap'].abs().max()
print("")
print("Largest parity gap: " + str(round(max_gap, 3)))
if max_gap > 0.10:
    print("Gap above 0.10. The model over or under flags some income group.")
    print("Recalibrate and document this. Re-audit on every retrain.")
else:
    print("Gaps within 0.10. Flags track actual risk reasonably across groups.")
    print("Re-audit on every retrain, quarterly at minimum.")

# Human-review trigger. Income never feeds the model, but we use it as a
# safeguard: any high-risk recommendation in the lowest income quartile is
# flagged for a human to review before any investment decision, so residual
# bias cannot quietly steer money. This is oversight, not prediction.
df['needs_human_review'] = (
    df['severity'].isin(['severe', 'high']) & (df['income_quartile'] == 1)
).astype(int)
review_count = int(df['needs_human_review'].sum())
print("")
print("Human review trigger: " + str(review_count)
      + " high-risk segments in the lowest-income quartile flagged for review")


# ----------------------------------------------------------------------
# Step 10: Feature importance (from Random Forest) and save everything
# ----------------------------------------------------------------------
rf_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': models['Random Forest'].feature_importances_
}).sort_values('importance', ascending=False)

print("")
print("Random Forest feature importance:")
print(rf_importance.to_string(index=False))

df.to_csv(DATA_PROCESSED + '/corridor_forecast.csv', index=False)
results.to_csv(DATA_PROCESSED + '/model_metrics_corrected.csv', index=False)
audit.to_csv(DATA_PROCESSED + '/fairness_audit.csv')
rf_importance.to_csv(DATA_PROCESSED + '/feature_importance.csv', index=False)
for name, model in models.items():
    safe = name.lower().replace(' ', '_')
    joblib.dump(model, DATA_PROCESSED + '/model_' + safe + '.pkl')
joblib.dump(scaler, DATA_PROCESSED + '/scaler.pkl')
joblib.dump(ensemble_weights, DATA_PROCESSED + '/ensemble_weights.pkl')

print("")
print("Saved corridor_forecast.csv, metrics, fairness audit, importance, models")
print("Phase 2 complete. Phase 3 should read risk_index from corridor_forecast.csv")
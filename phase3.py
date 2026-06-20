#!/usr/bin/env python3
"""
Phase 3: Cost of Doing Nothing simulator for NightSafe
Evaluates intervention strategies for high-risk corridors
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED = 'Data/processed'

# Load the forecast from Phase 2
features = pd.read_csv(DATA_PROCESSED + '/corridor_forecast.csv')
print(f"Loaded {len(features):,} road segments")
print(f"Risk index range: {features['risk_index'].min():.1f} to {features['risk_index'].max():.1f}")

class CostOfDoingNothingSimulator:
    
    def __init__(self):
        self.intervention_effects = {
            'do_nothing': 1.00,
            'add_lighting': 0.55,
            'add_lighting_and_booth': 0.40,
            'add_patrols': 0.50,
            'full_intervention': 0.30,
        }
        self.compounding_rate = {'low': 0.03, 'medium': 0.08, 'high': 0.15}
        self.incident_costs = {
            'minor': (5000, 20000),
            'moderate': (50000, 150000),
            'severe': (200000, 500000),
        }
        self.intervention_costs = {
            'add_lighting': (50000, 150000),
            'add_lighting_and_booth': (100000, 250000),
            'add_patrols': (150000, 400000),
            'full_intervention': (300000, 700000),
        }
        self.severity_thresholds = {'moderate': 40, 'high': 70, 'severe': 90}
    
    def get_compounding_rate(self, risk_score):
        if risk_score >= self.severity_thresholds['high']:
            return self.compounding_rate['high']
        elif risk_score >= self.severity_thresholds['moderate']:
            return self.compounding_rate['medium']
        return self.compounding_rate['low']
    
    def get_severity_category(self, risk_score):
        if risk_score >= self.severity_thresholds['severe']:
            return 'severe'
        elif risk_score >= self.severity_thresholds['high']:
            return 'high'
        elif risk_score >= self.severity_thresholds['moderate']:
            return 'moderate'
        return 'low'
    
    def get_cost_per_incident(self, severity):
        if severity == 'severe':
            return self.incident_costs['severe']
        elif severity in ('high', 'moderate'):
            return self.incident_costs['moderate']
        return self.incident_costs['minor']
    
    def project_incidents(self, current_risk, length_km, years, action):
        effect = self.intervention_effects.get(action, 1.00)
        comp_rate = self.get_compounding_rate(current_risk) if action == 'do_nothing' else 0
        base_annual = max(current_risk * effect / 100, 0.1)
        
        total = 0
        for year in range(1, years + 1):
            year_effect = (1 + comp_rate) ** (year - 1) if comp_rate > 0 else 1
            total += base_annual * year_effect * length_km
        
        uncertainty = 0.20
        return (max(0, int(total * (1 - uncertainty))), int(total * (1 + uncertainty)))
    
    def project_costs(self, incidents_range, severity):
        low, high = self.get_cost_per_incident(severity)
        return (int(incidents_range[0] * low), int(incidents_range[1] * high))
    
    def get_intervention_cost(self, action, length_km):
        low, high = self.intervention_costs.get(action, (50000, 150000))
        return (int(low * length_km), int(high * length_km))
    
    def simulate_corridor(self, row, years=5):
        risk = row['risk_index']
        length = row['length_km']
        severity = self.get_severity_category(risk)
        
        actions = ['do_nothing', 'add_lighting', 'add_lighting_and_booth',
                   'add_patrols', 'full_intervention']
        results = {}
        
        for action in actions:
            incidents = self.project_incidents(risk, length, years, action)
            cost = self.project_costs(incidents, severity)
            if action != 'do_nothing':
                inv = self.get_intervention_cost(action, length)
                net = (cost[0] - inv[1], cost[1] - inv[0])
            else:
                inv = (0, 0)
                net = cost
            results[action] = {'incidents': incidents, 'cost': cost,
                               'intervention_cost': inv, 'net_savings': net}
        
        if severity == 'severe':
            best = 'full_intervention'
        elif severity == 'high':
            best = 'add_patrols'
        elif severity == 'moderate':
            best = 'add_lighting_and_booth'
        else:
            best = 'add_lighting'
        
        prevented_min = results['do_nothing']['incidents'][0] - results[best]['incidents'][1]
        prevented_max = results['do_nothing']['incidents'][1] - results[best]['incidents'][0]
        
        results['summary'] = {
            'risk_score': risk,
            'severity': severity,
            'length_km': length,
            'prevented_incidents': (max(0, prevented_min), max(0, prevented_max)),
            'recommended_action': best,
        }
        return results

simulator = CostOfDoingNothingSimulator()
print("Simulator ready")

# Remove duplicates before simulation
def clean_name(x):
    if isinstance(x, (list, np.ndarray)):
        return x[0] if len(x) > 0 else 'Unknown'
    if pd.isna(x):
        return 'Unknown'
    return str(x)

features['name'] = features['name'].apply(clean_name)
features = features.drop_duplicates(subset='edge_id')
print(f"Unique segments to simulate: {len(features):,}")

# Run simulation
rows = []
for idx, row in features.iterrows():
    try:
        res = simulator.simulate_corridor(row)
        s = res['summary']
        rows.append({
            'edge_id': row['edge_id'],
            'risk_score': s['risk_score'],
            'severity': s['severity'],
            'prevented_incidents_min': s['prevented_incidents'][0],
            'prevented_incidents_max': s['prevented_incidents'][1],
            'recommended_action': s['recommended_action'],
        })
    except Exception as e:
        rows.append({
            'edge_id': row['edge_id'], 'risk_score': 0, 'severity': 'low',
            'prevented_incidents_min': 0, 'prevented_incidents_max': 0,
            'recommended_action': 'add_lighting',
        })

sim_df = pd.DataFrame(rows)
features = features.drop(columns=['severity'], errors='ignore')
features = features.merge(sim_df, on='edge_id', how='left')
features['priority_score'] = features['prevented_incidents_max']
print(f"Completed {len(sim_df):,} simulations")

# Aggregate by street name
def most_common(s):
    return s.mode().iloc[0] if not s.mode().empty else s.iloc[0]

corridors = features.groupby('name').agg(
    segments=('edge_id', 'count'),
    length_km=('length_km', 'sum'),
    highway=('highway', most_common),
    predicted_risk=('risk_score', 'mean'),
    risk_low=('risk_low', 'mean'),
    risk_high=('risk_high', 'mean'),
    severity=('severity', lambda s: max(s, key=lambda v:
              ['low', 'moderate', 'high', 'severe'].index(v))),
    prevented_incidents_min=('prevented_incidents_min', 'sum'),
    prevented_incidents_max=('prevented_incidents_max', 'sum'),
    recommended_action=('recommended_action', most_common),
    reported_crime=('incident_count', 'sum'),
).reset_index()

corridors = corridors[corridors['name'] != 'Unknown']
corridors['priority_score'] = corridors['prevented_incidents_max']
corridors = corridors.sort_values('priority_score', ascending=False)
print(f"Aggregated into {len(corridors):,} distinct streets")

# Display top corridors
print("\nTop priority corridors for investment:")
for rank, (idx, row) in enumerate(corridors.head(10).iterrows(), 1):
    name = clean_name(row['name'])[:28]
    prevented = f"{int(row['prevented_incidents_min'])}-{int(row['prevented_incidents_max'])}"
    action = str(row['recommended_action']).replace('_', ' ').title()
    print(f"{rank:2d}. {name:28s} risk {int(row['predicted_risk']):3d}  {action:24s} prevented {prevented}")

# Save outputs
output_columns = [
    'name', 'highway', 'segments', 'length_km',
    'predicted_risk', 'risk_low', 'risk_high', 'priority_score', 'severity',
    'prevented_incidents_min', 'prevented_incidents_max', 'recommended_action',
    'reported_crime',
]
output_columns = [c for c in output_columns if c in corridors.columns]
corridors[output_columns].to_csv(DATA_PROCESSED + '/corridor_rankings.csv', index=False)
print("\nSaved corridor_rankings.csv")

summary = {
    'total_corridors': len(corridors),
    'high_or_severe': int((corridors['severity'].isin(['high', 'severe'])).sum()),
    'severe': int((corridors['severity'] == 'severe').sum()),
    'preventable_min': int(corridors['prevented_incidents_min'].sum()),
    'preventable_max': int(corridors['prevented_incidents_max'].sum()),
}
pd.DataFrame([summary]).to_csv(DATA_PROCESSED + '/simulator_summary.csv', index=False)
print("Saved simulator_summary.csv")

corridors.head(100)[output_columns].to_json(
    DATA_PROCESSED + '/top_100_corridors.json', orient='records', indent=2
)
print("Saved top_100_corridors.json")
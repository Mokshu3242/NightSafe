"""
Phase 1: Build the feature table for NightSafe with Weather Integration.

Produces one row per Philadelphia road segment with the features the model
uses, plus per-year incident counts (2022 to 2026) so Phase 2 can train on
past years and forecast future ones.

Income is real. It comes from Census ACS median household income per tract,
joined spatially to each road. Run fetch_census_income.py first so that
Data/raw/tract_income.csv exists.

Weather data is integrated from NOAA Philadelphia International Airport
station, creating a daily weather risk score (0-10) that is then aggregated
to each road segment based on the weather conditions when crimes occurred.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import glob
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

DATA_RAW = 'Data/raw'
DATA_PROCESSED = 'Data/processed'
os.makedirs(DATA_PROCESSED, exist_ok=True)

YEARS = [2022, 2023, 2024, 2025, 2026]


# ----------------------------------------------------------------------
# Step 1: Load crime data, keep the year on every row
# ----------------------------------------------------------------------
def load_all_crime():
    frames = []
    for year in YEARS:
        path = DATA_RAW + '/Crime data ' + str(year) + '.csv'
        if os.path.exists(path):
            d = pd.read_csv(path)
            d['year'] = year
            frames.append(d)
            print("Loaded " + str(len(d)) + " incidents from " + str(year))
    combined = pd.concat(frames, ignore_index=True)
    print("Total: " + str(len(combined)) + " incidents across all years")
    return combined

crime_df = load_all_crime()


# ----------------------------------------------------------------------
# Step 1b: Weather data processing
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("PROCESSING WEATHER DATA")
print("=" * 70)

weather = pd.read_csv(DATA_RAW + '/Weather.csv')
print(f"Loaded {len(weather)} weather records")

weather['DATE'] = pd.to_datetime(weather['DATE'])

# Create weather risk score (0 to 10 scale)
weather['rain_risk'] = 0
weather['snow_risk'] = 0
weather['cold_risk'] = 0

# Rain risk
weather.loc[weather['PRCP'] > 0, 'rain_risk'] = 3
weather.loc[weather['PRCP'] > 10, 'rain_risk'] = 5
weather.loc[weather['PRCP'] > 25, 'rain_risk'] = 7

# Snow risk
weather.loc[weather['SNOW'] > 0, 'snow_risk'] = 4
weather.loc[weather['SNOW'] > 50, 'snow_risk'] = 7
weather.loc[weather['SNOW'] > 100, 'snow_risk'] = 10

# Cold risk
weather.loc[weather['TAVG'] < 40, 'cold_risk'] = 2
weather.loc[weather['TAVG'] < 32, 'cold_risk'] = 4
weather.loc[weather['TAVG'] < 20, 'cold_risk'] = 6

# Combined weather risk
weather['weather_risk'] = weather[['rain_risk', 'snow_risk', 'cold_risk']].max(axis=1)
weather['weather_risk'] = weather['weather_risk'].clip(0, 10)

print(f"Weather risk range: {weather['weather_risk'].min()} to {weather['weather_risk'].max()}")
print(f"Days with rain: {(weather['PRCP'] > 0).sum()}")
print(f"Days with snow: {(weather['SNOW'] > 0).sum()}")
print(f"Days below freezing: {(weather['TAVG'] < 32).sum()}")

# Save daily weather risk
daily_weather = weather[['DATE', 'weather_risk']].copy()
daily_weather.to_csv(DATA_PROCESSED + '/daily_weather_risk.csv', index=False)
print("Saved daily_weather_risk.csv")

# Join weather to crime incidents
crime_df['dispatch_date'] = pd.to_datetime(crime_df['dispatch_date'])
crime_df = crime_df.merge(
    daily_weather,
    left_on='dispatch_date',
    right_on='DATE',
    how='left'
)

avg_weather_risk = crime_df['weather_risk'].mean()
crime_df['weather_risk'] = crime_df['weather_risk'].fillna(avg_weather_risk)

print(f"Added weather_risk to {len(crime_df)} crime incidents")
print(f"Average weather risk: {avg_weather_risk:.2f}")


# ----------------------------------------------------------------------
# Step 2: Crime rows to map points
# ----------------------------------------------------------------------
crime_clean = crime_df.dropna(subset=['lat', 'lng']).copy()
print("Rows with valid coordinates: " + str(len(crime_clean)))

crime_gdf = gpd.GeoDataFrame(
    crime_clean,
    geometry=[Point(x, y) for x, y in zip(crime_clean['lng'], crime_clean['lat'])],
    crs='EPSG:4326'
)
crime_gdf = crime_gdf[['geometry', 'year', 'dispatch_date', 'hour', 'text_general_code', 'lat', 'lng', 'weather_risk']]
crime_gdf.to_file(DATA_PROCESSED + '/crime_points.geojson', driver='GeoJSON')


# ----------------------------------------------------------------------
# Step 3: Street poles (lighting proxy)
# ----------------------------------------------------------------------
poles_df = pd.read_csv(DATA_RAW + '/Street_Poles.csv')
poles_gdf = gpd.GeoDataFrame(
    poles_df,
    geometry=[Point(x, y) for x, y in zip(poles_df['X'], poles_df['Y'])],
    crs='EPSG:3857'
).to_crs('EPSG:4326')
poles_gdf = poles_gdf[['geometry', 'pole_num', 'type', 'height']]
print("Loaded " + str(len(poles_gdf)) + " street poles")


# ----------------------------------------------------------------------
# Step 4: Police stations
# ----------------------------------------------------------------------
police_df = pd.read_csv(DATA_RAW + '/Police_Stations.csv')
police_gdf = gpd.GeoDataFrame(
    police_df,
    geometry=[Point(x, y) for x, y in zip(police_df['X'], police_df['Y'])],
    crs='EPSG:3857'
).to_crs('EPSG:4326')
police_gdf = police_gdf[['geometry', 'dist_num', 'location']]
print("Loaded " + str(len(police_gdf)) + " police stations")


# ----------------------------------------------------------------------
# Step 5: Hospitals
# ----------------------------------------------------------------------
hospitals_gdf = gpd.read_file(DATA_RAW + '/pa_hospitals.geojson')
hospitals_gdf = hospitals_gdf[['geometry', 'FACILITY_N', 'LATITUDE', 'LONGITUDE']]
print("Loaded " + str(len(hospitals_gdf)) + " hospitals")


# ----------------------------------------------------------------------
# Step 6: Combine help locations
# ----------------------------------------------------------------------
police_gdf['type'] = 'police'
hospitals_gdf['type'] = 'hospital'
help_locations = pd.concat(
    [police_gdf[['geometry', 'type']], hospitals_gdf[['geometry', 'type']]],
    ignore_index=True
)
help_locations.to_file(DATA_PROCESSED + '/help_locations.geojson', driver='GeoJSON')
print("Total help locations: " + str(len(help_locations)))


# ----------------------------------------------------------------------
# Step 7: Download Philadelphia roads
# ----------------------------------------------------------------------
import osmnx as ox

print("Downloading roads from OpenStreetMap, this can take a minute or two")
G = ox.graph_from_place('Philadelphia, Pennsylvania, USA', network_type='drive', simplify=True)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

roads = edges[['geometry', 'length', 'name', 'highway']].copy().reset_index()
if 'osmid' in roads.columns:
    roads = roads.rename(columns={'osmid': 'edge_id'})
else:
    roads['edge_id'] = range(len(roads))
roads['length_km'] = roads['length'] / 1000
roads.to_file(DATA_PROCESSED + '/philadelphia_roads.geojson', driver='GeoJSON')
print("Downloaded " + str(len(roads)) + " road segments")


# ----------------------------------------------------------------------
# Step 8: Build the feature table
# ----------------------------------------------------------------------
roads = gpd.read_file(DATA_PROCESSED + '/philadelphia_roads.geojson')
roads_proj = roads.copy().to_crs('EPSG:3857')
roads_proj['buffer'] = roads_proj.geometry.buffer(10)
buffers_gdf = gpd.GeoDataFrame(roads_proj[['buffer']], geometry='buffer', crs='EPSG:3857')

crime_proj = crime_gdf.to_crs('EPSG:3857')
crimes_near_roads = gpd.sjoin(crime_proj, buffers_gdf, how='inner', predicate='within')

print(f"Found {len(crimes_near_roads)} crime-road pairs")

# Create a simple dictionary for weather by coordinates
print("Creating weather lookup by coordinates...")
weather_by_coord = {}
for idx, row in crime_df.iterrows():
    key = (round(row['lat'], 6), round(row['lng'], 6))
    weather_by_coord[key] = row['weather_risk']

# Add weather to crimes_near_roads
print("Adding weather risk to crimes near roads...")
weather_values = []
for idx, row in crimes_near_roads.iterrows():
    key = (round(row['lat'], 6), round(row['lng'], 6))
    weather_values.append(weather_by_coord.get(key, avg_weather_risk))

crimes_near_roads['weather_risk'] = weather_values
print(f"Added weather_risk to {len(crimes_near_roads)} records")

# 8.1: Total incidents per segment
crime_counts = crimes_near_roads.groupby(crimes_near_roads.index_right).size()
roads_proj['incident_count'] = crime_counts.reindex(roads_proj.index).fillna(0)

# 8.1b: Average weather risk per segment
avg_weather_per_segment = crimes_near_roads.groupby('index_right')['weather_risk'].mean()
roads_proj['avg_weather_risk'] = avg_weather_per_segment.reindex(roads_proj.index).fillna(avg_weather_risk)
print(f"Added avg_weather_risk. Range: {roads_proj['avg_weather_risk'].min():.2f} to {roads_proj['avg_weather_risk'].max():.2f}")

# 8.1c: Incidents per segment per year
per_year = (
    crimes_near_roads
    .groupby([crimes_near_roads.index_right, 'year'])
    .size()
    .unstack(fill_value=0)
)
for year in YEARS:
    col = 'incidents_' + str(year)
    roads_proj[col] = (per_year[year].reindex(roads_proj.index).fillna(0)
                       if year in per_year.columns else 0)
print("Added per-year incident columns: " + ", ".join('incidents_' + str(y) for y in YEARS))

# 8.2: Poles per segment
poles_proj = poles_gdf.to_crs('EPSG:3857')
poles_near_roads = gpd.sjoin(poles_proj, buffers_gdf, how='inner', predicate='within')
pole_counts = poles_near_roads.groupby(poles_near_roads.index_right).size()
roads_proj['pole_count'] = pole_counts.reindex(roads_proj.index).fillna(0)

# 8.2b: Intersection controls per segment (stop signs and signals)
# A real street property: how many regulated intersections it has. Knowable
# without any crime data, so no leak risk.
controls_gdf = gpd.read_file(DATA_RAW + '/Intersection_Controls.geojson').to_crs('EPSG:3857')
controls_near_roads = gpd.sjoin(controls_gdf, buffers_gdf, how='inner', predicate='within')
control_counts = controls_near_roads.groupby(controls_near_roads.index_right).size()
roads_proj['control_count'] = control_counts.reindex(roads_proj.index).fillna(0)
roads_proj['control_density'] = (roads_proj['control_count'] / roads_proj['length_km'])
roads_proj['control_density'] = roads_proj['control_density'].replace([np.inf, -np.inf], 0).fillna(0)
print("Intersection controls joined. Avg controls per segment: "
      + str(round(roads_proj['control_count'].mean(), 2)))

# 8.3: Distance to nearest help
help_proj = help_locations.to_crs('EPSG:3857')

def min_distance(row, help_gdf):
    if row.geometry.is_empty:
        return np.nan
    return help_gdf.distance(row.geometry).min() / 1000

roads_proj['distance_to_help_km'] = roads_proj.apply(lambda r: min_distance(r, help_proj), axis=1)

# 8.4: Clean highway column
def get_highway_safe(val):
    if val is None:
        return 'residential'
    if isinstance(val, float) and pd.isna(val):
        return 'residential'
    if isinstance(val, (list, tuple, np.ndarray)):
        return get_highway_safe(val[0]) if len(val) > 0 else 'residential'
    return str(val)

roads_proj['highway_clean'] = roads_proj['highway'].apply(get_highway_safe)

# 8.5: Derived features
if 'length' in roads_proj.columns:
    roads_proj['length_km'] = roads_proj['length'] / 1000
else:
    roads_proj['length_km'] = roads_proj.geometry.length / 1000

roads_proj['pole_density'] = (roads_proj['pole_count'] / roads_proj['length_km'])
roads_proj['pole_density'] = roads_proj['pole_density'].replace([np.inf, -np.inf], 0).fillna(0)

def lighting_score(density):
    if density > 20:
        return 3
    elif density > 10:
        return 2
    elif density > 3:
        return 1
    return 0

roads_proj['lighting_score'] = roads_proj['pole_density'].apply(lighting_score)

road_type_weights = {
    'primary': 80, 'secondary': 60, 'tertiary': 40, 'residential': 20,
    'service': 10, 'unclassified': 30, 'motorway': 5, 'trunk': 10,
    'motorway_link': 5, 'trunk_link': 10, 'primary_link': 80,
    'secondary_link': 60, 'tertiary_link': 40
}

def estimate_foot_traffic(row):
    base = road_type_weights.get(row.get('highway_clean', 'residential'), 30)
    edge_id = row.get('edge_id', 0)
    if isinstance(edge_id, (list, tuple, np.ndarray)):
        edge_id = edge_id[0] if len(edge_id) > 0 else 0
    noise = (hash(str(edge_id)) % 21) - 10
    return max(0, min(100, base + noise))

roads_proj['foot_traffic_score'] = roads_proj.apply(estimate_foot_traffic, axis=1)

# 8.5a: Real traffic volume from PennDOT (CUR_AADT = vehicles per day).
# This is real on the ~60% of segments PennDOT monitors (mostly major roads)
# and filled with the citywide median elsewhere. We add a flag so the model
# knows which values are real versus filled. County code is a STRING ('67').
traffic = gpd.read_file(DATA_RAW + '/roadwaytraffic.geojson').to_crs('EPSG:3857')
traffic = traffic[traffic['CTY_CODE'] == '67'].copy()
traffic = traffic[['geometry', 'CUR_AADT']].dropna(subset=['CUR_AADT'])
print("Philadelphia traffic segments: " + str(len(traffic)))

roads_for_join = roads_proj[['edge_id', 'geometry']].copy()
nearest = gpd.sjoin_nearest(
    roads_for_join, traffic, how='left', max_distance=100, distance_col='dist'
)
nearest = nearest.sort_values('dist').drop_duplicates('edge_id')
roads_proj = roads_proj.merge(nearest[['edge_id', 'CUR_AADT']], on='edge_id', how='left')

# Flag real vs filled, then fill and log-scale (traffic is very skewed).
roads_proj['has_traffic_data'] = roads_proj['CUR_AADT'].notna().astype(int)
median_aadt = roads_proj['CUR_AADT'].median()
coverage = roads_proj['has_traffic_data'].mean()
roads_proj['CUR_AADT'] = roads_proj['CUR_AADT'].fillna(median_aadt)
roads_proj['traffic_volume_log'] = np.log1p(roads_proj['CUR_AADT'])
print("Traffic coverage: " + str(round(coverage * 100, 1)) + " percent")

# Leak check: traffic should not strongly track crime counts.
print(roads_proj[['traffic_volume_log', 'incident_count']].corr())

# 8.5b: REAL income, joined from Census ACS by tract
tract_shp = glob.glob(DATA_RAW + '/Census_Tracts_2010-shp/*.shp')[0]
tracts = gpd.read_file(tract_shp).to_crs('EPSG:3857')
tracts['GEOID10'] = tracts['GEOID10'].astype(str)

income = pd.read_csv(DATA_RAW + '/tract_income.csv')
income['GEOID10'] = income['GEOID10'].astype(str)
tracts = tracts.merge(income, on='GEOID10', how='left')

road_points = roads_proj.copy()
road_points['geometry'] = road_points.geometry.centroid
joined = gpd.sjoin(
    road_points[['geometry']], tracts[['geometry', 'median_income']],
    how='left', predicate='within'
)
joined = joined[~joined.index.duplicated(keep='first')]
roads_proj['median_income'] = joined['median_income'].values

citywide_median = roads_proj['median_income'].median()
roads_proj['median_income'] = roads_proj['median_income'].fillna(citywide_median)

roads_proj['income_quartile'] = pd.qcut(
    roads_proj['median_income'], q=4, labels=[1, 2, 3, 4]
).astype(int)

print("Joined real income. Fallback median used: " + str(int(citywide_median)))
print("Income quartile counts: "
      + str(roads_proj['income_quartile'].value_counts().sort_index().to_dict()))

print("Lighting score distribution: " + str(roads_proj['lighting_score'].value_counts().to_dict()))
print("Avg foot traffic: " + str(round(roads_proj['foot_traffic_score'].mean(), 1)))
print("Avg weather risk: " + str(round(roads_proj['avg_weather_risk'].mean(), 2)))

# 8.6: Assemble the final table
roads_proj = roads_proj.reset_index(drop=True)
if 'edge_id' not in roads_proj.columns:
    roads_proj['edge_id'] = range(len(roads_proj))

# Add segment centroid coordinates so Phase 2 can do spatial validation.
roads_wgs = roads_proj.to_crs('EPSG:4326')
roads_proj['seg_lon'] = roads_wgs.geometry.centroid.x
roads_proj['seg_lat'] = roads_wgs.geometry.centroid.y

print(roads_proj[['control_density', 'incident_count']].corr())

per_year_cols = ['incidents_' + str(y) for y in YEARS]
feature_columns = (
    ['edge_id', 'length_km', 'incident_count'] + per_year_cols +
    ['pole_count', 'pole_density', 'lighting_score', 'foot_traffic_score',
     'distance_to_help_km', 'control_density', 'traffic_volume_log',
     'has_traffic_data', 'median_income', 'income_quartile',
     'seg_lon', 'seg_lat']
)

feature_table = roads_proj[[c for c in feature_columns if c in roads_proj.columns]].copy()
for col in ['name', 'highway_clean']:
    if col in roads_proj.columns:
        feature_table[col] = roads_proj[col]
feature_table = feature_table.rename(columns={'highway_clean': 'highway'})

feature_table = feature_table.dropna(subset=['length_km'])
feature_table = feature_table[(feature_table['length_km'] > 0) & (feature_table['length_km'] < 10)]

# 8.7: Save
feature_table.to_csv(DATA_PROCESSED + '/feature_table_temporal.csv', index=False)
feature_table.sample(min(100, len(feature_table))).to_csv(
    DATA_PROCESSED + '/feature_table_temporal_sample.csv', index=False
)

print("")
print("Final feature table: " + str(feature_table.shape[0]) + " segments, "
      + str(feature_table.shape[1]) + " columns")
print("Saved Data/processed/feature_table_temporal.csv")
print("")
print("Phase 1 complete. Ready for Phase 2.")
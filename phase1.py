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

def load_all_crime():
    """Load and combine crime data from all years"""
    frames = []
    for year in YEARS:
        path = DATA_RAW + '/Crime data ' + str(year) + '.csv'
        if os.path.exists(path):
            d = pd.read_csv(path)
            d['year'] = year
            frames.append(d)
            print(f"Loaded {len(d):,} incidents from {year}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(combined):,} incidents across all years")
    return combined

crime_df = load_all_crime()

# Process weather data
weather = pd.read_csv(DATA_RAW + '/Weather.csv')
print(f"Loaded {len(weather):,} weather records")

weather['DATE'] = pd.to_datetime(weather['DATE'])

# Create weather risk score (0-10 scale)
weather['rain_risk'] = 0
weather['snow_risk'] = 0
weather['cold_risk'] = 0

weather.loc[weather['PRCP'] > 0, 'rain_risk'] = 3
weather.loc[weather['PRCP'] > 10, 'rain_risk'] = 5
weather.loc[weather['PRCP'] > 25, 'rain_risk'] = 7

weather.loc[weather['SNOW'] > 0, 'snow_risk'] = 4
weather.loc[weather['SNOW'] > 50, 'snow_risk'] = 7
weather.loc[weather['SNOW'] > 100, 'snow_risk'] = 10

weather.loc[weather['TAVG'] < 40, 'cold_risk'] = 2
weather.loc[weather['TAVG'] < 32, 'cold_risk'] = 4
weather.loc[weather['TAVG'] < 20, 'cold_risk'] = 6

weather['weather_risk'] = weather[['rain_risk', 'snow_risk', 'cold_risk']].max(axis=1)
weather['weather_risk'] = weather['weather_risk'].clip(0, 10)

daily_weather = weather[['DATE', 'weather_risk']].copy()
daily_weather.to_csv(DATA_PROCESSED + '/daily_weather_risk.csv', index=False)

# Attach weather to crime incidents
crime_df['dispatch_date'] = pd.to_datetime(crime_df['dispatch_date'])
crime_df = crime_df.merge(
    daily_weather,
    left_on='dispatch_date',
    right_on='DATE',
    how='left'
)

avg_weather_risk = crime_df['weather_risk'].mean()
crime_df['weather_risk'] = crime_df['weather_risk'].fillna(avg_weather_risk)

# Convert crime records to spatial points
crime_clean = crime_df.dropna(subset=['lat', 'lng']).copy()
print(f"Rows with valid coordinates: {len(crime_clean):,}")

crime_gdf = gpd.GeoDataFrame(
    crime_clean,
    geometry=[Point(x, y) for x, y in zip(crime_clean['lng'], crime_clean['lat'])],
    crs='EPSG:4326'
)
crime_gdf = crime_gdf[['geometry', 'year', 'dispatch_date', 'hour', 'text_general_code', 'lat', 'lng', 'weather_risk']]
crime_gdf.to_file(DATA_PROCESSED + '/crime_points.geojson', driver='GeoJSON')

# Load infrastructure data
poles_df = pd.read_csv(DATA_RAW + '/Street_Poles.csv')
poles_gdf = gpd.GeoDataFrame(
    poles_df,
    geometry=[Point(x, y) for x, y in zip(poles_df['X'], poles_df['Y'])],
    crs='EPSG:3857'
).to_crs('EPSG:4326')
poles_gdf = poles_gdf[['geometry', 'pole_num', 'type', 'height']]
print(f"Loaded {len(poles_gdf):,} street poles")

police_df = pd.read_csv(DATA_RAW + '/Police_Stations.csv')
police_gdf = gpd.GeoDataFrame(
    police_df,
    geometry=[Point(x, y) for x, y in zip(police_df['X'], police_df['Y'])],
    crs='EPSG:3857'
).to_crs('EPSG:4326')
police_gdf = police_gdf[['geometry', 'dist_num', 'location']]
print(f"Loaded {len(police_gdf)} police stations")

hospitals_gdf = gpd.read_file(DATA_RAW + '/pa_hospitals.geojson')
hospitals_gdf = hospitals_gdf[['geometry', 'FACILITY_N', 'LATITUDE', 'LONGITUDE']]
print(f"Loaded {len(hospitals_gdf)} hospitals")

# Combine help locations
police_gdf['type'] = 'police'
hospitals_gdf['type'] = 'hospital'
help_locations = pd.concat(
    [police_gdf[['geometry', 'type']], hospitals_gdf[['geometry', 'type']]],
    ignore_index=True
)
help_locations.to_file(DATA_PROCESSED + '/help_locations.geojson', driver='GeoJSON')
print(f"Total help locations: {len(help_locations)}")

# Download road network
import osmnx as ox
print("Downloading Philadelphia road network...")
G = ox.graph_from_place('Philadelphia, Pennsylvania, USA', network_type='drive', simplify=True)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

roads = edges[['geometry', 'length', 'name', 'highway']].copy().reset_index()
if 'osmid' in roads.columns:
    roads = roads.rename(columns={'osmid': 'edge_id'})
else:
    roads['edge_id'] = range(len(roads))
roads['length_km'] = roads['length'] / 1000
roads.to_file(DATA_PROCESSED + '/philadelphia_roads.geojson', driver='GeoJSON')
print(f"Downloaded {len(roads):,} road segments")

# Build feature table
roads = gpd.read_file(DATA_PROCESSED + '/philadelphia_roads.geojson')
roads_proj = roads.copy().to_crs('EPSG:3857')
roads_proj['buffer'] = roads_proj.geometry.buffer(10)
buffers_gdf = gpd.GeoDataFrame(roads_proj[['buffer']], geometry='buffer', crs='EPSG:3857')

crime_proj = crime_gdf.to_crs('EPSG:3857')
crimes_near_roads = gpd.sjoin(crime_proj, buffers_gdf, how='inner', predicate='within')
print(f"Found {len(crimes_near_roads):,} crime-road pairs")

# Create weather lookup
weather_by_coord = {}
for idx, row in crime_df.iterrows():
    key = (round(row['lat'], 6), round(row['lng'], 6))
    weather_by_coord[key] = row['weather_risk']

weather_values = []
for idx, row in crimes_near_roads.iterrows():
    key = (round(row['lat'], 6), round(row['lng'], 6))
    weather_values.append(weather_by_coord.get(key, avg_weather_risk))
crimes_near_roads['weather_risk'] = weather_values

# Calculate incident counts per segment
crime_counts = crimes_near_roads.groupby(crimes_near_roads.index_right).size()
roads_proj['incident_count'] = crime_counts.reindex(roads_proj.index).fillna(0)

avg_weather_per_segment = crimes_near_roads.groupby('index_right')['weather_risk'].mean()
roads_proj['avg_weather_risk'] = avg_weather_per_segment.reindex(roads_proj.index).fillna(avg_weather_risk)

# Incidents per year
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

# Count poles per segment
poles_proj = poles_gdf.to_crs('EPSG:3857')
poles_near_roads = gpd.sjoin(poles_proj, buffers_gdf, how='inner', predicate='within')
pole_counts = poles_near_roads.groupby(poles_near_roads.index_right).size()
roads_proj['pole_count'] = pole_counts.reindex(roads_proj.index).fillna(0)

# Intersection controls
controls_gdf = gpd.read_file(DATA_RAW + '/Intersection_Controls.geojson').to_crs('EPSG:3857')
controls_near_roads = gpd.sjoin(controls_gdf, buffers_gdf, how='inner', predicate='within')
control_counts = controls_near_roads.groupby(controls_near_roads.index_right).size()
roads_proj['control_count'] = control_counts.reindex(roads_proj.index).fillna(0)
roads_proj['control_density'] = (roads_proj['control_count'] / roads_proj['length_km'])
roads_proj['control_density'] = roads_proj['control_density'].replace([np.inf, -np.inf], 0).fillna(0)

# Distance to nearest help
help_proj = help_locations.to_crs('EPSG:3857')

def min_distance(row, help_gdf):
    if row.geometry.is_empty:
        return np.nan
    return help_gdf.distance(row.geometry).min() / 1000

roads_proj['distance_to_help_km'] = roads_proj.apply(lambda r: min_distance(r, help_proj), axis=1)

# Clean highway column
def get_highway_safe(val):
    if val is None:
        return 'residential'
    if isinstance(val, float) and pd.isna(val):
        return 'residential'
    if isinstance(val, (list, tuple, np.ndarray)):
        return get_highway_safe(val[0]) if len(val) > 0 else 'residential'
    return str(val)

roads_proj['highway_clean'] = roads_proj['highway'].apply(get_highway_safe)

# Derived features
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

# Traffic volume from PennDOT
traffic = gpd.read_file(DATA_RAW + '/roadwaytraffic.geojson').to_crs('EPSG:3857')
traffic = traffic[traffic['CTY_CODE'] == '67'].copy()
traffic = traffic[['geometry', 'CUR_AADT']].dropna(subset=['CUR_AADT'])
print(f"Philadelphia traffic segments: {len(traffic):,}")

roads_for_join = roads_proj[['edge_id', 'geometry']].copy()
nearest = gpd.sjoin_nearest(
    roads_for_join, traffic, how='left', max_distance=100, distance_col='dist'
)
nearest = nearest.sort_values('dist').drop_duplicates('edge_id')
roads_proj = roads_proj.merge(nearest[['edge_id', 'CUR_AADT']], on='edge_id', how='left')

roads_proj['has_traffic_data'] = roads_proj['CUR_AADT'].notna().astype(int)
median_aadt = roads_proj['CUR_AADT'].median()
roads_proj['CUR_AADT'] = roads_proj['CUR_AADT'].fillna(median_aadt)
roads_proj['traffic_volume_log'] = np.log1p(roads_proj['CUR_AADT'])

# Income data from Census
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

# Assemble final feature table
roads_proj = roads_proj.reset_index(drop=True)
if 'edge_id' not in roads_proj.columns:
    roads_proj['edge_id'] = range(len(roads_proj))

roads_wgs = roads_proj.to_crs('EPSG:4326')
roads_proj['seg_lon'] = roads_wgs.geometry.centroid.x
roads_proj['seg_lat'] = roads_wgs.geometry.centroid.y

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

feature_table.to_csv(DATA_PROCESSED + '/feature_table_temporal.csv', index=False)
feature_table.sample(min(100, len(feature_table))).to_csv(
    DATA_PROCESSED + '/feature_table_temporal_sample.csv', index=False
)

print(f"\nFinal feature table: {feature_table.shape[0]:,} segments, {feature_table.shape[1]} columns")
print(f"Saved to {DATA_PROCESSED}/feature_table_temporal.csv")
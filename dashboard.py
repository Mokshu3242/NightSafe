"""
NightSafe Dashboard - Philadelphia Night Safety Decision Support Tool
With Neighborhood Aggregation, Real-Time Weather API, Cost-Benefit Analysis,
Time-Series Forecast, Export Report, Compare Corridors, and District Map
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

st.set_page_config(
    page_title="NightSafe - Philadelphia Night Safety Simulator",
    page_icon=":crescent_moon:",
    layout="wide"
)


def strip_brackets(series):
    """Clean street names from list format ['Name'] to Name"""
    return (series.astype(str)
            .str.replace(r"^\['?|'?\]$", "", regex=True)
            .str.replace("'", "")
            .str.strip())


def get_severity_color(severity):
    colors = {
        'low': '#2ecc71',
        'moderate': '#f1c40f',
        'high': '#e67e22',
        'severe': '#e74c3c'
    }
    return colors.get(severity, '#95a5a6')


def generate_report(corridor_name, risk_index, severity, length_km, results, best_action, years):
    action_names_report = {
        'do_nothing': 'Do Nothing',
        'add_lighting': 'Add Lighting',
        'add_lighting_and_booth': 'Add Lighting + Call Booth',
        'add_patrols': 'Add Police Patrols',
        'full_intervention': 'Full Intervention'
    }
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NightSafe Report - {corridor_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
            .severe {{ background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; }}
            .high {{ background-color: #e67e22; color: white; padding: 10px; border-radius: 5px; }}
            .moderate {{ background-color: #f1c40f; padding: 10px; border-radius: 5px; }}
            .low {{ background-color: #2ecc71; padding: 10px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .footer {{ font-size: 12px; color: #7f8c8d; margin-top: 30px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>NightSafe Report</h1>
            <p>Cost of Doing Nothing Simulator - Philadelphia</p>
        </div>
        
        <div class="section">
            <h2>Corridor Summary</h2>
            <p><strong>Name:</strong> {corridor_name}</p>
            <p><strong>Length:</strong> {length_km:.2f} km</p>
            <p><strong>Risk Index:</strong> {risk_index}/100</p>
            <p><strong>Severity:</strong> {severity.upper()}</p>
            <p><strong>Time Horizon:</strong> {years} years</p>
            <div class="{severity}">
                <strong>Risk Level:</strong> {severity.upper()} Risk Corridor
            </div>
        </div>
        
        <div class="section">
            <h2>Intervention Comparison ({years} Years)</h2>
            <table>
                <tr><th>Action</th><th>Projected Incidents</th><th>Net Savings vs Do Nothing</th></tr>
                <tr><td>Do Nothing</td><td>{results['do_nothing']['incidents'][0]:,} - {results['do_nothing']['incidents'][1]:,}</td><td>N/A</td></tr>
                <tr><td>Add Lighting</td><td>{results['add_lighting']['incidents'][0]:,} - {results['add_lighting']['incidents'][1]:,}</td><td>${results['add_lighting']['net'][0]:,.0f} - ${results['add_lighting']['net'][1]:,.0f}</td></tr>
                <tr><td>Add Lighting + Booth</td><td>{results['add_lighting_and_booth']['incidents'][0]:,} - {results['add_lighting_and_booth']['incidents'][1]:,}</td><td>${results['add_lighting_and_booth']['net'][0]:,.0f} - ${results['add_lighting_and_booth']['net'][1]:,.0f}</td></tr>
                <tr><td>Add Patrols</td><td>{results['add_patrols']['incidents'][0]:,} - {results['add_patrols']['incidents'][1]:,}</td><td>${results['add_patrols']['net'][0]:,.0f} - ${results['add_patrols']['net'][1]:,.0f}</td></tr>
                <tr><td>Full Intervention</td><td>{results['full_intervention']['incidents'][0]:,} - {results['full_intervention']['incidents'][1]:,}</td><td>${results['full_intervention']['net'][0]:,.0f} - ${results['full_intervention']['net'][1]:,.0f}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Recommendation</h2>
            <p><strong>{action_names_report[best_action]}</strong></p>
            <p>Prevents an estimated {max(0, results['do_nothing']['incidents'][0] - results[best_action]['incidents'][1]):,} - {results['do_nothing']['incidents'][1] - results[best_action]['incidents'][0]:,} incidents over {years} years.</p>
        </div>
        
        <div class="footer">
            <p>NightSafe - Cost of Doing Nothing Simulator</p>
            <p>Data: OpenDataPhilly, OpenStreetMap, US Census, PennDOT | Model: Random Forest (honest R2 = 0.26, leakage removed)</p>
            <p>All outputs are ranges. Human makes final investment decision. Fairness audited quarterly.</p>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    return report_html


def calculate_yearly_incidents(risk_index, length_km, action, years):
    intervention_effects = {
        'do_nothing': 1.00,
        'add_lighting': 0.55,
        'add_lighting_and_booth': 0.40,
        'add_patrols': 0.50,
        'full_intervention': 0.30
    }
    
    effect = intervention_effects.get(action, 1.00)
    base_rate = (risk_index / 100) * length_km
    
    if risk_index >= 90:
        comp_rate = 0.15
    elif risk_index >= 70:
        comp_rate = 0.08
    else:
        comp_rate = 0.03
    
    if action != 'do_nothing':
        comp_rate = 0
    
    yearly_incidents = []
    
    for year in range(1, years + 1):
        if comp_rate > 0:
            compound_factor = (1 + comp_rate) ** (year - 1)
        else:
            compound_factor = 1
        
        incidents = base_rate * effect * compound_factor
        lower = max(0, int(incidents * 0.8))
        upper = int(incidents * 1.2)
        yearly_incidents.append((lower, upper))
    
    return yearly_incidents


def calculate_cumulative_incidents(yearly_incidents):
    cumulative = []
    running_lower = 0
    running_upper = 0
    
    for lower, upper in yearly_incidents:
        running_lower += lower
        running_upper += upper
        cumulative.append((running_lower, running_upper))
    
    return cumulative


def get_philadelphia_weather():
    API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
    
    if not API_KEY:
        return None
    
    lat = 39.9526
    lon = -75.1652
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=imperial"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            weather = {
                'temperature': round(data['main']['temp']),
                'feels_like': round(data['main']['feels_like']),
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'wind_speed': round(data['wind']['speed']),
                'condition': data['weather'][0]['main']
            }
            return weather
        else:
            return None
    except Exception:
        return None


def get_weather_risk_adjustment(weather):
    if weather is None:
        return 1.0, "Weather data unavailable. Using baseline risk."
    
    condition = weather['condition']
    temp = weather['temperature']
    wind = weather['wind_speed']
    
    if condition in ['Rain', 'Drizzle', 'Thunderstorm']:
        return 1.3, "Rain increases night safety risk by 30%"
    elif condition == 'Snow':
        return 1.5, "Snow increases night safety risk by 50%"
    elif temp < 32:
        return 1.2, "Freezing temperatures increase risk by 20%"
    elif wind > 20:
        return 1.1, "High winds increase risk by 10%"
    elif condition == 'Clear':
        return 0.9, "Clear weather reduces baseline risk by 10%"
    else:
        return 1.0, "Normal weather conditions. Baseline risk applies."


@st.cache_data
def load_corridor_data():
    df = pd.read_csv('Data/processed/corridor_rankings.csv')
    df['name'] = strip_brackets(df['name'])
    return df


@st.cache_data
def load_road_coordinates():
    try:
        import geopandas as gpd
        roads = gpd.read_file('Data/processed/philadelphia_roads.geojson')
        roads_proj = roads.to_crs('EPSG:3857')
        centroids = roads_proj.geometry.centroid.to_crs('EPSG:4326')
        roads['centroid_lat'] = centroids.y
        roads['centroid_lon'] = centroids.x
        roads['name'] = strip_brackets(roads['name'])
        
        by_name = roads.groupby('name').agg(
            centroid_lat=('centroid_lat', 'mean'),
            centroid_lon=('centroid_lon', 'mean'),
        ).reset_index()
        return by_name
    except Exception as e:
        return None


@st.cache_data
def load_neighborhoods():
    try:
        import geopandas as gpd
        districts = gpd.read_file('Data/raw/police_districts/Boundaries_District.shp')
        districts = districts.to_crs('EPSG:4326')
        districts['neighborhood'] = 'District ' + districts['dist_numc'].astype(str)
        return districts
    except Exception as e:
        return None


@st.cache_data
def load_police_districts_geojson():
    try:
        import geopandas as gpd
        districts = gpd.read_file('Data/raw/police_districts/Boundaries_District.shp')
        districts = districts.to_crs('EPSG:4326')
        districts['district_name'] = 'District ' + districts['dist_numc'].astype(str)
        districts['centroid_lat'] = districts.geometry.centroid.y
        districts['centroid_lon'] = districts.geometry.centroid.x
        return districts
    except Exception as e:
        return None

def format_reason(reason_text):
    """Turn the raw SHAP reason into a clean sentence. No LLM, no invention."""
    if not reason_text or str(reason_text) == 'nan':
        return "This corridor's risk reflects its overall street characteristics."
    parts = [p.strip() for p in str(reason_text).split(';') if p.strip()]
    raised = [p.replace(' raised the risk', '') for p in parts if 'raised the risk' in p]
    lowered = [p.replace(' lowered the risk', '') for p in parts if 'lowered the risk' in p]
    sentence = ""
    if raised:
        sentence += "Factors that raised this corridor's risk: " + ", ".join(raised) + ". "
    if lowered:
        sentence += "Factors that lowered it: " + ", ".join(lowered) + "."
    return sentence.strip() if sentence else "This corridor's risk reflects its overall street characteristics."

@st.cache_data
def load_explanations():
    try:
        r = pd.read_csv('Data/processed/shap_reasons_by_street.csv')
        return {str(n): format_reason(reason) for n, reason in zip(r['name'], r['reason'])}
    except Exception:
        return {}

try:
    corridors = load_corridor_data()
    explanations = load_explanations()
    st.success(f"Loaded {len(corridors):,} road corridors")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

roads_gdf = load_road_coordinates()

if roads_gdf is not None:
    corridors_with_coords = corridors.merge(
        roads_gdf, on='name', how='left'
    ).dropna(subset=['centroid_lat', 'centroid_lon'])
    st.info(f"Matched {len(corridors_with_coords):,} corridors with real coordinates")
else:
    np.random.seed(42)
    corridors_with_coords = corridors.copy()
    corridors_with_coords['centroid_lat'] = 39.95 + np.random.normal(0, 0.05, len(corridors))
    corridors_with_coords['centroid_lon'] = -75.16 + np.random.normal(0, 0.08, len(corridors))
    st.warning("Using approximate coordinates")

st.sidebar.header("Filters")

st.sidebar.markdown("---")
sort_choice = st.sidebar.radio(
    "Rank corridors by",
    ["Model risk (default)", "Past reported crime"],
    index=0
)

severity_options = ['All', 'low', 'moderate', 'high', 'severe']
selected_severity = st.sidebar.selectbox("Risk Severity", severity_options)

min_length = st.sidebar.slider(
    "Minimum Corridor Length (km)",
    min_value=0.0,
    max_value=2.0,
    value=0.05,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Simulation Settings")
years = st.sidebar.selectbox("Time Horizon (Years)", [1, 3, 5], index=2)
year_label = f"{years} year" if years == 1 else f"{years} years"

filtered = corridors_with_coords.copy()
if selected_severity != 'All':
    filtered = filtered[filtered['severity'] == selected_severity]

filtered = filtered[filtered['length_km'] >= min_length]
if sort_choice == "Past reported crime" and 'reported_crime' in filtered.columns:
    filtered = filtered.sort_values('reported_crime', ascending=False)
else:
    filtered = filtered.sort_values('priority_score', ascending=False)

if len(filtered) == 0:
    st.warning("No corridors match these filters.")
    st.stop()

st.title("NightSafe")
st.markdown(f"### Cost of Doing Nothing Simulator for Philadelphia ({years}-Year Projection)")
st.markdown("---")

st.subheader("Live Weather Conditions")

weather = get_philadelphia_weather()

if weather:
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        st.metric("Temperature", f"{weather['temperature']}F", f"Feels like {weather['feels_like']}F")
    with col_w2:
        st.metric("Conditions", weather['description'].title())
    with col_w3:
        st.metric("Humidity", f"{weather['humidity']}%")
    with col_w4:
        st.metric("Wind Speed", f"{weather['wind_speed']} mph")
    
    risk_adjustment, weather_message = get_weather_risk_adjustment(weather)
    
    if risk_adjustment != 1.0:
        st.warning(weather_message)
    else:
        st.info(weather_message)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Live Weather")
    st.sidebar.markdown(f"**{weather['temperature']}F** · {weather['description'].title()}")
    if risk_adjustment != 1.0:
        st.sidebar.markdown(f"Risk factor: {risk_adjustment:.1f}x")
else:
    st.info("Live weather unavailable. Weather is shown for context only and does not affect risk scores.")

st.markdown("---")

# ============================================================================
# Neighborhood Summary for District Coloring
# ============================================================================

neighborhoods = load_neighborhoods()
neighborhood_summary = None

if neighborhoods is not None:
    import geopandas as gpd
    from shapely.geometry import Point
    
    corridor_points = filtered[['name', 'centroid_lat', 'centroid_lon', 'predicted_risk', 
                                 'severity', 'priority_score', 'prevented_incidents_min', 
                                 'prevented_incidents_max', 'length_km']].copy()
    
    geometry = [Point(lon, lat) for lon, lat in zip(corridor_points['centroid_lon'], corridor_points['centroid_lat'])]
    corridor_gdf = gpd.GeoDataFrame(corridor_points, geometry=geometry, crs='EPSG:4326')
    
    corridors_with_neighborhood = gpd.sjoin(corridor_gdf, neighborhoods, how='left', predicate='within')
    
    neighborhood_summary = corridors_with_neighborhood.groupby('neighborhood').agg(
        corridor_count=('name', 'count'),
        avg_risk=('predicted_risk', 'mean'),
        total_prevented_min=('prevented_incidents_min', 'sum'),
        total_prevented_max=('prevented_incidents_max', 'sum'),
        severe_count=('severity', lambda x: (x == 'severe').sum()),
        high_count=('severity', lambda x: (x == 'high').sum())
    ).reset_index()
    
    neighborhood_summary = neighborhood_summary.sort_values('avg_risk', ascending=False)

# ============================================================================
# Interactive Map with Plotly and District Outlines
# ============================================================================

st.subheader("Interactive Night Safety Risk Map")

map_style = st.selectbox(
    "Map Style",
    ["CartoDB Positron", "CartoDB Dark", "OpenStreetMap"],
    index=0
)

style_map = {
    "OpenStreetMap": "open-street-map",
    "CartoDB Positron": "carto-positron",
    "CartoDB Dark": "carto-darkmatter"
}

map_data = filtered.head(800).copy()

fig_map = px.scatter_mapbox(
    map_data,
    lat='centroid_lat',
    lon='centroid_lon',
    color='severity',
    color_discrete_map={
        'low': '#2ecc71',
        'moderate': '#f1c40f',
        'high': '#e67e22',
        'severe': '#e74c3c'
    },
    size='priority_score',
    size_max=18,
    hover_name='name',
    hover_data={
        'predicted_risk': ':.0f',
        'severity': True,
        'recommended_action': True,
        'prevented_incidents_min': ':.0f',
        'prevented_incidents_max': ':.0f',
        'length_km': ':.2f'
    },
    title=f"Philadelphia Night Safety Risk Map ({years}-Year Projection)",
    zoom=11,
    height=650
)

# Add district boundaries
districts_gdf = load_police_districts_geojson()

if districts_gdf is not None and neighborhood_summary is not None:
    for idx, row in districts_gdf.iterrows():
        district_num = str(row['dist_numc'])
        district_stats = neighborhood_summary[neighborhood_summary['neighborhood'] == f'District {district_num}']
        
        if len(district_stats) > 0:
            avg_risk = district_stats['avg_risk'].iloc[0]
            if avg_risk >= 70:
                line_color = '#e74c3c'
            elif avg_risk >= 50:
                line_color = '#e67e22'
            elif avg_risk >= 40:
                line_color = '#f1c40f'
            else:
                line_color = '#2ecc71'
        else:
            line_color = '#95a5a6'
        
        if row.geometry.geom_type == 'Polygon':
            coords = list(row.geometry.exterior.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            fig_map.add_trace(
                go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(width=2, color=line_color),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo='none',
                    name=f'District {district_num}'
                )
            )
            
            fig_map.add_trace(
                go.Scattermapbox(
                    lon=[row['centroid_lon']],
                    lat=[row['centroid_lat']],
                    mode='text',
                    text=[f'District {district_num}'],
                    textfont=dict(size=10, color=line_color, weight='bold'),
                    showlegend=False,
                    hoverinfo='none',
                    name=f'Label {district_num}'
                )
            )

fig_map.update_layout(mapbox_style=style_map.get(map_style, "carto-positron"))
fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
fig_map.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01,
    bgcolor='rgba(255,255,255,0.8)'
))

st.plotly_chart(fig_map, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("Green: Low Risk (Index < 40)")
with col2:
    st.markdown("Yellow: Moderate (40-70)")
with col3:
    st.markdown("Orange: High (70-90)")
with col4:
    st.markdown("Red: Severe (Index >= 90)")
st.caption(f"Showing {len(map_data):,} corridors. Circle size = Priority Score (larger = more preventable incidents). Colored outlines show police district boundaries.")
st.markdown("---")

# ============================================================================
# Two Column Layout
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Priority Corridors for Investment")
    
    display_df = filtered.head(20).copy()
    display_df['display_name'] = display_df['name'].str[:35]
    display_df['prevented'] = (
        display_df['prevented_incidents_min'].astype(int).astype(str) + 
        " - " + 
        display_df['prevented_incidents_max'].astype(int).astype(str)
    )
    display_df['action'] = display_df['recommended_action'].str.replace('_', ' ').str.title()
    
    table_df = pd.DataFrame({
        'Corridor': display_df['display_name'],
        'Length (km)': display_df['length_km'].round(2),
        'Model Risk (structural)': display_df['predicted_risk'].round(0).astype(int),
        'Past Reported Crime': display_df['reported_crime'].astype(int) if 'reported_crime' in display_df.columns else 0,
        'Severity': display_df['severity'].str.upper(),
        'Action': display_df['action'],
        f'Prevented ({years}y)': display_df['prevented']
    })
    
    st.dataframe(table_df, use_container_width=True)
    st.caption("Model Risk is the structural prediction and does not use crime data. "
               "Past Reported Crime is historical record shown for context only. "
               "The model deliberately excludes crime history to avoid reinforcing where "
               "policing already concentrates; officials see both and apply judgment.")

with col2:
    st.subheader("Summary Statistics")
    
    total_prevented_min = filtered['prevented_incidents_min'].sum()
    total_prevented_max = filtered['prevented_incidents_max'].sum()
    
    st.metric(
        f"Total Preventable Incidents ({years} years)",
        f"{int(total_prevented_min):,} - {int(total_prevented_max):,}"
    )
    
    high_severe_count = len(filtered[filtered['severity'].isin(['high', 'severe'])])
    st.metric(
        "High + Severe Risk Corridors",
        f"{high_severe_count:,}",
        delta=f"{high_severe_count/len(filtered)*100:.1f}% of total"
    )
    
    severity_counts = filtered['severity'].value_counts()
    fig_pie = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Risk Severity Distribution",
        color=severity_counts.index,
        color_discrete_map={
            'low': '#2ecc71',
            'moderate': '#f1c40f',
            'high': '#e67e22',
            'severe': '#e74c3c'
        }
    )
    fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================================
# Risk Distribution
# ============================================================================

st.markdown("---")
st.subheader("Risk Index Distribution")

fig_hist = px.histogram(
    filtered,
    x='predicted_risk',
    nbins=50,
    title='Distribution of Risk Index Scores (0 to 100 scale)',
    labels={'predicted_risk': 'Risk Index', 'count': 'Number of Corridors'},
    color_discrete_sequence=['steelblue']
)
fig_hist.add_vline(x=70, line_dash="dash", line_color="#e67e22", annotation_text="High Risk (70)")
fig_hist.add_vline(x=90, line_dash="dash", line_color="#e74c3c", annotation_text="Severe Risk (90)")
st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# Neighborhood Aggregation Table
# ============================================================================

st.markdown("---")
st.subheader("Neighborhood Risk Summary")

if neighborhood_summary is not None:
    top_neighborhoods = neighborhood_summary.head(10)[['neighborhood', 'avg_risk', 'severe_count', 
                                                        'high_count', 'total_prevented_min', 'total_prevented_max']].copy()
    top_neighborhoods['avg_risk'] = top_neighborhoods['avg_risk'].round(1)
    top_neighborhoods.columns = ['Neighborhood', 'Avg Risk Index', 'Severe', 'High', 'Prevented Min', 'Prevented Max']
    top_neighborhoods['Prevented Incidents'] = top_neighborhoods['Prevented Min'].astype(int).astype(str) + ' - ' + top_neighborhoods['Prevented Max'].astype(int).astype(str)
    top_neighborhoods = top_neighborhoods.drop(['Prevented Min', 'Prevented Max'], axis=1)
    
    st.dataframe(top_neighborhoods, use_container_width=True)
    
    fig_neighborhood = px.bar(
        top_neighborhoods.head(8),
        x='Neighborhood',
        y='Avg Risk Index',
        title='Average Risk Index by Neighborhood',
        color='Avg Risk Index',
        color_continuous_scale=['green', 'yellow', 'orange', 'red'],
        range_color=[0, 100]
    )
    st.plotly_chart(fig_neighborhood, use_container_width=True)
else:
    st.info("Neighborhood data not available. Using severity distribution as alternative.")

# ============================================================================
# Compare Two Corridors
# ============================================================================

st.markdown("---")
st.subheader("Compare Two Corridors")

compare_col1, compare_col2 = st.columns(2)


class CostOfDoingNothingSimulator:
    def __init__(self):
        self.intervention_effects = {
            'do_nothing': 1.00, 'add_lighting': 0.55,
            'add_lighting_and_booth': 0.40, 'add_patrols': 0.50, 'full_intervention': 0.30
        }
        self.incident_costs = {
            'low': (5000, 20000), 'moderate': (50000, 150000),
            'high': (100000, 300000), 'severe': (200000, 500000)
        }
        self.intervention_costs = {
            'add_lighting': (50000, 150000), 'add_lighting_and_booth': (100000, 250000),
            'add_patrols': (150000, 400000), 'full_intervention': (300000, 700000)
        }
    
    def get_compounding_rate(self, risk):
        return 0.15 if risk >= 90 else (0.08 if risk >= 70 else 0.03)
    
    def get_severity(self, risk):
        if risk >= 90:
            return 'severe'
        elif risk >= 70:
            return 'high'
        elif risk >= 40:
            return 'moderate'
        return 'low'
    
    def simulate(self, risk, length_km, years=5):
        severity = self.get_severity(risk)
        comp_rate = self.get_compounding_rate(risk)
        cost_per_incident = self.incident_costs.get(severity, (50000, 150000))
        
        results = {}
        actions = ['do_nothing', 'add_lighting', 'add_lighting_and_booth', 'add_patrols', 'full_intervention']
        
        for action in actions:
            effect = self.intervention_effects.get(action, 1.00)
            base_rate = max(risk * effect / 100, 0.1)
            total = 0
            for year in range(1, years + 1):
                if action == 'do_nothing':
                    year_effect = (1 + comp_rate) ** (year - 1)
                else:
                    year_effect = 1
                total += base_rate * year_effect * length_km
            
            incidents_min, incidents_max = max(0, int(total * 0.8)), int(total * 1.2)
            cost_min, cost_max = incidents_min * cost_per_incident[0], incidents_max * cost_per_incident[1]
            
            if action != 'do_nothing':
                invest = self.intervention_costs.get(action, (50000, 150000))
                invest_min, invest_max = int(invest[0] * length_km), int(invest[1] * length_km)
                net_min, net_max = cost_min - invest_max, cost_max - invest_min
            else:
                net_min, net_max = cost_min, cost_max
            
            results[action] = {'incidents': (incidents_min, incidents_max), 'net': (net_min, net_max)}
        
        if severity == 'severe':
            best = 'full_intervention'
        elif severity == 'high':
            best = 'add_patrols'
        elif severity == 'moderate':
            best = 'add_lighting_and_booth'
        else:
            best = 'add_lighting'
        
        return results, best, severity


simulator = CostOfDoingNothingSimulator()
action_names = {
    'do_nothing': 'Do Nothing', 'add_lighting': 'Add Lighting',
    'add_lighting_and_booth': 'Lighting + Booth',
    'add_patrols': 'Patrols', 'full_intervention': 'Full'
}

with compare_col1:
    st.markdown("**Corridor A**")
    corridor_options_a = filtered.head(100).reset_index(drop=True).copy()
    corridor_options_a['display_name'] = corridor_options_a['name'].str[:40] + " (Risk: " + corridor_options_a['predicted_risk'].round(0).astype(int).astype(str) + ")"
    selected_a_idx = st.selectbox("Select corridor A", options=corridor_options_a.index, format_func=lambda i: corridor_options_a.loc[i, 'display_name'], key="compare_a")
    selected_a = corridor_options_a.loc[selected_a_idx]
    results_a, best_a, severity_a = simulator.simulate(selected_a['predicted_risk'], selected_a['length_km'], years)
    
    st.markdown(f"**{selected_a['name']}**")
    st.markdown(f"Risk: {int(selected_a['predicted_risk'])} | Severity: {severity_a.upper()}")
    st.markdown(f"Length: {selected_a['length_km']:.2f} km")
    st.markdown(f"**Best Action:** {action_names[best_a]}")
    st.markdown(f"Prevented: {max(0, results_a['do_nothing']['incidents'][0] - results_a[best_a]['incidents'][1]):,} - {results_a['do_nothing']['incidents'][1] - results_a[best_a]['incidents'][0]:,} incidents")

with compare_col2:
    st.markdown("**Corridor B**")
    corridor_options_b = filtered.head(100).reset_index(drop=True).copy()
    corridor_options_b['display_name'] = corridor_options_b['name'].str[:40] + " (Risk: " + corridor_options_b['predicted_risk'].round(0).astype(int).astype(str) + ")"
    selected_b_idx = st.selectbox("Select corridor B", options=corridor_options_b.index, format_func=lambda i: corridor_options_b.loc[i, 'display_name'], key="compare_b")
    selected_b = corridor_options_b.loc[selected_b_idx]
    results_b, best_b, severity_b = simulator.simulate(selected_b['predicted_risk'], selected_b['length_km'], years)
    
    st.markdown(f"**{selected_b['name']}**")
    st.markdown(f"Risk: {int(selected_b['predicted_risk'])} | Severity: {severity_b.upper()}")
    st.markdown(f"Length: {selected_b['length_km']:.2f} km")
    st.markdown(f"**Best Action:** {action_names[best_b]}")
    st.markdown(f"Prevented: {max(0, results_b['do_nothing']['incidents'][0] - results_b[best_b]['incidents'][1]):,} - {results_b['do_nothing']['incidents'][1] - results_b[best_b]['incidents'][0]:,} incidents")

if st.button("Show Comparison Chart"):
    compare_data = pd.DataFrame([
        {'Corridor': selected_a['name'][:30], 'Do Nothing': results_a['do_nothing']['incidents'][1], 'Full Intervention': results_a['full_intervention']['incidents'][1]},
        {'Corridor': selected_b['name'][:30], 'Do Nothing': results_b['do_nothing']['incidents'][1], 'Full Intervention': results_b['full_intervention']['incidents'][1]}
    ])
    fig_compare = px.bar(compare_data, x='Corridor', y=['Do Nothing', 'Full Intervention'], barmode='group', title=f'Comparison: Do Nothing vs Full Intervention ({years} years)')
    st.plotly_chart(fig_compare, use_container_width=True)

# ============================================================================
# Detailed Corridor Simulation
# ============================================================================

st.markdown("---")
st.subheader("Detailed Simulation for Any Corridor")

corridor_options = filtered.head(200).reset_index(drop=True).copy()
corridor_options['sim_severity'] = corridor_options['predicted_risk'].apply(simulator.get_severity)
corridor_options['display_name'] = (
    corridor_options['name'].str[:50] + 
    " (Risk: " + corridor_options['predicted_risk'].round(0).astype(int).astype(str) + 
    ", Severity: " + corridor_options['sim_severity'].str.upper() + ")"
)

selected_idx = st.selectbox(
    "Select a corridor to analyze",
    options=corridor_options.index,
    format_func=lambda i: corridor_options.loc[i, 'display_name']
)

selected_row = corridor_options.loc[selected_idx]
results, best_action, severity = simulator.simulate(selected_row['predicted_risk'], selected_row['length_km'], years)

# SHAP + LLM explanation for the selected corridor
st.markdown("#### Why this corridor scored this way")
explanation_text = explanations.get(
    str(selected_row['name']),
    "This corridor's risk reflects its overall street characteristics."
)
st.write(explanation_text)
st.caption("Factors computed with SHAP directly from the model. These are the model's own "
           "reasons for the score, stated exactly, with nothing added.")

sim_col1, sim_col2 = st.columns([1, 1])

with sim_col1:
    st.markdown(f"**Corridor:** {selected_row['name']}")
    st.markdown(f"**Length:** {selected_row['length_km']:.2f} km")
    st.markdown(f"**Risk Index:** {int(selected_row['predicted_risk'])} / 100")
    st.markdown(f"**Time Horizon:** {years} years")
    st.markdown(f"**Severity:** {severity.upper()}")
    
    if severity == 'severe':
        st.error("This corridor is SEVERE risk. Full intervention recommended immediately.")
    elif severity == 'high':
        st.warning("This corridor is HIGH risk. Patrols recommended.")
    else:
        st.info("This corridor needs attention based on its risk level.")

with sim_col2:
    st.markdown(f"**Intervention Comparison ({years} years)**")
    comparison_df = pd.DataFrame([{
        'Action': action_names[a],
        'Incidents': f"{results[a]['incidents'][0]:,} - {results[a]['incidents'][1]:,}",
        'Net Savings': f"${results[a]['net'][0]:,.0f} - ${results[a]['net'][1]:,.0f}" if a != 'do_nothing' else "N/A"
    } for a in ['do_nothing', 'add_lighting', 'add_lighting_and_booth', 'add_patrols', 'full_intervention']])
    st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")
if best_action == 'full_intervention':
    st.success(f"**RECOMMENDATION: {action_names[best_action]}** - Severe risk level requires full intervention package")
elif best_action == 'add_patrols':
    st.info(f"**RECOMMENDATION: {action_names[best_action]}** - High risk level best addressed with patrols")
elif best_action == 'add_lighting_and_booth':
    st.info(f"**RECOMMENDATION: {action_names[best_action]}** - Moderate risk benefits from lighting and booth")
else:
    st.info(f"**RECOMMENDATION: {action_names[best_action]}** - Low risk needs basic lighting")

prevented_min = results['do_nothing']['incidents'][0] - results[best_action]['incidents'][1]
prevented_max = results['do_nothing']['incidents'][1] - results[best_action]['incidents'][0]
st.markdown(f"**Impact:** Prevents an estimated {max(0, prevented_min):,} - {prevented_max:,} incidents over {years} years")

chart_data = pd.DataFrame([{
    'Action': action_names[a],
    'Incidents': results[a]['incidents'][1]
} for a in ['do_nothing', 'add_lighting', 'add_lighting_and_booth', 'add_patrols', 'full_intervention']])

fig_compare = px.bar(
    chart_data, x='Action', y='Incidents',
    title=f'Projected Incidents by Intervention ({years} years)',
    color='Action',
    color_discrete_map={
        'Do Nothing': '#e74c3c', 'Add Lighting': '#f1c40f',
        'Lighting + Booth': '#e67e22', 'Patrols': '#3498db',
        'Full': '#2ecc71'
    }
)
st.plotly_chart(fig_compare, use_container_width=True)

# ============================================================================
# Cost-Benefit Analysis Chart
# ============================================================================

st.markdown("---")
st.subheader("Cost-Benefit Analysis")

cba_data = []

for action in ['add_lighting', 'add_lighting_and_booth', 'add_patrols', 'full_intervention']:
    net_savings_min = results[action]['net'][0]
    net_savings_max = results[action]['net'][1]
    net_savings_avg = (net_savings_min + net_savings_max) / 2
    
    if action == 'add_lighting':
        cost_min, cost_max = 50000, 150000
        action_name = 'Add Lighting'
    elif action == 'add_lighting_and_booth':
        cost_min, cost_max = 100000, 250000
        action_name = 'Lighting + Booth'
    elif action == 'add_patrols':
        cost_min, cost_max = 150000, 400000
        action_name = 'Add Patrols'
    else:
        cost_min, cost_max = 300000, 700000
        action_name = 'Full Intervention'
    
    cost_avg = ((cost_min + cost_max) / 2) * selected_row['length_km']
    
    if cost_avg > 0:
        roi_avg = (net_savings_avg / cost_avg) * 100
    else:
        roi_avg = 0
    
    annual_benefit_avg = net_savings_avg / years
    if annual_benefit_avg > 0:
        payback_years = cost_avg / annual_benefit_avg
    else:
        payback_years = float('inf')
    
    cba_data.append({
        'Intervention': action_name,
        'Net Savings (5yr)': net_savings_avg,
        'ROI (%)': roi_avg,
        'Payback (years)': payback_years
    })

cba_df = pd.DataFrame(cba_data)
cba_df = cba_df.sort_values('ROI (%)', ascending=False)

fig_roi = px.bar(
    cba_df,
    x='Intervention',
    y='ROI (%)',
    title=f'Return on Investment by Intervention ({years} years)',
    color='ROI (%)',
    color_continuous_scale=['red', 'yellow', 'green']
)
st.plotly_chart(fig_roi, use_container_width=True)

fig_savings = px.bar(
    cba_df,
    x='Intervention',
    y='Net Savings (5yr)',
    title=f'Net Savings by Intervention ({years} years)',
    color='Intervention',
    color_discrete_sequence=['#f1c40f', '#e67e22', '#3498db', '#2ecc71']
)
st.plotly_chart(fig_savings, use_container_width=True)

payback_df = cba_df[['Intervention', 'ROI (%)', 'Payback (years)']].copy()
payback_df['ROI (%)'] = payback_df['ROI (%)'].round(0).astype(int)
payback_df['Payback (years)'] = payback_df['Payback (years)'].apply(
    lambda x: f'{x:.1f}' if x != float('inf') else '> 10 years'
)
st.dataframe(payback_df, use_container_width=True)

if len(cba_df) > 0:
    best_roi_action = cba_df.loc[cba_df['ROI (%)'].idxmax(), 'Intervention']
    best_roi_value = cba_df['ROI (%)'].max()
    st.info(f"Investment Insight: {best_roi_action} offers the highest ROI at {best_roi_value:.0f}% over {years} years. A balanced approach combining safety and financial priorities is recommended.")

# ============================================================================
# Time-Series Forecast
# ============================================================================

st.markdown("---")
st.subheader("Year-by-Year Projection")

do_nothing_yearly = calculate_yearly_incidents(
    selected_row['predicted_risk'], 
    selected_row['length_km'], 
    'do_nothing', 
    years
)

full_yearly = calculate_yearly_incidents(
    selected_row['predicted_risk'], 
    selected_row['length_km'], 
    'full_intervention', 
    years
)

recommended_yearly = calculate_yearly_incidents(
    selected_row['predicted_risk'], 
    selected_row['length_km'], 
    best_action, 
    years
)

do_nothing_cumulative = calculate_cumulative_incidents(do_nothing_yearly)
full_cumulative = calculate_cumulative_incidents(full_yearly)
recommended_cumulative = calculate_cumulative_incidents(recommended_yearly)

years_list = list(range(1, years + 1))
do_nothing_upper = [x[1] for x in do_nothing_cumulative]
do_nothing_lower = [x[0] for x in do_nothing_cumulative]
recommended_upper = [x[1] for x in recommended_cumulative]
recommended_lower = [x[0] for x in recommended_cumulative]
full_upper = [x[1] for x in full_cumulative]

fig_time = go.Figure()

fig_time.add_trace(go.Scatter(
    x=years_list,
    y=do_nothing_upper,
    mode='lines+markers',
    name='Do Nothing',
    line=dict(color='red', width=3),
    marker=dict(size=8),
    error_y=dict(
        type='data',
        symmetric=False,
        array=[do_nothing_upper[i] - do_nothing_lower[i] for i in range(len(do_nothing_lower))],
        arrayminus=[0] * len(do_nothing_lower),
        visible=True,
        color='rgba(255,0,0,0.3)'
    )
))

action_color = {'add_lighting': '#f1c40f', 'add_lighting_and_booth': '#e67e22', 
                'add_patrols': '#3498db', 'full_intervention': '#2ecc71'}.get(best_action, '#2ecc71')

fig_time.add_trace(go.Scatter(
    x=years_list,
    y=recommended_upper,
    mode='lines+markers',
    name=action_names.get(best_action, 'Recommended'),
    line=dict(color=action_color, width=3),
    marker=dict(size=8),
    error_y=dict(
        type='data',
        symmetric=False,
        array=[recommended_upper[i] - recommended_lower[i] for i in range(len(recommended_lower))],
        arrayminus=[0] * len(recommended_lower),
        visible=True,
        color='rgba(0,255,0,0.3)'
    )
))

fig_time.add_trace(go.Scatter(
    x=years_list,
    y=full_upper,
    mode='lines',
    name='Full Intervention (Reference)',
    line=dict(color='gray', width=2, dash='dash'),
    opacity=0.7
))

fig_time.update_layout(
    title=f'Cumulative Incidents Over Time ({years} Years)',
    xaxis=dict(title='Year', tickmode='linear', tick0=1, dtick=1),
    yaxis=dict(title='Cumulative Incidents'),
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=500
)

fig_time.add_vline(x=years, line_dash="dot", line_color="gray", opacity=0.5)

st.plotly_chart(fig_time, use_container_width=True)

st.subheader("Year-by-Year Breakdown")

yearly_table_data = []
for year in range(1, years + 1):
    do_nothing_lower_y, do_nothing_upper_y = do_nothing_yearly[year-1]
    recommended_lower_y, recommended_upper_y = recommended_yearly[year-1]
    saved_lower = do_nothing_lower_y - recommended_upper_y
    saved_upper = do_nothing_upper_y - recommended_lower_y
    
    yearly_table_data.append({
        'Year': year,
        'Do Nothing Incidents': f'{do_nothing_lower_y:,} - {do_nothing_upper_y:,}',
        f'{action_names.get(best_action, "Recommended")} Incidents': f'{recommended_lower_y:,} - {recommended_upper_y:,}',
        'Incidents Prevented (This Year)': f'{max(0, saved_lower):,} - {saved_upper:,}'
    })

yearly_table_df = pd.DataFrame(yearly_table_data)
st.dataframe(yearly_table_df, use_container_width=True)

if len(do_nothing_lower) > 0 and len(recommended_upper) > 0:
    total_saved_lower = do_nothing_lower[-1] - recommended_upper[-1]
    total_saved_upper = do_nothing_upper[-1] - recommended_lower[-1]
else:
    total_saved_lower = 0
    total_saved_upper = 0

if years > 1:
    st.info(f"""
    Key Insight: The gap between Do Nothing and {action_names.get(best_action, 'Recommended')} widens each year.
    By Year {years}, the cumulative difference reaches {max(0, total_saved_lower):,} - {total_saved_upper:,} incidents prevented.
    Acting now compounds into larger savings over time due to risk compounding.
    """)

# ============================================================================
# Export Report
# ============================================================================

st.markdown("---")
st.subheader("Export Report")

if st.button("Download Report for Selected Corridor"):
    report_html = generate_report(
        selected_row['name'],
        int(selected_row['predicted_risk']),
        severity,
        selected_row['length_km'],
        results,
        best_action,
        years
    )
    
    b64 = base64.b64encode(report_html.encode()).decode()
    safe_name = selected_row['name'].replace("/", "_").replace(" ", "_")
    href = f'<a href="data:text/html;base64,{b64}" download="NightSafe_Report_{safe_name}.html">Click here to download HTML report</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("Report generated. Click the link above to download.")

st.markdown("---")
st.caption("NightSafe - Cost of Doing Nothing Simulator")
st.caption("Data: OpenDataPhilly, OpenStreetMap, US Census, PennDOT | Model: Random Forest (honest R2 = 0.26 after removing leaked features)")
st.caption(f"Time Horizon: {years} years | All outputs are ranges. Human makes final investment decision. Fairness audited quarterly.")
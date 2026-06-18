# NightSafe — Cost of Doing Nothing Simulator

**USAII Global AI Hackathon 2026 · Graduate Track · Brief 6 (Public Systems & Policy)**

NightSafe is a decision-support tool for city safety planners. It scores every
road corridor in Philadelphia by night-safety risk using only structural street
features, then simulates the cost of doing nothing versus investing in lighting,
patrols, or call boxes over 1, 3, and 5 years. The goal is to help a planner
decide *where* to spend a limited safety budget, and to make the cost of delay
visible.

The model deliberately does **not** use crime history, income, or weather as
inputs. It predicts risk from the physical characteristics of a street, so it
does not simply send resources back to wherever policing already concentrated.

---

## What it does

- **Risk scoring.** An ensemble model scores ~4,100 Philadelphia corridors on a
  0–100 risk index from eight structural features (street length, lighting,
  pole density, distance to nearest help, intersection controls, vehicle
  traffic volume, and a monitored-road flag).
- **Cost-of-doing-nothing simulator.** For each corridor it projects incidents
  and dollar costs over 1/3/5 years for five options (do nothing, add lighting,
  lighting + call box, patrols, full intervention), as ranges, not point claims.
- **Explainability.** SHAP computes why each corridor scored as it did, in plain
  language ("distance from help and intersection controls raised this corridor's
  risk").
- **Fairness oversight.** A demographic-parity audit by income quartile, plus a
  human-review trigger for high-risk corridors in the lowest-income areas.
- **Dashboard.** A Streamlit app with an interactive map, corridor comparison,
  the simulator, cost-benefit/ROI, and per-corridor explanations.

---

## The model (honest numbers)

- Ensemble of Random Forest, XGBoost, LightGBM, and a neural network.
- Trained on 2022–2024 incidents to predict the 2025–2026 average (a real
  hold-out in time), then used to score all corridors.
- **Random Forest R² = 0.26, Recall@10% = 0.33** on the held-out test set.
- This number is intentionally honest. We removed three sources of data leakage
  (a poles-per-incident feature, income as a predictor, and a weather-from-crime
  feature). Each removal lowered R² but made the model defensible. A higher R²
  on this problem usually means the target has leaked into the features.

Top features by SHAP: intersection controls, monitored-road status, distance
from help, pole density.

---

## Responsible AI

- **Fairness journey.** Measured demographic-parity gap across income quartiles.
  With a geographic income proxy the gap was 0.34; adding real income as a
  predictor pushed it to 0.62 (the model was using poverty as a proxy for
  danger); removing income from the model entirely brought it to ~0.25. We
  accept lower accuracy for a fairer tool.
- **Income is oversight, not input.** Income is used only for the fairness audit
  and to flag high-risk low-income corridors for human review (~6,400 segments),
  never as a model feature.
- **Crime history is context, not input.** The dashboard shows past reported
  crime beside the model score, clearly labeled, but the model never uses it,
  to avoid reinforcing where policing already concentrates. The default ranking
  is the model score; viewing by crime history is optional.
- **Human-in-the-loop.** The tool ranks and explains; a human planner makes the
  final funding decision. All outputs are ranges.
- **Re-audit on every retrain.** Fairness is monitored over the lifecycle, not
  measured once.

---

## Data sources

All public. Download into `Data/raw/` (also mirrored on Google Drive — https://drive.google.com/drive/folders/1OQntLM_ac7X8NBTJKDf0l7CoWN5IJ8Fv).

- Philadelphia crime incidents 2022–2026 — OpenDataPhilly
- Street poles (lighting proxy) — OpenDataPhilly
- Police stations — OpenDataPhilly
- PA hospitals — OpenDataPhilly
- Census tracts (2010) — OpenDataPhilly / US Census
- Median household income (ACS 2019, B19013) — US Census API
- PA DOT traffic counts (RMSTRAFFIC) — PennShare / OpenDataPhilly
- Intersection controls — OpenDataPhilly
- Police districts — OpenDataPhilly
- Road network — OpenStreetMap (fetched in code via osmnx)

Weather (NOAA) was evaluated but excluded from the model (it leaked and is
citywide); it appears in the dashboard as live context only.

---

## Project structure
 
```
Code
├── fetch_census_income.py        # pulls ACS median income (run once)
├── phase1.py                     # builds the feature table
├── phase2.py                     # trains the ensemble, fairness audit
├── phase3.py                     # cost-of-doing-nothing simulator
├── phase4.py                     # SHAP explanations per corridor
├── dashboard.py                  # Streamlit decision-support app
├── Data_Exploration.ipynb        # initial dataset inspection
├── Data_Preprocessing.ipynb      # income fetch + data checks
├── requirements.txt
├── .env                          # API keys (not committed)
│
├── Data/raw/                     # public source data (see Data sources)
    ├── Crime data 2022..2026.csv
    ├── Street_Poles.csv
    ├── Police_Stations.csv
    ├── pa_hospitals.geojson
    ├── Intersection_Controls.geojson
    ├── roadwaytraffic.geojson    # PennDOT RMSTRAFFIC
    ├── tract_income.csv          # from the Census API
    ├── Weather.csv               # context only, not a model input
    ├── Census_Tracts_2010-shp/
    └── police_districts/
```

The pipeline runs in order (phase1 → phase4, then the dashboard); each phase
reads the previous phase's output from `Data/processed/`.

## How to run

Requires Python 3.11. Install dependencies, then:

```
pip install -r requirements.txt
# First, fetch Census income: run Data_Preprocessing.ipynb
# (needs a free Census API key in .env). It writes Data/raw/tract_income.csv
python phase1.py                  # build the feature table
python phase2.py                  # train models, fairness audit
python phase3.py                  # cost-of-doing-nothing simulator
python phase4.py                  # SHAP explanations
streamlit run dashboard.py        # launch the dashboard
```

Environment variables (in a `.env` file, not committed):

```
CENSUS_API_KEY=your_key
OPEN_WEATHER_API_KEY=your_key   # optional, for the live weather widget
```

### Getting the API keys

**Census API key (required for income data).** Free and instant.
1. Go to https://api.census.gov/data/key_signup.html
2. Enter your name, email, and organization (any is fine).
3. The key arrives by email within a minute. Paste it into `.env` as
   `CENSUS_API_KEY`.

**OpenWeather API key (optional, live weather widget only).** Free tier.
1. Create a free account at https://home.openweathermap.org/users/sign_up
2. Open the "API keys" tab in your account and copy the default key (new keys
   can take up to an hour to activate).
3. Paste it into `.env` as `OPEN_WEATHER_API_KEY`.

The dashboard runs fine without the OpenWeather key — the weather widget simply
shows "unavailable" and nothing else is affected, since weather is display-only
and not a model input.

---

## Limitations (stated honestly)

- Risk is predicted from structural proxies, not measured danger. The model
  finds streets whose characteristics match higher-incident streets; it is a
  prioritization aid, not a prediction of specific events.
- Foot traffic is estimated from road type, not measured.
- Traffic counts cover ~60% of segments (mostly major roads); the rest use a
  citywide median, flagged by the monitored-road feature.
- Income is a 2019 ACS proxy joined on 2010 tracts.
- Intended for prioritizing streets that already have data history, not for
  scoring entirely unmonitored neighborhoods.

---

## Tools used

Python, pandas, GeoPandas, scikit-learn, XGBoost, LightGBM, OSMnx, SHAP, Plotly,
Streamlit, requests. AI coding assistant (Claude) used for development support;
all design decisions, leak removal, and fairness analysis directed by the team.

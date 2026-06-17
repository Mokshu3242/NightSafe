import os
import glob
import pandas as pd
import requests
import geopandas as gpd
from dotenv import load_dotenv

load_dotenv()  

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

DATA_RAW = 'Data/raw'
OUTPUT_CSV = DATA_RAW + '/tract_income.csv'
CENSUS_NO_DATA = -666666666
def fetch_income(year):
    base = "https://api.census.gov/data/" + str(year) + "/acs/acs5"
    params = {
        "get": "NAME,B19013_001E",
        "for": "tract:*",
        "in": "state:42 county:101",
        "key": CENSUS_API_KEY,
    }
    headers = {"User-Agent": "NightSafe-hackathon/1.0"}

    print("Requesting ACS " + str(year) + " ...")
    resp = requests.get(base, params=params, headers=headers, timeout=60)
    print("  HTTP status: " + str(resp.status_code))

    text_head = resp.text[:300].strip()
    if not text_head.startswith("["):
        print("  Server did not return JSON. First 300 characters:")
        print("  " + text_head.replace("\n", " "))
        raise ValueError("Non-JSON response from ACS " + str(year))

    data = resp.json()
    header, rows = data[0], data[1:]
    print("  Received " + str(len(rows)) + " tract records")
    return pd.DataFrame(rows, columns=header)
def clean_income(df):
    df["GEOID10"] = df["state"] + df["county"] + df["tract"]
    df["median_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    before = len(df)
    df.loc[df["median_income"] <= CENSUS_NO_DATA, "median_income"] = pd.NA
    df.loc[df["median_income"] < 0, "median_income"] = pd.NA
    df = df[["GEOID10", "median_income"]].dropna()
    print("Kept " + str(len(df)) + " tracts with valid income (dropped "
          + str(before - len(df)) + ")")
    return df
def verify_join(income_df):
    shp = glob.glob(DATA_RAW + '/Census_Tracts_2010-shp/*.shp')
    if not shp:
        print("Warning: tract shapefile not found, skipping join check")
        return
    tracts = gpd.read_file(shp[0])
    tracts["GEOID10"] = tracts["GEOID10"].astype(str)
    income_df["GEOID10"] = income_df["GEOID10"].astype(str)
    matched = tracts["GEOID10"].isin(set(income_df["GEOID10"])).sum()
    total = len(tracts)
    print("Join check: " + str(matched) + " of " + str(total)
          + " shapefile tracts matched income data")
    if matched < total * 0.8:
        print("Warning: under 80 percent matched. GEOID formats may differ.")
        print("Sample shapefile GEOID10: " + str(tracts['GEOID10'].iloc[0]))
        print("Sample income GEOID10:    " + str(income_df['GEOID10'].iloc[0]))
def main():
    if CENSUS_API_KEY == "PUT_YOUR_KEY_HERE":
        raise SystemExit(
            "No API key set. Get a free key at "
            "https://api.census.gov/data/key_signup.html and paste it into "
            "CENSUS_API_KEY at the top of this file."
        )

    os.makedirs(DATA_RAW, exist_ok=True)

    df_raw = None
    for year in [2019, 2018, 2021, 2022]:
        try:
            df_raw = fetch_income(year)
            print("  Using ACS " + str(year))
            break
        except Exception as e:
            print("  ACS " + str(year) + " failed: " + str(e))
            print("")
    if df_raw is None:
        raise SystemExit("Got a key error or non-JSON for every year. "
                         "Double-check the key was pasted correctly.")

    income = clean_income(df_raw)
    verify_join(income)
    income.to_csv(OUTPUT_CSV, index=False)
    print("")
    print("Saved " + OUTPUT_CSV)
    print("Income range: " + str(int(income["median_income"].min()))
          + " to " + str(int(income["median_income"].max())))
    print("Median across tracts: " + str(int(income["median_income"].median())))
    print("Done. Now run Phase 1 with the real-income join.")
if __name__ == "__main__":
    main()

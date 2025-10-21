# # Purpose: central configuration, data access, weather fetch stub,
# # feature engineering helpers, and model selection/training
# # ================================================================
import pandas as pd
import numpy as np
import requests
from datetime import timezone
from sqlalchemy import create_engine, text
import holidays

# --- DB CONFIG ---
DATABASE_URL = "mysql+pymysql://flexCOCKPITread:tzUEWdtXdBCv5yyt@192.168.20.9:3306/flexcockpit?charset=utf8mb4"
IO_ID     = 31
START_TS  = "2022-05-01 00:00:00"   # <-- set your window
END_TS    = "2025-06-01 00:00:00"

# --- PLANT COORDINATES (set your site!) ---
LAT, LON = 49.96, 8.27  # example

# Germany holidays (you can change to holidays.CountryHoliday('DE', prov='RP') for Rheinland-Pfalz)
GER_HOLIDAYS = holidays.CountryHoliday('DE', prov='RP')

# --- 1) Query hourly-bucketed values from MariaDB (robust epoch bucketing) ---
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

stmt = text("""
SELECT
  FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(arTimestamp - INTERVAL 1 MINUTE)/3600)*3600) AS hour_start,
  AVG(arValue) AS value
FROM archive
WHERE arIoId = :io_id
  AND arTimestamp > :start_ts
  AND arTimestamp <= :end_ts
  AND arIsAverage = 1
GROUP BY hour_start
ORDER BY hour_start ASC;
""")

params = {"io_id": IO_ID, "start_ts": START_TS, "end_ts": END_TS}

with engine.connect() as conn:
    df = pd.read_sql(stmt, conn, params=params, parse_dates=["hour_start"])

# normalize to UTC (assumes DB is UTC; if not, localize appropriately first)
df["timestamp"] = pd.to_datetime(df["hour_start"], utc=True)
df = df.drop(columns=["hour_start"]).sort_values("timestamp")

# --- 2) Fetch hourly weather from Open-Meteo (archive) for the same date range ---
def get_weather_data(lat, lon, start_utc, end_utc):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_utc.date().isoformat(),
        "end_date":   end_utc.date().isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,rain,snowfall",
        "timezone": "UTC"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"], utc=True),
        "temperature": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"],
        "rain": h["rain"],
        "snowfall": h["snowfall"],
    })
    return w

start_utc = df["timestamp"].min().to_pydatetime()
end_utc   = df["timestamp"].max().to_pydatetime()
weather   = get_weather_data(LAT, LON, start_utc, end_utc)


# --- 3) Simple features ---
def make_features(df):
    d = df.sort_values("timestamp").set_index("timestamp")
    d["hour"] = d.index.hour
    d["day_of_week"] = d.index.dayofweek
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["date"] = d.index.date
    d["is_holiday"] = d["date"].isin(GER_HOLIDAYS).astype(int)
    d["is_bank_holiday"] = d["is_holiday"]
    d["lag_1"] = d["value"].shift(1)
    d["lag_24"] = d["value"].shift(24)
    d["roll3"] = d["value"].rolling(3, min_periods=2).mean()
    d["roll6"] = d["value"].rolling(6, min_periods=3).mean()
    d["roll24"] = d["value"].rolling(24, min_periods=12).mean()
    d["dow"] = d.index.dayofweek
    d = d.drop(columns=["date"]).dropna()
    return d.reset_index()

# features= make_features(df)
# print(features.head())


# --- 3) Merge & preview ---
merged = df.merge(weather, on="timestamp", how="left").sort_values("timestamp")
data = make_features(merged)
def get_final_data():
    """Return final feature-ready DataFrame for training/prediction."""
    return data
print(data.head())
print("Rows:", len(merged), "Range:", merged["timestamp"].min(), "→", merged["timestamp"].max())


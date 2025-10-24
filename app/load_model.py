# app/load_model.py
import os, json, joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from tensorflow import keras
from datetime import datetime

# ---------- small helpers ----------
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _time_feats(ts: pd.Timestamp) -> Dict[str, float]:
    hour = int(ts.hour)
    dow = int(ts.dayofweek)
    is_weekend = int(dow >= 5)
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    return {
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_holiday": 0.0,
        "is_bank_holiday": 0.0,
    }

def _ensure_cols(df: pd.DataFrame, cols: list, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")

# ---------- main API ----------
def predict_iterative_hours_to_excel(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    model_dir: str,
    hours: int = 24,
    excel_name_prefix: str = "forecast",
) -> Tuple[pd.DataFrame, str]:
    """
    Iterative 1-step roll-forward prediction for `hours`.
    Uses saved model + scalers + feature order from metadata.json.
    Saves an Excel (xlsx) file and returns (forecast_df, excel_path).

    history_df columns (at least):
      - 'timestamp'
      - all features used in training, including engineered ones if any
      - target column (meta['target_col'])

    future_df columns (at least):
      - 'timestamp', 'temperature', 'rain', 'humidity', 'snowfall'
      (any other exogenous you need can be added similarly)
    """
    # --- load artifacts ---
    meta_path = os.path.join(model_dir, "metadata.json")
    model_path = os.path.join(model_dir, "final_model.keras")
    xsc_path   = os.path.join(model_dir, "scaler_X.joblib")
    ysc_path   = os.path.join(model_dir, "scaler_y.joblib")
    for p in [meta_path, model_path, xsc_path, ysc_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols: list = meta["feature_cols"]        # order must match training
    target_col: str   = meta["target_col"]
    timesteps: int    = int(meta["timesteps"])
    model             = keras.models.load_model(model_path)
    X_scaler          = joblib.load(xsc_path)
    y_scaler          = joblib.load(ysc_path)

    # --- basic checks ---
    _ensure_cols(history_df, ["timestamp", target_col], "history_df")
    _ensure_cols(future_df,  ["timestamp", "temperature", "rain", "humidity", "snowfall"], "future_df")

    # Make sure we have enough window
    if len(history_df) < timesteps:
        raise ValueError(f"Not enough history rows for timesteps={timesteps} (have {len(history_df)})")
    if len(future_df) < hours:
        raise ValueError(f"future_df has only {len(future_df)} rows but hours={hours}")

    # Sort by time
    hist = history_df.sort_values("timestamp").reset_index(drop=True).copy()
    fut  = future_df.sort_values("timestamp").reset_index(drop=True).copy()

    # Keep past values in ORIGINAL scale (for lags/rolls)
    past_vals = hist[target_col].astype(float).tolist()

    # Starting scaled window for features (last T rows)
    # If your history already contains all feature_cols computed during training, you can scale directly.
    # Otherwise, we will construct rows per hour using both history and fut exogenous below.
    # Here we start with the last timesteps feature rows from history (where present), else we build them.
    # Safer: reconstruct from history using the same feature builder used for future hours.
    # We'll build a small helper to assemble a single feature row dict for any given timestamp:

    # cached constants / last-knowns
    aeration_index_const = float(hist.get("aeration_index", pd.Series([0])).iloc[-1]) if "aeration_index" in hist.columns else 0.0

    def _engineer_row(ts: pd.Timestamp, i_future: int) -> Dict[str, float]:
        """
        Build a feature row for future hour index i_future (0..hours-1),
        using future exogenous at that index and lags/rolls from `past_vals`.
        Missing engineered features default to 0.0 unless filled below.
        """
        # exogenous from future_df
        temperature = float(fut.at[i_future, "temperature"])
        rain        = float(fut.at[i_future, "rain"])
        humidity    = float(fut.at[i_future, "humidity"])
        snowfall    = float(fut.at[i_future, "snowfall"])
        tf = _time_feats(ts)

        # lags/roll in ORIGINAL units
        lag_1  = past_vals[-1]
        lag_24 = past_vals[-24] if len(past_vals) >= 24 else past_vals[0]
        r3     = float(np.mean(past_vals[-3:])) if len(past_vals) >= 3 else float(np.mean(past_vals))
        r6     = float(np.mean(past_vals[-6:])) if len(past_vals) >= 6 else float(np.mean(past_vals))

        # engineered rain history using FUTURE rain seen so far
        rain_hist = fut.loc[max(0, i_future-5):i_future, "rain"].astype(float).tolist()
        rain_last3h = float(np.mean(rain_hist[-3:])) if len(rain_hist) >= 1 else 0.0
        rain_last6h = float(np.mean(rain_hist[-6:])) if len(rain_hist) >= 1 else 0.0

        base = {
            "temperature": temperature,
            "rain": rain,
            "humidity": humidity,
            "snowfall": snowfall,
            "hour": tf["hour"],
            "day_of_week": tf["day_of_week"],
            "is_weekend": tf["is_weekend"],
            "hour_sin": tf["hour_sin"],
            "hour_cos": tf["hour_cos"],
            "is_holiday": tf["is_holiday"],
            "is_bank_holiday": tf["is_bank_holiday"],
            "value_lag_1h": lag_1,
            "value_lag_24h": lag_24,
            "value_roll_mean_3h": r3,
            "value_roll_mean_6h": r6,
            "aeration_index": aeration_index_const,
            "rain_last3h": rain_last3h,
            "rain_last6h": rain_last6h,
        }

        # compute optional engineered terms if your metadata featured them
        if "rain_temp" in feature_cols:
            base["rain_temp"] = rain * temperature
        if "rain_weekend" in feature_cols:
            base["rain_weekend"] = rain * tf["is_weekend"]
        if "rain_binary" in feature_cols:
            base["rain_binary"] = 1.0 if rain > 0 else 0.0

        # For any feature present in feature_cols but not in base, fill with last-known history value if available; else 0.
        for col in feature_cols:
            if col not in base:
                if col in hist.columns:
                    base[col] = float(hist[col].iloc[-1])
                else:
                    base[col] = 0.0
        return base

    # Build initial scaled window from the last T timestamps using history rows:
    # We will reconstruct feature rows for those last T timestamps using the same logic but with i_future=-1 and future exog taken from last known history values.
    # Simpler: if all feature_cols exist in history, use them directly:
    have_all_hist_feats = all(c in hist.columns for c in feature_cols)
    if have_all_hist_feats:
        hist_window_raw = hist[feature_cols].iloc[-timesteps:].values
    else:
        # fallback: synthesize using last-known exog from history (temperature/rain/etc.) if present
        synth_rows = []
        last_hist_exog = {
            k: float(hist.get(k, pd.Series([0.0])).iloc[-1])
            for k in ["temperature","rain","humidity","snowfall"]
        }
        last_ts_idx = hist.index[-timesteps:]
        for _ in range(timesteps):
            ts = pd.to_datetime(hist["timestamp"].iloc[-timesteps])  # approx
            # reuse engineered builder but override with last_hist_exog
            row = _engineer_row(ts, 0)
            row.update(last_hist_exog)
            synth_rows.append([row[c] for c in feature_cols])
        hist_window_raw = np.array(synth_rows, dtype=float)

    window_scaled = X_scaler.transform(hist_window_raw)  # (T, F)

    # --- roll forward predictions ---
    out_rows = []
    for i in range(hours):
        ts = pd.to_datetime(fut.at[i, "timestamp"])
        feat_dict = _engineer_row(ts, i)
        row_raw   = np.array([feat_dict[c] for c in feature_cols], dtype=float).reshape(1, -1)
        row_scaled = X_scaler.transform(row_raw)  # (1,F)

        # advance the window
        window_scaled = np.vstack([window_scaled[1:], row_scaled])  # (T,F)
        x_input = window_scaled.reshape(1, timesteps, len(feature_cols))

        yhat_scaled = model.predict(x_input, verbose=0)
        # support Dense(1) or Dense(H) models: take first step
        if yhat_scaled.ndim == 2:
            next_scaled = yhat_scaled[:, 0]
        else:
            next_scaled = yhat_scaled.flatten()[0:1]

        next_val = float( y_scaler.inverse_transform(next_scaled.reshape(-1,1))[0,0] )
        next_val = max(0.0, next_val)  # clamp negatives
        out_rows.append((ts, next_val))
        past_vals.append(next_val)     # update lags in ORIGINAL units

    forecast_df = pd.DataFrame(out_rows, columns=["timestamp", "predicted_value"])

    # --- save to Excel (timestamped inside outputs/YYYYMMDD) ---
    day_folder = datetime.now().strftime("%Y%m%d")
    out_dir = os.path.join(model_dir, "outputs", day_folder)
    os.makedirs(out_dir, exist_ok=True)
    excel_path = os.path.join(out_dir, f"{excel_name_prefix}_{hours}h_{_ts()}.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        forecast_df.to_excel(writer, index=False, sheet_name="forecast")

    return forecast_df, excel_path

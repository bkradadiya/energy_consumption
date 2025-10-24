# # save_model.py
# import os, json, joblib, numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras import Sequential, Input
# from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense, BatchNormalization

# def _create_sequences(data_np, timesteps=240):
#     """
#     Builds (X, y) where:
#       X: (n_samples, timesteps, n_features)   features only
#       y: (n_samples, OUTPUT_STEPS)            next-24 targets (last column)
#     """
#     X, y = [], []
#     for i in range(len(data_np) - timesteps):
#         X.append(data_np[i:i+timesteps, :-1])  # features
#         y.append(data_np[i+timesteps, -1])     # target (last col)
#     return np.array(X), np.array(y)

# def _split(A, val_ratio=0.2, test_ratio=0.2):
#     n = A.shape[0]; n_val = int(val_ratio*n); n_test = int(test_ratio*n); n_train = n-n_val-n_test
#     return A[:n_train], A[n_train:n_train+n_val], A[n_train+n_val:]
# timesteps = 24
# n_features = X.shape[2] 
# def _build_model(timesteps, n_features):
#     m = Sequential([
#         Input(shape=(timesteps, n_features)),
#         Conv1D(32, 3, activation="relu", padding="causal", dilation_rate=1),
#         BatchNormalization(),
#         Conv1D(32, 3, activation="relu", padding="causal", dilation_rate=2),
#         BatchNormalization(),
#         LSTM(64, return_sequences=True),
#         Dropout(0.2),
#         LSTM(32),
#         Dense(24)  # next 24 hours direct multi-step
#     ])
#     m.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
#               loss=tf.keras.losses.Huber(delta=5.0),
#               metrics=[tf.keras.metrics.MeanSquaredError(name="mse"),
#                        tf.keras.metrics.MeanAbsoluteError(name="mae")])
#     return m

# def train_and_save(data, target_col="value", timesteps=240, out_dir="models"):
#     os.makedirs(out_dir, exist_ok=True)

#     # clean & select columns
#     data = data.replace([np.inf, -np.inf], np.nan).dropna()
#     feature_cols = [c for c in data.columns if c not in ["timestamp", target_col]]
#     if not feature_cols:
#         raise ValueError("No feature columns besides target.")

#     # scale (fit on all for simplicity in this test flow)
#     X_scaler = MinMaxScaler().fit(data[feature_cols].values)
#     y_scaler = MinMaxScaler().fit(data[[target_col]].values)
#     X_scaled = X_scaler.transform(data[feature_cols].values)
#     y_scaled = y_scaler.transform(data[[target_col]].values)
#     XY = np.concatenate([X_scaled, y_scaled], axis=1)

#     # sequences & splits
#     X, Y = _create_sequences(XY, timesteps=timesteps)
#     x_tr, x_val, x_te = _split(X)
#     y_tr, y_val, y_te = _split(Y)

#     model = _build_model(timesteps, n_features=X.shape[2])
#     cbs = [
#         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
#         tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=12, restore_best_weights=True, verbose=1)
#     ]
#     model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=50, batch_size=64, callbacks=cbs, verbose=1)

#     # save artifacts
#     model.save(os.path.join(out_dir, "final_model.keras"))
#     joblib.dump(X_scaler, os.path.join(out_dir, "scaler_X.joblib"))
#     joblib.dump(y_scaler, os.path.join(out_dir, "scaler_y.joblib"))
#     meta = {
#         "feature_cols": feature_cols,
#         "target_col": target_col,
#         "timesteps": timesteps,
#         "output_steps": 24
#     }
#     with open(os.path.join(out_dir, "metadata.json"), "w") as f:
#         json.dump(meta, f, indent=2)

#     return {"model_dir": out_dir, "meta": meta}



# ===============================================================
# save_model.py
# Purpose: Train and save a multi-step forecasting model
# ===============================================================
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense, BatchNormalization
# from app.plots import save_training_curves, save_test_sample_plot, save_history_json
# from app.plots import save_actual_vs_predicted, save_avp_arrays # Graph 1
from app.plots import save_series_avp_with_time, save_last48_avp


# ---------------------------------------------------------------
# Sequence generator (multi-step)
# ---------------------------------------------------------------
def _create_sequences(data_np, timesteps=240, output_steps=24):
    """
    Builds (X, y) pairs for multi-step forecasting.

    X: (samples, timesteps, n_features)
    y: (samples, output_steps)
    """
    X, y = [], []
    n_total = len(data_np)
    last_col = data_np[:, -1]  # target column (scaled)

    end = n_total - timesteps - output_steps + 1
    for i in range(end):
        X.append(data_np[i:i + timesteps, :-1])                   # feature window
        y.append(last_col[i + timesteps : i + timesteps + output_steps])  # next output_steps values

    return np.asarray(X, np.float32), np.asarray(y, np.float32)

# ---------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------
def _split(A, val_ratio=0.2, test_ratio=0.2):
    n = A.shape[0]
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    n_train = n - n_val - n_test
    return A[:n_train], A[n_train:n_train + n_val], A[n_train + n_val:]

# ---------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------
def _build_model(timesteps, n_features, output_steps=24):
    m = Sequential([
        Input(shape=(timesteps, n_features)),

        # Convolutional front-end to learn temporal features
        Conv1D(32, 3, activation="relu", padding="causal", dilation_rate=1),
        BatchNormalization(),
        Conv1D(32, 3, activation="relu", padding="causal", dilation_rate=2),
        BatchNormalization(),
        # LSTM backbone
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),

        # Fully connected output: direct multi-step prediction
        Dense(output_steps)
    ])

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.Huber(delta=5.0),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae")
        ]
    )
    return m

# ---------------------------------------------------------------
# Main training + saving function
# ---------------------------------------------------------------
def train_and_save(data, target_col="value", timesteps=240, output_steps=24, out_dir="models_seq"):
    """
    Trains a Conv1D+LSTM multi-step forecasting model and saves artifacts.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with features + target
    target_col : str
        Target column name
    timesteps : int
        Lookback window size
    output_steps : int
        How many hours to predict ahead (e.g., 24)
    out_dir : str
        Directory to save model + scalers + metadata
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Clean data ---
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = [c for c in data.columns if c not in ["timestamp", target_col]]
    if not feature_cols:
        raise ValueError("No feature columns besides target.")

    # --- Scaling ---
    X_scaler = MinMaxScaler().fit(data[feature_cols].values)
    y_scaler = MinMaxScaler().fit(data[[target_col]].values)
    X_scaled = X_scaler.transform(data[feature_cols].values)
    y_scaled = y_scaler.transform(data[[target_col]].values)
    XY = np.concatenate([X_scaled, y_scaled], axis=1)

    # --- Build sequences ---
    X, Y = _create_sequences(XY, timesteps=timesteps, output_steps=output_steps)

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    assert Y.ndim == 2 and Y.shape[1] == output_steps, f"Y shape mismatch {Y.shape}"

    # --- Split data ---
    x_tr, x_val, x_te = _split(X)
    y_tr, y_val, y_te = _split(Y)
    print("Train:", x_tr.shape, y_tr.shape, "Val:", x_val.shape, y_val.shape)

    # --- Build + train model ---
    model = _build_model(timesteps, n_features=X.shape[2], output_steps=output_steps)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=12, restore_best_weights=True, verbose=1)
    ]
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # ------------ For Normal Grap ---------------
    # yhat_te_scaled = model.predict(x_te, verbose=0)            # (N_test, output_steps)
    # yhat_last_scaled = yhat_te_scaled[-1].reshape(-1, 1)       # (H, 1)
    # y_true_last_scaled = y_te[-1].reshape(-1, 1)               # (H, 1)

    # # inverse-transform both with target scaler
    # y_pred_last = y_scaler.inverse_transform(yhat_last_scaled).ravel()  # (H,)
    # y_true_last = y_scaler.inverse_transform(y_true_last_scaled).ravel()# (H,)

    # # save arrays for later re-plotting (no inference needed)
    # save_avp_arrays(y_true_last, y_pred_last, out_dir)

    # # save a timestamped AVP figure now
    # save_actual_vs_predicted(y_true_last, y_pred_last, out_dir, show=False)

# ------------ For Normal Graph with 48 Hour ---------------
    yhat_te_scaled = model.predict(x_te, verbose=0)           # (N_test, output_steps)
    y1_pred_scaled = yhat_te_scaled[:, 0].reshape(-1, 1)      # first-step prediction for each test sample
    y1_true_scaled = y_te[:, 0].reshape(-1, 1)
    # Inverse-transform to original units
    y1_pred = y_scaler.inverse_transform(y1_pred_scaled).ravel()
    y1_true = y_scaler.inverse_transform(y1_true_scaled).ravel()

    # --- Build timestamps for the first-step target of each test window ---
    timestamps = pd.to_datetime(data["timestamp"]) if "timestamp" in data.columns else pd.date_range("2000-01-01", periods=len(data), freq="H")

    # Compute ALL sample-level first-target timestamps
    n_total_samples = len(data) - timesteps - meta["output_steps"] + 1 if isinstance(meta := {"output_steps": 24}, dict) else None
    # But we already have X, Y built; match their total:
    n_total_samples = X.shape[0] + 0  # total windows created

    # The first target row index for sample k (relative to the original dataframe) is:
    # base_idx = timesteps + k
    all_first_target_ts = timestamps[timesteps : timesteps + n_total_samples]

    # Now split timestamps exactly like X/Y were split
    ts_tr, ts_val, ts_te = _split(all_first_target_ts.to_numpy())

    # --- Save the full time-series AVP plot and the last-48 zoom ---
    save_series_avp_with_time(ts_te, y1_true, y1_pred, out_dir, title="Predicted vs Actual — Test Set", show=False)
    save_last48_avp(ts_te, y1_true, y1_pred, out_dir, title="Predicted vs Actual (Last 48 Hours)", show=False)


    # --- Save artifacts ---
    model.save(os.path.join(out_dir, "final_model.keras"))
    joblib.dump(X_scaler, os.path.join(out_dir, "scaler_X.joblib"))
    joblib.dump(y_scaler, os.path.join(out_dir, "scaler_y.joblib"))
    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "timesteps": timesteps,
        "output_steps": output_steps,
        "n_features": len(feature_cols)
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Model trained and saved in {out_dir}")
    return {"model_dir": out_dir, "meta": meta}

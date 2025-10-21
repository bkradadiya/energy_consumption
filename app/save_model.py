# save_model.py
import os, json, joblib, numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense, BatchNormalization

def _create_sequences(data_np, timesteps=240):
    X, y = [], []
    for i in range(len(data_np) - timesteps):
        X.append(data_np[i:i+timesteps, :-1])  # features
        y.append(data_np[i+timesteps, -1])     # target (last col)
    return np.array(X), np.array(y)

def _split(A, val_ratio=0.2, test_ratio=0.2):
    n = A.shape[0]; n_val = int(val_ratio*n); n_test = int(test_ratio*n); n_train = n-n_val-n_test
    return A[:n_train], A[n_train:n_train+n_val], A[n_train+n_val:]

def _build_model(timesteps, n_features):
    m = Sequential([
        Input(shape=(timesteps, n_features)),
        Conv1D(32, 3, activation="relu", padding="causal", dilation_rate=1),
        BatchNormalization(),
        Conv1D(32, 3, activation="relu", padding="causal", dilation_rate=2),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(24)  # next 24 hours direct multi-step
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss=tf.keras.losses.Huber(delta=5.0),
              metrics=[tf.keras.metrics.MeanSquaredError(name="mse"),
                       tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return m

def train_and_save(data, target_col="value", timesteps=240, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)

    # clean & select columns
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = [c for c in data.columns if c not in ["timestamp", target_col]]
    if not feature_cols:
        raise ValueError("No feature columns besides target.")

    # scale (fit on all for simplicity in this test flow)
    X_scaler = MinMaxScaler().fit(data[feature_cols].values)
    y_scaler = MinMaxScaler().fit(data[[target_col]].values)
    X_scaled = X_scaler.transform(data[feature_cols].values)
    y_scaled = y_scaler.transform(data[[target_col]].values)
    XY = np.concatenate([X_scaled, y_scaled], axis=1)

    # sequences & splits
    X, Y = _create_sequences(XY, timesteps=timesteps)
    x_tr, x_val, x_te = _split(X)
    y_tr, y_val, y_te = _split(Y)

    model = _build_model(timesteps, n_features=X.shape[2])
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=12, restore_best_weights=True, verbose=1)
    ]
    model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=50, batch_size=64, callbacks=cbs, verbose=1)

    # save artifacts
    model.save(os.path.join(out_dir, "final_model.keras"))
    joblib.dump(X_scaler, os.path.join(out_dir, "scaler_X.joblib"))
    joblib.dump(y_scaler, os.path.join(out_dir, "scaler_y.joblib"))
    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "timesteps": timesteps,
        "output_steps": 24
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"model_dir": out_dir, "meta": meta}

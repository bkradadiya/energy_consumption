# # app/plots.py
# import os
# from datetime import datetime
# import numpy as np
# import matplotlib.pyplot as plt

# def _ts():
#     return datetime.now().strftime("%Y%m%d_%H%M%S")

# def ensure_graphs_dir(model_dir: str) -> str:
#     gdir = os.path.join(model_dir, "graphs")
#     os.makedirs(gdir, exist_ok=True)
#     return gdir

# def save_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_dir: str, show: bool = False) -> str:
#     """
#     Save a timestamped Actual vs Predicted line plot.
#     y_true, y_pred: shape (H,)
#     """
#     graphs_dir = ensure_graphs_dir(model_dir)
#     fname = f"{_ts()}_actual_vs_pred.png"
#     path = os.path.join(graphs_dir, fname)

#     plt.figure()
#     plt.plot(y_true, label="actual")
#     plt.plot(y_pred, label="predicted")
#     plt.title("Actual vs Predicted (last test sample)")
#     plt.xlabel("Horizon hour")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.savefig(path, bbox_inches="tight")
#     if show: plt.show()
#     plt.close()
#     return path

# def save_avp_arrays(y_true: np.ndarray, y_pred: np.ndarray, model_dir: str) -> str:
#     """
#     Save arrays so you can re-plot later without running the model again.
#     """
#     graphs_dir = ensure_graphs_dir(model_dir)
#     npz_path = os.path.join(graphs_dir, "last_test_forecast.npz")
#     np.savez(npz_path, y_true=y_true, y_pred=y_pred)
#     return npz_path

# def plot_avp_from_saved(model_dir: str, show: bool = False) -> str:
#     """
#     Recreate a timestamped AVP plot from saved arrays (no model inference).
#     """
#     graphs_dir = ensure_graphs_dir(model_dir)
#     npz_path = os.path.join(graphs_dir, "last_test_forecast.npz")
#     if not os.path.exists(npz_path):
#         raise FileNotFoundError(f"Missing {npz_path}. Train once to create it.")
#     data = np.load(npz_path)
#     y_true, y_pred = data["y_true"], data["y_pred"]
#     return save_actual_vs_predicted(y_true, y_pred, model_dir, show=show)



# app/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_graphs_dir(model_dir: str) -> str:
    gdir = os.path.join(model_dir, "graphs")
    os.makedirs(gdir, exist_ok=True)
    return gdir

def _to_datetime_index(ts_like):
    """Accept pandas.DatetimeIndex / Series / array of strings or Timestamps; return DatetimeIndex."""
    if isinstance(ts_like, pd.DatetimeIndex):
        return ts_like
    return pd.to_datetime(np.asarray(ts_like))

def save_series_avp_with_time(ts, y_true_series, y_pred_series, model_dir: str, title="Predicted vs Actual — Test Set", show=False):
    """
    ts: sequence of timestamps (len N)
    y_true_series, y_pred_series: arrays length N (aligned 1-step-ahead or any aligned series)
    Saves a full time-series AVP plot with smart date ticks.
    """
    graphs_dir = ensure_graphs_dir(model_dir)
    fname = f"{_ts()}_avp_timeseries.png"
    path = os.path.join(graphs_dir, fname)

    ts_idx = _to_datetime_index(ts)

    plt.figure(figsize=(14, 6))
    plt.plot(ts_idx, y_true_series, label="Actual", linewidth=2)
    plt.plot(ts_idx, y_pred_series, label="Predicted", linewidth=2, linestyle="-")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    return path

def save_last48_avp(ts, y_true_series, y_pred_series, model_dir: str, title="Predicted vs Actual (Last 48 Hours)", show=False):
    """
    Zoomed view of the last 48 samples.
    """
    graphs_dir = ensure_graphs_dir(model_dir)
    fname = f"{_ts()}_avp_last48.png"
    path = os.path.join(graphs_dir, fname)

    ts_idx = _to_datetime_index(ts)
    last_2_days = min(48, len(ts_idx))
    ts_last = ts_idx[-last_2_days:]
    y_last_t = np.asarray(y_true_series)[-last_2_days:]
    y_last_p = np.asarray(y_pred_series)[-last_2_days:]

    plt.figure(figsize=(12, 6))
    plt.plot(ts_last, y_last_t, label="Actual (Last 48)", marker="o", markersize=4, linewidth=1)
    plt.plot(ts_last, y_last_p, label="Predicted (Last 48)", marker="x", markersize=4, linewidth=1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    return path

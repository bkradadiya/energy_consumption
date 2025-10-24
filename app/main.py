# main.py
import argparse
import json
import os
import pandas as pd
from app.config import get_final_data
from app.save_model import train_and_save
from app.load_model import predict_iterative_hours_to_excel
# from app.load_model import predict_next_24

# from app.plots import save_training_curves

def main():
    ap = argparse.ArgumentParser(description="Energy forecasting runner")
    ap.add_argument("cmd", choices=["data", "train", "predict", "graph", "graph_avp"], help="What to run")
    ap.add_argument("--timesteps", type=int, default=240, help="Lookback window for sequences")
    ap.add_argument("--output_steps", type=int, default=24, help="How many future hours to predict")
    ap.add_argument("--model-dir", type=str, default="models", help="Where to save/load model")
    ap.add_argument("--plot", action="store_true", help="Show plots on screen (train/graph)")
    ap.add_argument("--hours", type=int, default=24, help="Number of future hours to predict (e.g., 24, 48)")
    ap.add_argument("--future-csv", type=str, required=False, help="CSV with future weather/time exogenous features")

    args = ap.parse_args()

    if args.cmd == "data":
        df = get_final_data()
        print(df.head())
        print("Rows:", len(df))
        return

    # if args.cmd == "train":
    #     df = get_final_data()
    #     info = train_and_save(df, timesteps=args.timesteps, out_dir=args.model_dir)
    #     print("✅ Trained & saved:", info)
    #     return

    if args.cmd == "train":
        df = get_final_data()
        info = train_and_save(df, timesteps=args.timesteps, output_steps=args.output_steps, out_dir=args.model_dir)
        print("✅ Trained & saved:", info)
        return
    
    if args.cmd == "graph_avp":
        out_path = plot_avp_from_saved(args.model_dir, show=args.plot)
        print(f"✅ Saved AVP graph: {out_path}")
        return

    
    # if args.cmd == "predict":
    #     df_hist = get_final_data()  # your existing loader
    #     if not args.future_csv:
    #         raise SystemExit("Please provide --future-csv <path> for future exogenous data.")
    #     fut = pd.read_csv(args.future_csv, parse_dates=["timestamp"])
    #     forecast_df, xlsx_path = predict_iterative_hours_to_excel(
    #         history_df=df_hist,
    #         future_df=fut,
    #         model_dir=args.model_dir,
    #         hours=args.hours,
    #         excel_name_prefix="forecast"
    #     )
    #     print("✅ Saved Excel:", xlsx_path)
    #     print(forecast_df.head())
    #     return

    if args.cmd == "predict":
        df_hist = get_final_data()

        # --- set default path here ---
        # --- set default path here ---
        DEFAULT_FUTURE_CSV = r"E:\Web Project\SIPE\energy_consumption\KA_Gunterblum_weather_next2days.csv"

        # use CLI argument if provided, else fallback
        future_csv_path = args.future_csv or DEFAULT_FUTURE_CSV

        # safety check
        if not os.path.exists(future_csv_path):
            raise FileNotFoundError(f"Future CSV not found at: {future_csv_path}")

        fut = pd.read_csv(future_csv_path, parse_dates=["timestamp"])
        forecast_df, xlsx_path = predict_iterative_hours_to_excel(
            history_df=df_hist,
            future_df=fut,
            model_dir=args.model_dir,
            hours=args.hours,
            excel_name_prefix="forecast"
        )
        print("✅ Saved Excel:", xlsx_path)


    if args.cmd == "graph":
        # Re-render training curves from saved history.json (no retrain)
        hist_path = os.path.join(args.model_dir, "graphs", "history.json")
        if not os.path.exists(hist_path):
            raise FileNotFoundError(f"No history.json found under {args.model_dir}/graphs")
        with open(hist_path, "r") as f:
            hist = json.load(f)
        save_training_curves(hist, args.model_dir, show=args.plot)
        print(f"✅ Graphs saved under {os.path.join(args.model_dir, 'graphs')}")
        return
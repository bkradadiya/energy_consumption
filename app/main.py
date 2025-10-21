# main.py
import argparse
from app.config import get_final_data
from app.save_model import train_and_save
# from app.load_model import predict_next_24

def main():
    ap = argparse.ArgumentParser(description="Energy forecasting runner")
    ap.add_argument("cmd", choices=["data", "train", "predict"], help="What to run")
    ap.add_argument("--timesteps", type=int, default=240, help="Lookback window for sequences")
    ap.add_argument("--model-dir", type=str, default="models", help="Where to save/load model")
    args = ap.parse_args()

    if args.cmd == "data":
        df = get_final_data()
        print(df.head())
        print("Rows:", len(df))
        return

    if args.cmd == "train":
        df = get_final_data()
        info = train_and_save(df, timesteps=args.timesteps, out_dir=args.model_dir)
        print("✅ Trained & saved:", info)
        return

    if args.cmd == "predict":
        df = get_final_data()
        preds = predict_next_24(df, model_dir=args.model_dir)
        print("✅ Next 24 predictions:")
        for i, v in enumerate(preds, 1):
            print(f"+{i:02d}h -> {v:.4f}")


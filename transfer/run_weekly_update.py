import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run weekly FPL prediction update for a target GW.")
    parser.add_argument("--pred-gw", type=int, required=True, help="Target gameweek to predict.")
    parser.add_argument("--pred-year", type=int, default=26, help="Prediction season suffix, e.g. 26 for 2025-26.")
    parser.add_argument("--model", choices=["elasticnet", "ridge", "lasso", "linear", "xgboost"], default="elasticnet")
    parser.add_argument("--skip-eval", action="store_true", help="Skip train/test RMSE printout.")
    return parser.parse_args()


def main():
    args = parse_args()
    transfer_dir = Path(__file__).resolve().parent
    predict_script = transfer_dir / "predict_gw_scores.py"

    cmd = [
        sys.executable,
        str(predict_script),
        "--pred-gw",
        str(args.pred_gw),
        "--pred-year",
        str(args.pred_year),
        "--model",
        args.model,
    ]
    if args.skip_eval:
        cmd.append("--skip-eval")

    subprocess.run(cmd, cwd=transfer_dir, check=True)

    print(f"Predictions refreshed for GW {args.pred_gw}.")
    print("Open transfer/outputs and use latest_gw_tools.load_latest_outputs() in your notebook.")


if __name__ == "__main__":
    main()

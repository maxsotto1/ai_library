import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from codebase.models.gmlp_class import gMLP_pipeline
from codebase.models.xgb_class import XGBoost_pipeline
from codebase.helpers.pivot_df import pivot_df 

saved_files_dir = parent_dir / "saved_files"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

pipeline_type = config["pipeline_type"]
window = config["window"]
targets = config["prediction_target"]
cols_to_drop = config.get("cols_to_drop", [])

if pipeline_type == "gmlp":
    pipeline = gMLP_pipeline()
    pipeline.model = torch.load(saved_files_dir / "trained_model.pth", map_location="cpu")
    pipeline.scaler_x = pickle.load(open(saved_files_dir / "scaler_x.pkl", "rb"))
    pipeline.scaler_y = pickle.load(open(saved_files_dir / "scaler_y.pkl", "rb"))
    pipeline.clipping_min = pickle.load(open(saved_files_dir / "clipping_min.pkl", "rb"))
    pipeline.clipping_max = pickle.load(open(saved_files_dir / "clipping_max.pkl", "rb"))
elif pipeline_type == "xgb":
    pipeline = XGBoost_pipeline()
    pipeline.model = pickle.load(open(saved_files_dir / "trained_model.pkl", "rb"))
    pipeline.scaler_x = pickle.load(open(saved_files_dir / "scaler_x.pkl", "rb"))
    pipeline.scaler_y = pickle.load(open(saved_files_dir / "scaler_y.pkl", "rb"))
    pipeline.clipping_min = pickle.load(open(saved_files_dir / "clipping_min.pkl", "rb"))
    pipeline.clipping_max = pickle.load(open(saved_files_dir / "clipping_max.pkl", "rb"))
else:
    raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

df = pd.read_parquet(config["parquet_path"])
df = pivot_df(df)
df = df.dropna()
last_window = df.iloc[-window:]

window_tensor = pipeline.preprocess_inference(
    last_window,
    targets,
    n_past=window,
    exclude_columns=cols_to_drop,
)
predictions = pipeline.infer(window_tensor)

print(predictions)
